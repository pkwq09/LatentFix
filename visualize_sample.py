"""Batch-render motion samples from folders of .np files (source / target / generated).

Usage:
    python visualize_sample.py --path <motion.npy>
    python visualize_sample.py --dir <folder_with_npy>
    python visualize_sample.py --dir <folder> --ghost --ghost_frames 5
    python visualize_sample.py --dir <folder> --smpl_path <SMPL_models_path>
    python visualize_sample.py --dir <folder> --no-skip

Example:
    python visualize_sample.py --dir experiments/vae_motionfix/.../ld_txt-1.0_ld_mot-1.0
"""

from src.render.mesh_viz import render_motion
from src.model.utils.tools import pack_to_render
from src.render.video import get_offscreen_renderer
from src.utils.art_utils import color_map
import os
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import re
import logging
from typing import Optional, Sequence

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.getLogger('aitviewer').setLevel(logging.WARNING)
logging.getLogger('moderngl').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)


def parse_color_arg(arg: Optional[str], fallback: Sequence[float]) -> np.ndarray:
    """Parse comma-separated RGBA string to [0,1] floats; supports 3 or 4 channels."""
    if arg is None:
        return np.array(fallback, dtype=float)
    parts = [p.strip() for p in arg.split(',') if p.strip() != '']
    if len(parts) not in (3, 4):
        raise ValueError("Color must be 3 or 4 comma-separated floats, e.g. 0.6,0.8,1 or 0.6,0.8,1,1")
    vals = [float(p) for p in parts]
    if any(v > 1.0 for v in vals):
        vals = [v / 255.0 for v in vals]
    if len(vals) == 3:
        vals.append(float(fallback[3]))
    return np.array(vals, dtype=float)


def extract_sample_id(filename: str) -> str:
    """Extract 6-digit sample id from filename, or None."""
    match = re.match(r'^(\d{6})', filename)
    if match:
        return match.group(1)
    return None


def guess_motion_type_from_name(name: str) -> str:
    """Infer motion role from filename: *_source, *_target, else generated."""
    stem = Path(name).stem
    if stem.endswith("_source"):
        return "source"
    if stem.endswith("_target"):
        return "target"
    return "generated"


def load_motion_from_npy(npy_path: Path):
    """Load motion array [T, D] from a .npy file."""
    try:
        data = np.load(npy_path, allow_pickle=True)
    except Exception as e:
        raise ValueError(f"Failed to load file {npy_path}: {e}")

    if isinstance(data, np.ndarray):
        if data.ndim == 0:
            try:
                item = data.item()
                if isinstance(item, dict):
                    if 'pose' in item:
                        motion = item['pose']
                        if isinstance(motion, np.ndarray):
                            return motion
                        else:
                            return np.array(motion)
                    else:
                        raise ValueError(f"No 'pose' key in dict: {npy_path}, keys: {list(item.keys())}")
                elif isinstance(item, np.ndarray):
                    return item
                else:
                    return np.array(item)
            except Exception as e:
                raise ValueError(f"Cannot extract motion from 0-dim array: {npy_path}, error: {e}")
        else:
            return data

    if hasattr(data, 'item'):
        try:
            item = data.item()
            if isinstance(item, dict):
                if 'pose' in item:
                    motion = item['pose']
                    if isinstance(motion, np.ndarray):
                        return motion
                    else:
                        return np.array(motion)
                else:
                    raise ValueError(f"No 'pose' key in dict: {npy_path}, keys: {list(item.keys())}")
            elif isinstance(item, np.ndarray):
                return item
            else:
                return np.array(item)
        except (AttributeError, ValueError) as e:
            pass

    if isinstance(data, np.ndarray):
        return data
    else:
        try:
            return np.array(data)
        except Exception as e:
            raise ValueError(f"Failed to load motion from {npy_path}: {e}, type: {type(data)}")


def render_single_motion(
    renderer,
    motion_data: np.ndarray,
    output_path: Path,
    motion_type: str,
    skip_existing: bool = True,
    ghost: bool = False,
    ghost_frames: int = 5,
    ghost_start_color: Optional[str] = None,
    ghost_end_color: Optional[str] = None
):
    """Render one motion to video; optional multi-frame ghost overlay PNG."""
    try:
        if not isinstance(motion_data, np.ndarray):
            motion_data = np.array(motion_data)

        if motion_data.ndim == 0:
            raise ValueError("Motion data is 0-dim (scalar); cannot render")
        elif motion_data.ndim == 1:
            motion_data = motion_data.reshape(1, -1)
            logger.warning(f"Motion was 1-D; reshaped to [1, {motion_data.shape[1]}]")
        elif motion_data.ndim > 2:
            raise ValueError(f"Motion has too many dims: {motion_data.ndim}, expected 2 [T, D]")

        if motion_data.shape[1] < 3:
            raise ValueError(f"Feature dim too small: {motion_data.shape[1]}, need at least 3 (translation)")

        smpl_params = pack_to_render(
            trans=torch.from_numpy(motion_data[..., :3]),
            rots=torch.from_numpy(motion_data[..., 3:])
        )

        if motion_type == 'source':
            color = color_map['source']
        elif motion_type == 'target':
            color = color_map['target']
        else:
            color = color_map['generation']

        video_file = output_path.with_suffix('.mp4')
        if not (skip_existing and video_file.exists()):
            video_path = render_motion(
                renderer,
                smpl_params,
                pose_repr='aa',
                filename=str(output_path),
                text_for_vid=None,
                color=color
            )
        else:
            video_path = str(video_file)

        if ghost:
            total_frames = motion_data.shape[0]
            frames_to_pick = min(ghost_frames, total_frames)
            idxs = np.linspace(0, total_frames - 1, frames_to_pick, dtype=int)
            base_color = np.array(color)
            start_col = parse_color_arg(ghost_start_color, base_color)
            end_col = parse_color_arg(ghost_end_color, base_color)
            factors = np.linspace(0.0, 1.0, frames_to_pick)
            ghost_colors = [tuple(start_col * (1 - f) + end_col * f) for f in factors]

            ghost_datums = []
            for fi in idxs:
                frame = motion_data[fi]
                packed = pack_to_render(
                    trans=torch.from_numpy(frame[:3]).unsqueeze(0),
                    rots=torch.from_numpy(frame[3:]).unsqueeze(0)
                )
                ghost_datums.append(packed)

            ghost_base = output_path.with_name(output_path.name + "_ghost").with_suffix('')
            ghost_file = render_motion(
                renderer,
                ghost_datums,
                pose_repr='aa',
                filename=str(ghost_base),
                text_for_vid=None,
                color=ghost_colors
            )
            logger.info(f"Ghost image saved: {ghost_file}")

        return video_path
    except Exception as e:
        logger.error(f"Render failed {output_path}: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None


def render_pair_motion(
    renderer,
    motions: Sequence[np.ndarray],
    motion_types: Sequence[str],
    output_path: Path,
    skip_existing: bool = True,
    ghost: bool = False,
    ghost_frames: int = 5,
    ghost_start_color: Optional[str] = None,
    ghost_end_color: Optional[str] = None
):
    """Render two motions side-by-side with distinct colors."""
    try:
        if len(motions) != 2 or len(motion_types) != 2:
            raise ValueError("Need exactly two motions and two types for paired render")

        min_len = min(m.shape[0] for m in motions)
        if min_len <= 0:
            raise ValueError("Empty motion length; cannot render pair")
        motions = [np.asarray(m[:min_len]) for m in motions]

        packed = []
        for m in motions:
            if m.ndim == 1:
                m = m.reshape(1, -1)
            if m.ndim != 2 or m.shape[1] < 3:
                raise ValueError(f"Invalid motion shape {m.shape}, expected [T, D>=3]")
            packed.append(
                pack_to_render(
                    trans=torch.from_numpy(m[..., :3]),
                    rots=torch.from_numpy(m[..., 3:])
                )
            )

        base_color = {
            'source': (1.0, 0.2, 0.2, 1.0),
            'target': (0.2, 1.0, 0.2, 1.0),
            'generated': (1.0, 0.85, 0.0, 1.0)
        }
        colors = []
        for t in motion_types:
            colors.append(base_color.get(t, color_map['generation']))

        video_file = output_path.with_suffix('.mp4')
        if not (skip_existing and video_file.exists()):
            video_path = render_motion(
                renderer,
                packed,
                pose_repr='aa',
                filename=str(output_path),
                text_for_vid=None,
                color=colors
            )
        else:
            video_path = str(video_file)

        if ghost:
            ghost_datums = []
            ghost_colors = []

            def ghost_palette(t: str):
                if t == 'source':
                    return (223/255.0, 92/255.0, 100/255.0, 1.0), (216/255.0, 67/255.0, 86/255.0, 1.0)
                if t == 'target':
                    return (124/255.0, 213/255.0, 149/255.0, 1.0), (0/255.0, 199/255.0, 118/255.0, 1.0)
                return (209/255.0, 162/255.0, 70/255.0, 1.0), (188/255.0, 133/255.0, 33/255.0, 1.0)

            for m, t in zip(motions, motion_types):
                total_frames = m.shape[0]
                frames_to_pick = min(ghost_frames, total_frames)
                idxs = np.linspace(0, total_frames - 1, frames_to_pick, dtype=int)
                default_start, default_end = ghost_palette(t)
                start_col = parse_color_arg(ghost_start_color, np.array(default_start))
                end_col = parse_color_arg(ghost_end_color, np.array(default_end))
                factors = np.linspace(0.0, 1.0, frames_to_pick)
                for fi, f in zip(idxs, factors):
                    frame = m[fi]
                    packed_frame = pack_to_render(
                        trans=torch.from_numpy(frame[:3]).unsqueeze(0),
                        rots=torch.from_numpy(frame[3:]).unsqueeze(0)
                    )
                    ghost_datums.append(packed_frame)
                    ghost_colors.append(tuple(np.array(start_col) * (1 - f) + np.array(end_col) * f))

            ghost_base = output_path.with_name(output_path.name + "_ghost").with_suffix('')
            ghost_file = render_motion(
                renderer,
                ghost_datums,
                pose_repr='aa',
                filename=str(ghost_base),
                text_for_vid=None,
                color=ghost_colors
            )
            logger.info(f"Comparison ghost image saved: {ghost_file}")

        return video_path
    except Exception as e:
        logger.error(f"Paired render failed {output_path}: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None


def render_sample_from_dir(
    npy_dir: Path,
    sample_id: str,
    renderer,
    skip_existing: bool = True,
    ghost: bool = False,
    ghost_frames: int = 5,
    ghost_start_color: Optional[str] = None,
    ghost_end_color: Optional[str] = None,
    compare_mode: Optional[str] = None,
    render_individual: bool = True
):
    """Render all npy files for one sample id under npy_dir."""
    rendered_dir = npy_dir / "rendered" / sample_id
    rendered_dir.mkdir(exist_ok=True, parents=True)

    results = {
        'sample_id': sample_id,
        'rendered_dir': rendered_dir,
        'rendered_files': []
    }

    file_types = [
        ('source', f'{sample_id}_source.npy', 'source.mp4'),
        ('target', f'{sample_id}_target.npy', 'target.mp4'),
        ('generated', f'{sample_id}.npy', 'generated.mp4')
    ]

    if render_individual:
        for motion_type, npy_filename, video_filename in file_types:
            npy_path = npy_dir / npy_filename
            video_path = rendered_dir / video_filename

            if npy_path.exists():
                try:
                    motion_data = load_motion_from_npy(npy_path)
                    rendered_video = render_single_motion(
                        renderer,
                        motion_data,
                        video_path.with_suffix(''),
                        motion_type,
                        skip_existing=skip_existing,
                        ghost=ghost,
                        ghost_frames=ghost_frames,
                        ghost_start_color=ghost_start_color,
                        ghost_end_color=ghost_end_color
                    )

                    if rendered_video:
                        results['rendered_files'].append(motion_type)
                except Exception as e:
                    logger.error(f"Sample {sample_id} {motion_type} render error: {e}")

    if compare_mode:
        try:
            if compare_mode == 's_t':
                pair = [('source', f'{sample_id}_source.npy'), ('target', f'{sample_id}_target.npy')]
                out_name = 'compare_st.mp4'
            elif compare_mode == 's_g':
                pair = [('source', f'{sample_id}_source.npy'), ('generated', f'{sample_id}.npy')]
                out_name = 'compare_sg.mp4'
            elif compare_mode == 'g_t':
                pair = [('generated', f'{sample_id}.npy'), ('target', f'{sample_id}_target.npy')]
                out_name = 'compare_gt.mp4'
            else:
                raise ValueError("compare_mode must be one of: s_t, s_g, g_t")

            npy_paths = []
            motion_types = []
            for m_type, fname in pair:
                p = npy_dir / fname
                if not p.exists():
                    raise FileNotFoundError(f"Missing comparison file: {fname}")
                npy_paths.append(p)
                motion_types.append(m_type)

            motions = [load_motion_from_npy(p) for p in npy_paths]
            rendered_video = render_pair_motion(
                renderer,
                motions,
                motion_types,
                rendered_dir / Path(out_name).with_suffix(''),
                skip_existing=skip_existing,
                ghost=ghost,
                ghost_frames=ghost_frames,
                ghost_start_color=ghost_start_color,
                ghost_end_color=ghost_end_color
            )
            if rendered_video:
                results['rendered_files'].append(f'compare_{compare_mode}')
        except Exception as e:
            logger.error(f"Sample {sample_id} comparison render error: {e}")

    return results


def batch_render_from_dir(
    npy_dir: Path,
    smpl_path: str = './data/body_models',
    skip_existing: bool = True,
    ghost: bool = False,
    ghost_frames: int = 5,
    ghost_start_color: Optional[str] = None,
    ghost_end_color: Optional[str] = None,
    compare_mode: Optional[str] = None
):
    """Batch-render all samples in a directory of .npy files."""
    npy_dir = Path(npy_dir)

    if not npy_dir.exists():
        raise ValueError(f"Directory does not exist: {npy_dir}")

    if not npy_dir.is_dir():
        raise ValueError(f"Path is not a directory: {npy_dir}")

    npy_files = list(npy_dir.glob("*.npy"))

    if len(npy_files) == 0:
        logger.warning(f"No .npy files found in {npy_dir}")
        return

    logger.info(f"Found {len(npy_files)} .npy files")

    sample_ids = set()
    for npy_file in npy_files:
        sample_id = extract_sample_id(npy_file.name)
        if sample_id:
            sample_ids.add(sample_id)
        else:
            logger.debug(f"Could not extract id from filename: {npy_file.name}")

    if len(sample_ids) == 0:
        logger.warning("Could not extract any sample ids from filenames")
        return

    logger.info(f"Extracted {len(sample_ids)} unique sample ids; starting batch render...")

    try:
        renderer = get_offscreen_renderer(smpl_path)
    except Exception as e:
        logger.error(f"Renderer init failed: {e}")
        raise

    success_count = 0
    skip_count = 0
    fail_count = 0
    total_files_rendered = 0

    for sample_id in tqdm(sorted(sample_ids), desc="Rendering"):
        try:
            result = render_sample_from_dir(
                npy_dir,
                sample_id,
                renderer,
                skip_existing=skip_existing,
                ghost=ghost,
                ghost_frames=ghost_frames,
                ghost_start_color=ghost_start_color,
                ghost_end_color=ghost_end_color,
                compare_mode=compare_mode,
                render_individual=compare_mode is None
            )

            if len(result['rendered_files']) > 0:
                success_count += 1
                total_files_rendered += len(result['rendered_files'])
            else:
                fail_count += 1
                logger.warning(f"Sample {sample_id}: no files rendered successfully")
        except Exception as e:
            fail_count += 1
            logger.error(f"Sample {sample_id} render failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())

    print("\n" + "="*60)
    print("Batch rendering finished.")
    print(f"Succeeded: {success_count} samples")
    print(f"Failed: {fail_count} samples")
    print(f"Total video outputs: {total_files_rendered}")
    print(f"Output directory: {npy_dir / 'rendered'}")
    print("="*60)


def render_single_file(path_to_motion: str, output_path: str = None, ghost: bool = False, ghost_frames: int = 5,
                      ghost_start_color: Optional[str] = None, ghost_end_color: Optional[str] = None,
                      compare_mode: Optional[str] = None):
    """Render a single .npy file (CLI single-file mode)."""
    npy_path = Path(path_to_motion)

    if not npy_path.exists():
        raise FileNotFoundError(f"File not found: {path_to_motion}")

    renderer = get_offscreen_renderer('./data/body_models')

    motion_data = load_motion_from_npy(npy_path)

    motion_type = guess_motion_type_from_name(npy_path.name)

    npy_dir = npy_path.parent
    sample_id = extract_sample_id(npy_path.name)

    if output_path is None:
        if sample_id:
            rendered_dir = npy_dir / "rendered" / sample_id
            rendered_dir.mkdir(exist_ok=True, parents=True)
            output_path = rendered_dir / motion_type
        else:
            output_path = npy_dir / f"{npy_path.stem}_{motion_type}"
    else:
        output_path = Path(output_path).with_suffix('')
        if motion_type not in output_path.stem:
            output_path = output_path.with_name(output_path.stem + f"_{motion_type}")

    if compare_mode:
        if sample_id is None:
            raise ValueError("compare_mode requires a 6-digit sample id in the filename")
        if compare_mode == 's_t':
            pair = [('source', f'{sample_id}_source.npy'), ('target', f'{sample_id}_target.npy')]
            out_name = 'compare_st'
        elif compare_mode == 's_g':
            pair = [('source', f'{sample_id}_source.npy'), ('generated', f'{sample_id}.npy')]
            out_name = 'compare_sg'
        elif compare_mode == 'g_t':
            pair = [('generated', f'{sample_id}.npy'), ('target', f'{sample_id}_target.npy')]
            out_name = 'compare_gt'
        else:
            raise ValueError("compare_mode must be one of: s_t, s_g, g_t")

        npy_paths = []
        motion_types = []
        for m_type, fname in pair:
            p = npy_dir / fname
            if not p.exists():
                raise FileNotFoundError(f"Missing comparison file: {fname}")
            npy_paths.append(p)
            motion_types.append(m_type)
        motions = [load_motion_from_npy(p) for p in npy_paths]
        output_path = output_path.with_name(out_name)
        video_path = render_pair_motion(
            renderer,
            motions,
            motion_types,
            output_path,
            skip_existing=False,
            ghost=ghost,
            ghost_frames=ghost_frames,
            ghost_start_color=ghost_start_color,
            ghost_end_color=ghost_end_color
        )
    else:
        video_path = render_single_motion(
            renderer,
            motion_data,
            output_path,
            motion_type=motion_type,
            skip_existing=False,
            ghost=ghost,
            ghost_frames=ghost_frames,
            ghost_start_color=ghost_start_color,
            ghost_end_color=ghost_end_color
        )

    if video_path:
        logger.info(f"Render done: {video_path}")
    else:
        logger.error("Render failed")


def main():
    parser = argparse.ArgumentParser(
        description="Render motion samples to video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize_sample.py --path sample.npy
  python visualize_sample.py --dir experiments/.../ld_txt-1.0_ld_mot-1.0
  python visualize_sample.py --dir experiments/.../ld_txt-1.0_ld_mot-1.0 --no-skip
  python visualize_sample.py --dir <folder> --smpl_path ./data/body_models
        """
    )
    parser.add_argument(
        '--path',
        type=str,
        help='Single .npy file (single-file mode)'
    )
    parser.add_argument(
        '--dir',
        type=str,
        help='Folder containing .npy files (batch mode)'
    )
    parser.add_argument(
        '--smpl_path',
        type=str,
        default='./data/body_models',
        help='SMPL model path (default: ./data/body_models)'
    )
    parser.add_argument(
        '--no-skip',
        action='store_true',
        help='Re-render even if output video already exists'
    )
    parser.add_argument(
        '--ghost',
        action='store_true',
        help='Also write ghost overlay images (multi-frame composite)'
    )
    parser.add_argument(
        '--ghost_frames',
        type=int,
        default=5,
        help='Number of frames to sample for ghost image (default 5)'
    )
    parser.add_argument(
        '--ghost_start_color',
        type=str,
        default=None,
        help='Ghost gradient start color "r,g,b[,a]" (0-1 or 0-255)'
    )
    parser.add_argument(
        '--ghost_end_color',
        type=str,
        default=None,
        help='Ghost gradient end color "r,g,b[,a]" (0-1 or 0-255)'
    )
    parser.add_argument(
        '--compare_mode',
        type=str,
        choices=['s_t', 's_g', 'g_t'],
        default=None,
        help='Side-by-side: s_t=source vs target, s_g=source vs generated, g_t=generated vs target'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose logging'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

    if args.dir:
        batch_render_from_dir(
            npy_dir=args.dir,
            smpl_path=args.smpl_path,
            skip_existing=not args.no_skip,
            ghost=args.ghost,
            ghost_frames=args.ghost_frames,
            ghost_start_color=args.ghost_start_color,
            ghost_end_color=args.ghost_end_color,
            compare_mode=args.compare_mode
        )
    elif args.path:
        render_single_file(args.path,
                           ghost=args.ghost,
                           ghost_frames=args.ghost_frames,
                           ghost_start_color=args.ghost_start_color,
                           ghost_end_color=args.ghost_end_color,
                           compare_mode=args.compare_mode)
    else:
        parser.print_help()
        print("\nError: specify --path or --dir")


if __name__ == "__main__":
    try:
        os.system("Xvfb :12 -screen 1 640x480x24 &")
        os.environ['DISPLAY'] = ":12"
    except:
        pass

    main()
