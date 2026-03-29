"""VAE FID evaluation (MLD-style latent sampling and latent-space FID / diversity).

Usage:
    python compute_vae_fid.py checkpoint=<vae_checkpoint_path> data=<data_config>
"""

from omegaconf import DictConfig
import hydra
import torch
from tqdm import tqdm
from pathlib import Path
import numpy as np
from hydra.utils import instantiate
import pytorch_lightning as pl
import logging
import src.launch.prepare  # noqa: custom resolvers (code_path, working_path, etc.)

log = logging.getLogger(__name__)


@hydra.main(config_path="configs", version_base="1.2", config_name="compute_vae_fid")
def compute_vae_fid(cfg: DictConfig):
    """Compute FID-style metrics for VAE generations (latent sampling, aligned with MLD)."""
    if cfg.get('device', 'auto') == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif cfg.get('device', 'auto') == 'cuda':
        if not torch.cuda.is_available():
            log.warning("CUDA not available, falling back to CPU")
            device = torch.device('cpu')
        else:
            gpu_id = cfg.get('gpu_id', None)
            if gpu_id is not None:
                device = torch.device(f'cuda:{int(gpu_id)}')
            else:
                device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    log.info(f"Using device: {device}")

    log.info(f"Loading VAE model from: {cfg.checkpoint}")
    try:
        from src.model.base_diffusion import MD
        model = MD.load_from_checkpoint(cfg.checkpoint, map_location='cpu')
    except Exception as e:
        log.error(f"Failed to load model: {e}")
        raise

    if not hasattr(model, 'vae') or model.vae is None:
        raise ValueError("Model does not have a VAE. Please check the checkpoint.")

    if not hasattr(model, 'input_feats'):
        raise ValueError("Model does not have 'input_feats' attribute. This is required for data preprocessing.")

    log.info(f"Moving model to device: {device}")
    model.eval()
    model = model.to(device)

    if hasattr(model, 'stats') and model.stats is not None:
        def move_stats_to_device(stats_dict, target_device):
            """Recursively move tensors in a stats dict to target_device."""
            if isinstance(stats_dict, dict):
                for key, value in stats_dict.items():
                    if isinstance(value, dict):
                        move_stats_to_device(value, target_device)
                    elif torch.is_tensor(value):
                        stats_dict[key] = value.to(target_device)
                    elif isinstance(value, np.ndarray):
                        stats_dict[key] = torch.from_numpy(value).float().to(target_device)

        move_stats_to_device(model.stats, device)

        def verify_stats_device(stats_dict, target_device):
            """Warn if any tensor in stats is not on target_device."""
            if isinstance(stats_dict, dict):
                for key, value in stats_dict.items():
                    if isinstance(value, dict):
                        verify_stats_device(value, target_device)
                    elif torch.is_tensor(value):
                        if value.device != target_device:
                            log.warning(f"Stats[{key}] is on {value.device}, expected {target_device}")

        verify_stats_device(model.stats, device)
        log.info(f"Stats moved to device: {device}")

    log.info("Model loaded successfully")

    vae_latent_dim = model.vae.latent_dim
    if hasattr(model.vae, 'latent_size'):
        vae_latent_size = model.vae.latent_size
    else:
        log.info("Latent size not found in VAE attributes, will determine from encoding...")
        vae_latent_size = None

    log.info(f"VAE latent_dim: {vae_latent_dim}")
    if vae_latent_size:
        log.info(f"VAE latent_size: {vae_latent_size}")

    log.info(f"Loading dataset: {cfg.data.get('dataname', 'motionfix')}")
    split = cfg.get('split', 'val')
    log.info(f"Using {split} split")

    try:
        data_module = instantiate(cfg.data)
        stage = 'test' if split == 'test' else 'validate'
        data_module.setup(stage=stage)
    except Exception as e:
        log.error(f"Failed to load dataset: {e}")
        raise

    if split == 'val':
        dataloader = data_module.val_dataloader()
        dataset = data_module.dataset['val']
    elif split == 'test':
        dataloader = data_module.test_dataloader()
        dataset = data_module.dataset['test']
    else:
        raise ValueError(f"Unknown split: {split}. Must be 'val' or 'test'")

    from src.model.metrics.mr import MRMetrics
    from src.model.metrics.vae_fid import VAEFIDMetrics

    mr_metrics = MRMetrics(njoints=22, jointstype="smplnh", force_in_meter=True, align_root=True, dist_sync_on_step=False)
    vae_fid_metrics = VAEFIDMetrics(dist_sync_on_step=False, diversity_times=300)

    log.info("Collecting ground truth motions and computing reconstruction metrics...")
    gt_lengths = []
    gt_keyids = []
    gt_feats_list = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collecting GT & computing reconstruction"):
            processed_batch = model.norm_and_cat(batch, model.input_feats)

            if 'target' not in processed_batch:
                continue

            target_feats = processed_batch['target'].to(device)
            target_lengths = batch.get('length_target', batch.get('length'))
            if isinstance(target_lengths, torch.Tensor):
                target_lengths = target_lengths.tolist()
            elif not isinstance(target_lengths, list):
                target_lengths = [target_lengths] if target_lengths is not None else []

            if len(target_lengths) == 0:
                continue

            try:
                target_z, _ = model.encode_with_vae(target_feats, target_lengths)
                target_feats_rst = model.decode_with_vae(target_z, target_lengths)
                target_feats_bt = target_feats.permute(1, 0, 2)

                joints_rst = model.feats2joints(target_feats_rst, target_lengths)
                joints_ref = model.feats2joints(target_feats_bt, target_lengths)

                mr_metrics.update(joints_rst, joints_ref, target_lengths)

            except Exception as e:
                log.warning(f"Error computing reconstruction metrics for batch: {e}")
                import traceback
                log.debug(traceback.format_exc())

            target_feats_bt = target_feats.permute(1, 0, 2)
            for j, length in enumerate(target_lengths):
                gt_feats_list.append({
                    'feats': target_feats_bt[j, :length, :].cpu(),
                    'length': length
                })

            gt_lengths.extend(target_lengths)

            if 'keyid' in batch:
                if isinstance(batch['keyid'], list):
                    gt_keyids.extend(batch['keyid'])
                else:
                    gt_keyids.extend(batch['keyid'].tolist() if hasattr(batch['keyid'], 'tolist') else [batch['keyid']])

    if len(gt_lengths) == 0:
        raise ValueError("No ground truth lengths collected. Please check the data loader.")

    num_samples = cfg.get('num_samples', len(gt_lengths))
    if num_samples > len(gt_lengths):
        num_samples = len(gt_lengths)
        log.warning(f"Requested {cfg.get('num_samples')} samples but only {len(gt_lengths)} available. Using all {len(gt_lengths)} samples.")

    if num_samples < len(gt_lengths):
        indices = np.random.choice(len(gt_lengths), num_samples, replace=False)
        gt_lengths = [gt_lengths[i] for i in indices]
        gt_feats_list = [gt_feats_list[i] for i in indices]
        if len(gt_keyids) > 0:
            gt_keyids = [gt_keyids[i] for i in indices]

    log.info(f"Will generate {num_samples} motions for FID evaluation")

    log.info("Computing reconstruction metrics...")
    try:
        mr_metrics_dict = mr_metrics.compute(sanity_flag=False)
        log.info("Reconstruction metrics computed successfully")
    except Exception as e:
        log.error(f"Error computing reconstruction metrics: {e}")
        import traceback
        log.error(traceback.format_exc())
        mr_metrics_dict = None

    if vae_latent_size is None:
        log.info("Determining latent_size by encoding a sample...")
        sample_length = gt_lengths[0]
        dummy_z = torch.randn(1, 1, vae_latent_dim, device=device)
        try:
            _ = model.vae.decode(dummy_z, [sample_length])
            test_batch = next(iter(dataloader))
            processed_batch = model.norm_and_cat(test_batch, model.input_feats)
            if 'target' in processed_batch:
                test_feats = processed_batch['target'][:1, :1, :].to(device)
                test_lengths = [gt_lengths[0]]
                test_z, _ = model.encode_with_vae(test_feats, test_lengths)
                vae_latent_size = test_z.shape[0]
                log.info(f"Determined latent_size: {vae_latent_size}")
        except Exception as e:
            log.warning(f"Could not determine latent_size automatically: {e}")
            log.warning("Assuming latent_size=1. If this is incorrect, please specify in config.")
            vae_latent_size = 1

    log.info(f"Generating motions from latent space (latent_size={vae_latent_size}, latent_dim={vae_latent_dim})...")
    batch_size = cfg.get('gen_batch_size', 32)

    with torch.no_grad():
        for i in tqdm(range(0, num_samples, batch_size), desc="Generating motions & encoding latents"):
            batch_end = min(i + batch_size, num_samples)
            batch_lengths = gt_lengths[i:batch_end]
            batch_size_actual = len(batch_lengths)

            z = torch.randn(vae_latent_size, batch_size_actual, vae_latent_dim, device=device)

            try:
                gen_feats = model.vae.decode(z, batch_lengths)

                gen_z, _ = model.encode_with_vae(gen_feats, batch_lengths)

                gt_feats_batch = []
                gt_lengths_batch = []
                for j in range(i, batch_end):
                    gt_feats_batch.append(gt_feats_list[j]['feats'])
                    gt_lengths_batch.append(gt_feats_list[j]['length'])

                max_len = max(gt_lengths_batch)
                gt_feats_padded = torch.zeros(max_len, batch_size_actual, model.nfeats)
                for j, (feat, length) in enumerate(zip(gt_feats_batch, gt_lengths_batch)):
                    gt_feats_padded[:length, j, :] = feat

                gt_feats_seq = gt_feats_padded.to(device)
                gt_z, _ = model.encode_with_vae(gt_feats_seq, gt_lengths_batch)

                vae_fid_metrics.update(
                    gtmotion_embeddings=gt_z.permute(1, 0, 2),
                    lengths=gt_lengths_batch,
                    recmotion_embeddings=gen_z.permute(1, 0, 2)
                )

            except Exception as e:
                log.error(f"Error generating/encoding motions for batch {i//batch_size}: {e}")
                import traceback
                log.error(traceback.format_exc())
                continue

    log.info("Computing VAE FID and Diversity in latent space (aligned with MLD)...")
    try:
        fid_metrics_dict = vae_fid_metrics.compute(sanity_flag=False)
        log.info("VAE FID metrics computed successfully")
    except Exception as e:
        log.error(f"Error computing VAE FID metrics: {e}")
        import traceback
        log.error(traceback.format_exc())
        fid_metrics_dict = None

    print("\n" + "="*70)
    print("===== VAE EVALUATION RESULTS =====")
    print("="*70)

    if mr_metrics_dict is not None:
        print(f"\n📊 VAE Reconstruction Metrics:")
        print(f"  MPJPE:              {mr_metrics_dict['MPJPE'].item():.4f} mm")
        print(f"  PAMPJPE:            {mr_metrics_dict['PAMPJPE'].item():.4f} mm")
        if 'ACCEL' in mr_metrics_dict:
            print(f"  ACCEL:              {mr_metrics_dict['ACCEL'].item():.4f} mm/s²")
        elif 'ACCL' in mr_metrics_dict:
            print(f"  ACCL:               {mr_metrics_dict['ACCL'].item():.4f} mm/s²")
            print(f"  Number of samples:  {mr_metrics.count_seq.item()}")

    if fid_metrics_dict is not None:
        print(f"\n📈 VAE Generated Motion Metrics (Latent Space, aligned with MLD):")
        print(f"  FID:                {fid_metrics_dict['FID'].item():.4f}")
        print(f"  Diversity:          {fid_metrics_dict['Diversity'].item():.4f}")
        print(f"  GT Diversity:       {fid_metrics_dict['gt_Diversity'].item():.4f}")
        print(f"  Number of samples:  {vae_fid_metrics.count_seq.item()}")
    print("\n" + "="*70)
    print("💡 Interpretation:")
    if mr_metrics_dict is not None:
        print("  • MPJPE: Lower is better (mean per-joint position error)")
        print("  • PAMPJPE: Lower is better (Procrustes-aligned MPJPE)")
        print("  • ACCL: Lower is better (acceleration error)")
    if fid_metrics_dict is not None:
        print("  • FID: Lower is better (measures distribution quality in latent space)")
        print("  • Diversity: Measures diversity in latent space")
        print("  • These metrics evaluate VAE's generation quality (latent sampling)")
        print("  • Computed in latent space (aligned with MLD's UncondMetrics)")
    print("="*70 + "\n")

    if cfg.get('save_results', False):
        results_path = Path(cfg.get('results_dir', '.')) / 'vae_evaluation_results.json'
        results = {
            'split': split,
            'checkpoint': str(cfg.checkpoint),
            'latent_size': int(vae_latent_size),
            'latent_dim': int(vae_latent_dim)
        }

        if mr_metrics_dict is not None:
            results['reconstruction'] = {
                'MPJPE': float(mr_metrics_dict['MPJPE'].item()),
                'PAMPJPE': float(mr_metrics_dict['PAMPJPE'].item()),
                'num_samples': int(mr_metrics.count_seq.item())
            }
            if 'ACCEL' in mr_metrics_dict:
                results['reconstruction']['ACCEL'] = float(mr_metrics_dict['ACCEL'].item())
            elif 'ACCL' in mr_metrics_dict:
                results['reconstruction']['ACCL'] = float(mr_metrics_dict['ACCL'].item())

        if fid_metrics_dict is not None:
            results['generation'] = {
                'FID': float(fid_metrics_dict['FID'].item()),
                'Diversity': float(fid_metrics_dict['Diversity'].item()),
                'GT_Diversity': float(fid_metrics_dict['gt_Diversity'].item()),
                'num_samples': int(vae_fid_metrics.count_seq.item())
            }

        import json
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        log.info(f"Results saved to: {results_path}")

    return {
        'reconstruction': mr_metrics_dict,
        'generation': fid_metrics_dict
    }


if __name__ == '__main__':
    compute_vae_fid()
