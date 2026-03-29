"""Compute MotionFix evaluation metrics via TMR (text-motion retrieval).

Main metrics include motion-to-motion retrieval accuracy.

Usage:
    python compute_metrics.py folder=<path_to_generated_samples_folder>
"""

from omegaconf import DictConfig
import logging
import hydra
import torch
from tqdm import tqdm
from pathlib import Path
import numpy as np
import json


def collect_gen_samples(motion_gen_path, normalizer, device):
    """Collect generated motion samples and preprocess them for evaluation."""
    cur_samples = {}
    cur_samples_raw = {}
    print("Collecting Generated Samples")
    from src.data.features import _get_body_transl_delta_pelv_infer
    import glob

    sample_files = glob.glob(f'{motion_gen_path}/*.npy')
    sample_files = [f for f in sample_files if not Path(f).name.endswith('_source.npy') and not Path(f).name.endswith('_target.npy')]
    for fname in tqdm(sample_files):
        keyid = str(Path(fname).name).replace('.npy', '')
        gen_motion_b = np.load(fname,
                               allow_pickle=True).item()['pose']
        gen_motion_b = torch.from_numpy(gen_motion_b)

        trans = gen_motion_b[..., :3]
        global_orient_6d = gen_motion_b[..., 3:9]
        body_pose_6d = gen_motion_b[..., 9:]

        trans_delta = _get_body_transl_delta_pelv_infer(global_orient_6d, trans)

        gen_motion_b_fixed = torch.cat([trans_delta, body_pose_6d,
                                        global_orient_6d], dim=-1)
        gen_motion_b_fixed = normalizer(gen_motion_b_fixed)
        cur_samples[keyid] = gen_motion_b_fixed.to(device)
        cur_samples_raw[keyid] = torch.cat([trans, global_orient_6d,
                                            body_pose_6d], dim=-1).to(device)
    return cur_samples, cur_samples_raw


@hydra.main(config_path="configs", version_base="1.2", config_name="compute_metrics")
def _compute_metrics(cfg: DictConfig):
    """Hydra entry point for metric computation."""
    return compute_metrics(cfg)

def compute_metrics(newcfg: DictConfig) -> None:
    """Run TMR-based motion-to-motion retrieval evaluation."""
    from tmr_evaluator.motion2motion_retr import retrieval
    from pathlib import Path

    samples_folder = newcfg.folder
    evaluate_gt = newcfg.get('evaluate_gt', False)
    compute_distance = newcfg.get('compute_l2_distance', True)

    results = retrieval(samples_folder, evaluate_gt=evaluate_gt, compute_distance=compute_distance)
    metrs_batches, metrs_full, fid_metrics, distance_metric = results[:4]

    print("\n" + "="*70)
    print("===== EVALUATION RESULTS FOR GENERATED MOTIONS =====")
    print("="*70)

    print("\n📊 Retrieval Metrics (Batches of 32):")
    _print_metrics_dict(metrs_batches)

    print("\n📊 Retrieval Metrics (Full Test Set):")
    _print_metrics_dict(metrs_full)

    if fid_metrics:
        print("\n📈 Distribution Quality Metrics:")
        print(f"  FID:                {fid_metrics.get('FID', 'N/A'):.4f}")
        print(f"  Diversity:          {fid_metrics.get('Diversity', 'N/A'):.4f}")
        print(f"  Number of samples:  {fid_metrics.get('num_samples', 'N/A')}")

    if distance_metric is not None:
        print("\n📐 Geometric Distance Metric:")
        print(f"  L2 Distance (gen vs target): {distance_metric:.4f}")
        print("  (Lower is better - measures avg L2 in SMPL joint space)")

    if evaluate_gt and len(results) > 4:
        gt_metrs_batches, gt_metrs_full, gt_fid_metrics, gt_distance_metric = results[4:]

        print("\n\n" + "="*70)
        print("===== GROUND TRUTH METRICS (Upper Bound Reference) =====")
        print("="*70)
        print("Note: GT metrics compare target motions with themselves (theoretical optimum)\n")

        if gt_fid_metrics:
            print("📈 GT Distribution Quality:")
            print(f"  FID:       {gt_fid_metrics.get('gt_FID', 'N/A'):.6f} (should be ≈0)")
            print(f"  Diversity: {gt_fid_metrics.get('gt_Diversity', 'N/A'):.4f}")

        print("\n\n" + "="*70)
        print("===== COMPREHENSIVE COMPARISON: Generated vs GT =====")
        print("="*70)
        _print_comparison(metrs_batches, gt_metrs_batches, fid_metrics, gt_fid_metrics,
                         distance_metric, folder=newcfg.folder)


def _print_metrics_dict(metrics_dict):
    """Print a metrics dict in a readable layout."""
    if not metrics_dict:
        print("  No metrics available")
        return

    s2t_keys = [k for k in metrics_dict.keys() if 's2t' in k]
    t2g_keys = [k for k in metrics_dict.keys() if 's2t' not in k]

    if s2t_keys:
        print("  Source → Target:")
        for key in sorted(s2t_keys):
            print(f"    {key:<12} {metrics_dict[key]}")

    if t2g_keys:
        print("  Target → Generated:")
        for key in sorted(t2g_keys):
            print(f"    {key:<12} {metrics_dict[key]}")


def _print_comparison(metrs_batches, gt_metrs_batches, fid_metrics, gt_fid_metrics,
                     distance_metric=None, folder=None):
    """Print a side-by-side comparison of generated vs GT metrics."""

    print("\n1️⃣  Retrieval Metrics (Batches of 32):")
    print(f"{'Metric':<15} {'Generated':<15} {'GT Bound':<15} {'Gap':<10}")
    print("-" * 60)

    for key in sorted(metrs_batches.keys()):
        gt_key = f"gt_{key}"
        gen_val = metrs_batches.get(key, 'N/A')
        gt_val = gt_metrs_batches.get(gt_key, 'N/A')

        try:
            gen_float = float(gen_val)
            gt_float = float(gt_val)
            if 'AvgR' in key:
                gap = f"{gen_float - gt_float:+.2f}"
            else:
                gap = f"{gen_float - gt_float:+.2f}"
        except:
            gap = 'N/A'

        print(f"{key:<15} {str(gen_val):<15} {str(gt_val):<15} {gap:<10}")

    if fid_metrics and gt_fid_metrics:
        print("\n2️⃣  Distribution Quality Metrics:")
        print(f"{'Metric':<15} {'Generated':<15} {'GT Bound':<15} {'Gap':<10}")
        print("-" * 60)

        for key in ['FID', 'Diversity']:
            if key in fid_metrics:
                gt_key = f"gt_{key}"
                gen_val = fid_metrics[key]
                gt_val = gt_fid_metrics.get(gt_key, None)

                if isinstance(gen_val, (int, float)) and gt_val is not None:
                    gap = f"{gen_val - gt_val:+.4f}"
                    print(f"{key:<15} {gen_val:<15.4f} {gt_val:<15.4f} {gap:<10}")

    if distance_metric is not None:
        print("\n3️⃣  Geometric Distance Metric (L2 in SMPL joint space):")
        print(f"{'Type':<20} {'Distance':<15} {'Note':<30}")
        print("-" * 65)

        if distance_metric is not None:
            print(f"{'Generated vs Target':<20} {distance_metric:<15.4f} {'(lower is better)'}")

    print("\n" + "="*70)
    print("💡 Interpretation Guide:")
    print("  • Retrieval R@k: Higher is better (closer to GT = better editing quality)")
    print("  • AvgR: Lower is better (closer to GT = better ranking)")
    print("  • FID: Lower is better (closer to 0 = more realistic distribution)")
    print("  • L2 Distance: Lower is better (closer to target motion)")
    print("="*70)

    speed_stats_file = Path(folder) / 'speed_statistics.json' if folder else None
    if speed_stats_file and speed_stats_file.exists():
        print("\n" + "="*70)
        print("===== GENERATION SPEED STATISTICS =====")
        print("="*70)

        with open(speed_stats_file, 'r') as f:
            speed_stats = json.load(f)

        inf_cfg = speed_stats.get('inference_config', {})
        print("\n⚙️  Inference Configuration:")
        print(f"  Inference steps:    {inf_cfg.get('num_inference_steps', 'N/A')}")
        print(f"  Use VAE:            {inf_cfg.get('use_vae', 'N/A')}")
        if inf_cfg.get('latent_shape'):
            print(f"  Latent shape:       {inf_cfg.get('latent_shape')}")
        print(f"  Diffusion input:    {inf_cfg.get('diffusion_input_size', 'N/A')} dims" if isinstance(inf_cfg.get('diffusion_input_size'), int) else f"  Diffusion input:    {inf_cfg.get('diffusion_input_size', 'N/A')}")

        model_comp = speed_stats.get('model_complexity', {})
        print("\n📊 Model Complexity:")
        total_params = model_comp.get('total_params')
        if total_params:
            print(f"  Total parameters:   {total_params/1e6:.2f}M")

        denoiser_params = model_comp.get('denoiser_params')
        if denoiser_params:
            print(f"  Denoiser params:    {denoiser_params/1e6:.2f}M")

        vae_params = model_comp.get('vae_params')
        if vae_params:
            print(f"  VAE params:         {vae_params/1e6:.2f}M")

        flops = model_comp.get('flops_per_sample')
        if flops:
            print(f"  FLOPs per sample:   {flops:.2f}G")

        timing = speed_stats.get('timing_stats', {})
        if timing.get('total_samples', 0) > 0:
            avg_time = timing['total_time'] / timing['total_samples']
            print("\n⏱️  Timing Statistics:")
            print(f"  Total samples:      {timing['total_samples']}")
            print(f"  Total time:         {timing['total_time']:.2f} seconds")
            print(f"  Avg time/sample:    {avg_time:.4f} seconds")
            print(f"  Throughput:         {1.0/avg_time:.2f} samples/sec")

        print("\n" + "="*70)
    else:
        print("\n⚠️  Speed statistics file not found. Run generation first to collect speed metrics.")
        print(f"   Expected file: {speed_stats_file}")

    print("")

if __name__ == '__main__':
    _compute_metrics()
