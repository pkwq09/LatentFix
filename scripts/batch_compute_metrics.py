

import argparse
import sys
from pathlib import Path
import subprocess
import json
from datetime import datetime
import re


project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def find_sample_folders(exp_folder, pattern="ld_txt-*_ld_mot-*"):

    exp_path = Path(exp_folder)

    if not exp_path.exists():
        print(f"⚠️  Warning: Folder {exp_folder} does not exist, skipping...")
        return []


    sample_folders = sorted(exp_path.glob(pattern))


    valid_folders = []
    for folder in sample_folders:
        if folder.is_dir():
            npy_files = list(folder.glob("*.npy"))
            if len(npy_files) > 0:
                valid_folders.append(folder)
            else:
                print(f"⚠️  Skipping {folder.name}: No .npy files found")

    return valid_folders


def run_compute_metrics(sample_folder):

    sample_path = Path(sample_folder)

    print(f"\n{'='*70}")
    print(f"📊 Computing metrics for: {sample_path.name}")
    print(f"{'='*70}")


    cmd = [
        "python", "compute_metrics.py",
        f"folder={sample_path}"
    ]

    try:

        result = subprocess.run(
            cmd,
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=600
        )

        if result.returncode == 0:
            print("✅ Metrics computation successful")


            metrics_dict = extract_metrics_from_output(result.stdout, sample_path)

            return True, result.stdout, metrics_dict
        else:
            print(f"❌ Error computing metrics:")
            print(result.stderr)


            error_file = sample_path / "evaluation_error.txt"
            with open(error_file, 'w', encoding='utf-8') as f:
                f.write(f"Error occurred at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*70 + "\n\n")
                f.write("STDOUT:\n")
                f.write(result.stdout)
                f.write("\n\nSTDERR:\n")
                f.write(result.stderr)
            print(f"📁 Error log saved to: {error_file}")

            return False, result.stderr, None

    except subprocess.TimeoutExpired:
        print(f"❌ Timeout: Metrics computation took longer than 10 minutes")
        return False, "Timeout", None
    except Exception as e:
        print(f"❌ Exception occurred: {e}")
        return False, str(e), None


def extract_metrics_from_output(stdout_text, sample_folder):

    try:
        metrics = {}
        lines = stdout_text.split('\n')


        metrics['config'] = Path(sample_folder).name

        retrieval_section = None  # 'batch' or 'full'
        retrieval_direction = None  # 's2t' or 't2g'


        for i, line in enumerate(lines):

            if 'FID:' in line and 'Distribution Quality Metrics' in '\n'.join(lines[max(0,i-5):i]):
                try:
                    fid_val = float(line.split('FID:')[1].strip().split()[0])
                    metrics['FID'] = fid_val
                except:
                    pass


            if 'Diversity:' in line and 'Distribution Quality Metrics' in '\n'.join(lines[max(0,i-5):i]):
                try:
                    div_val = float(line.split('Diversity:')[1].strip().split()[0])
                    metrics['Diversity'] = div_val
                except:
                    pass


            if 'L2 Distance (gen vs target):' in line:
                try:
                    l2_val = float(line.split(':')[1].strip().split()[0])
                    metrics['L2_Distance'] = l2_val
                except:
                    pass

            stripped = line.strip()

            if stripped.startswith('📊 Retrieval Metrics (Batches'):
                retrieval_section = 'batch'
                retrieval_direction = None
                continue
            if stripped.startswith('📊 Retrieval Metrics (Full'):
                retrieval_section = 'full'
                retrieval_direction = None
                continue
            if stripped.startswith('📐') or stripped.startswith('📈') or stripped.startswith('====='):
                retrieval_section = None
                retrieval_direction = None
            if 'Source → Target' in stripped:
                retrieval_direction = 's2t'
                continue
            if 'Target → Generated' in stripped:
                retrieval_direction = 't2g'
                continue

            if retrieval_section and retrieval_direction:
                match = re.match(r'^([A-Za-z@0-9_]+)\s+([-+]?\d*\.?\d+)', stripped)
                if match:
                    metric_key = match.group(1)
                    try:
                        value = float(match.group(2))
                        key = f"{retrieval_section}_{retrieval_direction}_{metric_key}"
                        metrics[key] = value
                    except ValueError:
                        pass

            if 'Number of samples:' in line:
                try:
                    metrics['num_samples'] = float(line.split('Number of samples:')[1].strip().split()[0])
                except:
                    pass


        speed_stats_file = Path(sample_folder) / 'speed_statistics.json'
        if speed_stats_file.exists():
            with open(speed_stats_file, 'r') as f:
                speed_stats = json.load(f)

                if 'timing_stats' in speed_stats:
                    timing = speed_stats['timing_stats']
                    if timing.get('total_samples', 0) > 0:
                        metrics['avg_time_per_sample'] = timing['total_time'] / timing['total_samples']

                if 'model_complexity' in speed_stats:
                    comp = speed_stats['model_complexity']
                    if comp.get('flops_per_sample'):
                        metrics['flops'] = comp['flops_per_sample']

        return metrics

    except Exception as e:
        print(f"⚠️  Warning: Could not extract metrics: {e}")
        return {'config': Path(sample_folder).name}


def save_aggregated_results(exp_folder, all_results, all_metrics):

    exp_path = Path(exp_folder)


    output_file = exp_path / "all_configs_evaluation.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Aggregated Evaluation Results for {exp_path.name}\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*70 + "\n\n")

        for result in all_results:
            if result['success']:
                config_name = Path(result['folder']).name
                f.write("\n" + "="*70 + "\n")
                f.write(f"Configuration: {config_name}\n")
                f.write("="*70 + "\n\n")
                f.write(result['output'])
                f.write("\n\n")
            else:
                config_name = Path(result['folder']).name
                f.write(f"\n❌ {config_name}: FAILED\n")

    print(f"\n📁 Detailed results saved to: {output_file}")


    json_file = exp_path / "all_configs_summary.json"
    summary = {
        'experiment_folder': str(exp_path),
        'timestamp': datetime.now().isoformat(),
        'total_configs': len(all_results),
        'successful_configs': sum(1 for r in all_results if r['success']),
        'failed_configs': sum(1 for r in all_results if not r['success']),
        'configurations': all_metrics
    }
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"📁 JSON summary saved to: {json_file}")


    if all_metrics:
        csv_file = exp_path / "all_configs_comparison.csv"
        import csv


        all_keys = set()
        for m in all_metrics:
            all_keys.update(m.keys())
        all_keys = sorted(all_keys)

        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(all_metrics)

        print(f"📁 CSV comparison saved to: {csv_file}")


        print("\n" + "="*70)
        print("📊 Metrics Comparison Table")
        print("="*70)


        key_metrics = ['config', 'FID', 'R@1', 'L2_Distance', 'avg_time_per_sample']
        available_metrics = [k for k in key_metrics if any(k in m for m in all_metrics)]

        if available_metrics:

            header = " | ".join([f"{k:>15}" for k in available_metrics])
            print(header)
            print("-" * len(header))


            for m in all_metrics:
                row = " | ".join([f"{str(m.get(k, 'N/A')):>15}" for k in available_metrics])
                print(row)


def batch_compute_metrics(exp_folders, pattern="ld_txt-*_ld_mot-*"):

    print("\n" + "="*70)
    print("🚀 Batch Metrics Computation Started")
    print("="*70)

    all_exp_results = []

    for exp_folder in exp_folders:
        print(f"\n📂 Processing experiment folder: {exp_folder}")
        exp_path = Path(exp_folder)


        sample_folders = find_sample_folders(exp_folder, pattern)

        if not sample_folders:
            print(f"⚠️  No sample folders found in {exp_folder}")
            continue

        print(f"✅ Found {len(sample_folders)} configuration(s)")


        config_results = []
        config_metrics = []

        for sample_folder in sample_folders:
            success, stdout, metrics = run_compute_metrics(sample_folder)

            config_results.append({
                'folder': str(sample_folder),
                'success': success,
                'output': stdout
            })

            if success and metrics:
                config_metrics.append(metrics)


        if config_results:
            save_aggregated_results(exp_path, config_results, config_metrics)

        all_exp_results.append({
            'exp_folder': str(exp_path),
            'total': len(config_results),
            'success': sum(1 for r in config_results if r['success']),
            'failed': sum(1 for r in config_results if not r['success'])
        })


    print("\n" + "="*70)
    print("📊 Overall Summary")
    print("="*70)

    for exp_result in all_exp_results:
        print(f"\n📂 {Path(exp_result['exp_folder']).name}:")
        print(f"  ✅ Successful: {exp_result['success']}/{exp_result['total']}")
        if exp_result['failed'] > 0:
            print(f"  ❌ Failed: {exp_result['failed']}/{exp_result['total']}")

    print("\n" + "="*70)
    print("✅ Batch computation completed!")
    print("📁 Results saved in experiment folders as:")
    print("  • all_configs_evaluation.txt - Full detailed output")
    print("  • all_configs_summary.json - JSON summary")
    print("  • all_configs_comparison.csv - CSV comparison table")
    print("="*70)

    return all_exp_results


def main():
    parser = argparse.ArgumentParser(
        description="Batch compute evaluation metrics for multiple sample folders",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compute metrics for specific experiment folders
  python scripts/batch_compute_metrics.py \\
      --exp_folders experiments/vae_motionfix/diffusion/5/3way_steps_300_motionfix_noise_799 \\
                    experiments/vae_motionfix/diffusion/5/3way_steps_300_motionfix_noise_1199

  # Use custom pattern
  python scripts/batch_compute_metrics.py \\
      --exp_folders experiments/vae_motionfix/diffusion/5/* \\
      --pattern "ld_txt-2.0_ld_mot-*"
        """
    )

    parser.add_argument(
        '--exp_folders',
        nargs='+',
        required=True,
        help='Experiment folder paths (supports wildcards)'
    )

    parser.add_argument(
        '--pattern',
        type=str,
        default='ld_txt-*_ld_mot-*',
        help='Pattern to match sample folders (default: ld_txt-*_ld_mot-*)'
    )

    args = parser.parse_args()


    from glob import glob
    expanded_folders = []
    for folder_pattern in args.exp_folders:
        matches = glob(folder_pattern)
        if matches:
            expanded_folders.extend(matches)
        else:
            expanded_folders.append(folder_pattern)


    batch_compute_metrics(
        expanded_folders,
        pattern=args.pattern
    )


if __name__ == '__main__':
    main()
