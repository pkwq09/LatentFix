

import os
import sys
from pathlib import Path

import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf


project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import src.launch.prepare  # noqa: F401


@hydra.main(config_path="../configs", config_name="motionfix_eval", version_base=None)
def compute_avg_seq_len(cfg: DictConfig) -> None:

    from hydra.utils import instantiate
    from src.data.tools.collate import collate_batch_last_padding
    import torch


    exp_folder = Path(hydra.utils.to_absolute_path(cfg.folder))
    prevcfg = OmegaConf.load(exp_folder / ".hydra/config.yaml")
    cfg = OmegaConf.merge(prevcfg, cfg)

    print("\n" + "=" * 70)
    print("===== MotionFix Test Length Statistics =====")
    print("=" * 70)
    print(f"📁 Experiment folder: {exp_folder}")


    data_module = instantiate(cfg.data, amt_only=True, load_splits=["test"])
    test_dataset = data_module.dataset["test"]

    print(f"✅ Test dataset loaded, num_samples = {len(test_dataset)}")

    features_to_load = test_dataset.load_feats
    collate_fn = lambda b: collate_batch_last_padding(b, features_to_load)

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        num_workers=8,
        batch_size=64,
        collate_fn=collate_fn,
    )

    all_lengths = []
    for batch in testloader:

        lengths = batch.get("length_target", None)
        if lengths is None:
            continue

        all_lengths.extend(list(lengths))

    if not all_lengths:
        print("⚠️ No length_target collected; check dataset and collate.")
        return

    lengths_arr = np.array(all_lengths, dtype=np.float32)

    mean_len = float(lengths_arr.mean())
    median_len = float(np.median(lengths_arr))
    max_len = int(lengths_arr.max())
    p95_len = float(np.percentile(lengths_arr, 95))

    print("\n📊 Target Length Statistics (length_target on test split)")
    print(f"  Mean length:     {mean_len:.2f}")
    print(f"  Median length:   {median_len:.2f}")
    print(f"  95th percentile: {p95_len:.2f}")
    print(f"  Max length:      {max_len:d}")

    print("\nSuggestions:")
    print("  - Use mean or median as representative length T for FLOPs;")
    print("  - Write this value to configs/motionfix_eval.yaml as avg_target_len_for_flops.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    compute_avg_seq_len()


