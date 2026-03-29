#!/bin/bash

python scripts/batch_compute_metrics.py \
    --exp_folders experiments/vae_motionfix/diffusion/5/3way_steps_300_motionfix_noise_799

#            experiments/.../noise_799/all_configs_summary.json
#            experiments/.../noise_799/all_configs_comparison.csv

python scripts/batch_compute_metrics.py \
    --exp_folders experiments/vae_motionfix/diffusion/5/3way_steps_300_motionfix_noise_799 \
                  experiments/vae_motionfix/diffusion/5/3way_steps_300_motionfix_noise_1199 \
                  experiments/vae_motionfix/diffusion/5/3way_steps_300_motionfix_noise_1599

python scripts/batch_compute_metrics.py \
    --exp_folders experiments/vae_motionfix/diffusion/5/3way_steps_300_motionfix_noise_*

python scripts/batch_compute_metrics.py \
    --exp_folders experiments/vae_motionfix/diffusion/5/3way_steps_300_motionfix_noise_799 \
    --pattern "ld_txt-2.0_ld_mot-*"
