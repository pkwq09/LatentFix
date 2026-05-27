# LatentFix：Efficient Text-Driven 3D Human Motion Editing in Latent Space
## Environment Setup
Please follow [MotionFix](https://github.com/atnikos/motionfix) to download the dataset and set up the environment.

Pretraind Model Can be download form [this link](https://pan.baidu.com/s/15ZQNdyiB9Rkche14xeNUMw?pwd=7b89).

## Evaluation

### Step 1: Extract Samples

Before running inference, update the following fields in `configs/motionfix_eval.yaml`:

- `folder`: path to the experiment directory that contains the target checkpoint.
- `ckpt_name`: training step of the checkpoint to evaluate.
- `guidance_scale_text_n_motion`: guidance scale for the text condition.
- `guidance_scale_motion`: guidance scale for the motion condition.

After setting the configuration, run:

```bash
python motionfix_evaluate.py
```

You can also override these fields directly from the command line:

```bash
python motionfix_evaluate.py folder=/path/to/exp/ ckpt_name=2000 guidance_scale_text_n_motion=2.5 guidance_scale_motion=4.5
```
### Step 2: Compute Metrics

Update the `folder` field in `configs/compute_metrics.yaml` to the sample directory generated in Step 1:

Then run:

```bash
python compute_metrics.py
```
## Training
### Stage 1: Train the VAE

Set `stage` to `vae` in `configs/train.yaml`. The compression scale can be adjusted through `latent_dim` in `configs/model/vae/vae.yaml`.

Then run：
```bash
python train.py
```

### Stage 2: Train the Diffusion Model

Set `stage` to `diffusion` in `configs/train.yaml`, and set `pretrained_ckpt` in `configs/model/vae/vae.yaml` to the pretrained VAE checkpoint path.

Then run：
```bash
python train.py
```

For both stages, the number of training epochs can be adjusted through `max_epochs` in `configs/trainer/base.yaml`.

## Visualization

To visualize a generated motion sample, run:

```bash
python visualize_sample.py --path /path/to/motion.npy
```

## Acknowledgements

Our code is based on [MotionFix](https://github.com/atnikos/motionfix).
