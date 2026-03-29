

import os
import sys
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import numpy as np
from thop import profile
from fvcore.nn import FlopCountAnalysis
import logging


project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import src.launch.prepare  # noqa
from src.diffusion import create_diffusion

logger = logging.getLogger(__name__)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def format_number(num):
    if num >= 1e9:
        return f"{num/1e9:.2f}G"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return f"{num:.2f}"


def _extract_tensor(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output, dict):
        preferred_keys = [
            'pred_motion', 'pred', 'motion', 'recons_motion', 'denoised_motion'
        ]
        for key in preferred_keys:
            val = output.get(key)
            if torch.is_tensor(val):
                return val
        for val in output.values():
            if torch.is_tensor(val):
                return val
    if isinstance(output, (list, tuple)):
        for val in output:
            if torch.is_tensor(val):
                return val
    return None


class FullPipelineWrapper(nn.Module):

    def __init__(self, model, batch, cfg, runner):
        super().__init__()
        self.model = model
        self.batch = batch
        self.cfg = cfg
        self.runner = runner

    def forward(self, dummy_input=None):
        output = self.runner(self.model, self.batch, self.cfg)
        tensor = _extract_tensor(output)
        if tensor is None:
            raise ValueError("Full pipeline output does not contain a Tensor for FLOPs analysis")
        return tensor


@hydra.main(config_path="../configs", config_name="motionfix_eval", version_base=None)
def compute_flops(cfg: DictConfig):

    print("\n" + "="*70)
    print("===== MotionFix FLOPs & Parameters Analysis =====")
    print("="*70)


    exp_folder = Path(hydra.utils.to_absolute_path(cfg.folder))
    last_ckpt_path = cfg.last_ckpt_path
    prevcfg = OmegaConf.load(exp_folder / ".hydra/config.yaml")
    cfg = OmegaConf.merge(prevcfg, cfg)

    print(f"\n📁 Experiment: {cfg.folder}")
    print(f"📦 Checkpoint: {cfg.ckpt_name}")


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")


    print("\n⏳ Loading model...")
    from src.model.base_diffusion import MD
    model = MD.load_from_checkpoint(last_ckpt_path, renderer=None, strict=False)
    model = model.to(device)
    model.eval()
    model.freeze()

    use_vae = hasattr(model, 'vae') and model.vae is not None
    print(f"✅ Model loaded (VAE: {use_vae})")


    print("\n⏳ Loading dataset...")
    from hydra.utils import instantiate
    data_module = instantiate(cfg.data, amt_only=True, load_splits=['test'])
    test_dataset = data_module.dataset['test']
    print(f"✅ Dataset loaded ({len(test_dataset)} samples)")


    from src.data.tools.collate import collate_batch_last_padding
    features_to_load = test_dataset.load_feats
    collate_fn = lambda b: collate_batch_last_padding(b, features_to_load)

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=1,
        num_workers=4,
        collate_fn=collate_fn
    )


    print("\n" + "="*70)
    print("📊 Model Parameters")
    print("="*70)

    total_params, trainable_params = count_parameters(model)
    print(f"  Total parameters:      {format_number(total_params)} ({total_params:,})")
    print(f"  Trainable parameters:  {format_number(trainable_params)} ({trainable_params:,})")

    if hasattr(model, 'denoiser'):
        denoiser_params, _ = count_parameters(model.denoiser)
        print(f"  Denoiser parameters:   {format_number(denoiser_params)} ({denoiser_params:,})")

    if use_vae and hasattr(model, 'vae'):
        vae_params, _ = count_parameters(model.vae)
        print(f"  VAE parameters:        {format_number(vae_params)} ({vae_params:,})")

    if hasattr(model, 'text_encoder'):
        text_params, _ = count_parameters(model.text_encoder)
        print(f"  Text encoder params:   {format_number(text_params)} ({text_params:,})")


    print("\n" + "="*70)
    print("🔢 FLOPs Analysis (Full Pipeline)")
    print("="*70)

    print("\n⏳ Profiling model with real batch...")


    batch = next(iter(testloader))


    def prepare_batch(model, batch):
        batch = {k: v.to(model.device) if torch.is_tensor(v) else v
                 for k, v in batch.items()}
        input_batch = model.norm_and_cat(batch, model.input_feats)
        for k, v in input_batch.items():
            batch[f'{k}_motion'] = v
        return batch

    input_batch = prepare_batch(model, batch)


    def generate_motion_wrapper(model, batch, cfg):
        with torch.no_grad():
            text = [t.lower() for t in batch['text']]
            source_motion = batch['source_motion'] if model.motion_condition == 'source' else None
            target_lens = batch['length_target']
            if model.motion_condition == 'source':
                source_lens = batch['length_source']
                if model.pad_inputs:
                    mask_source, mask_target = model.prepare_mot_masks(source_lens, target_lens, max_len=300)
                else:
                    mask_source, mask_target = model.prepare_mot_masks(source_lens, target_lens, max_len=None)
            else:
                from src.data.tools.tensors import lengths_to_mask
                mask_target = lengths_to_mask(target_lens, model.device)
                mask_source = None
            num_infer_steps = cfg.model.diff_params.num_inference_timesteps
            diffusion_process = create_diffusion(
                timestep_respacing=None,
                learn_sigma=False,
                sigma_small=True,
                diffusion_steps=num_infer_steps,
                noise_schedule=cfg.model.diff_params.noise_schedule,
                predict_xstart=False if cfg.model.diff_params.predict_type == 'noise' else True
            )
            diffout = model.generate_motion(
                text,
                source_motion,
                mask_source,
                mask_target,
                diffusion_process,
                init_vec=None,
                init_vec_method='noise',
                condition_mode='full_cond' if model.motion_condition else 'text_cond',
                gd_motion=1.0,
                gd_text=1.0,
                num_diff_steps=num_infer_steps,
                inpaint_dict=None,
                use_linear=False,
                prob_way='3way'
            )
            return diffout

    dummy_input = torch.zeros(1, device=model.device)
    try:
        print("  Profiling with thop (full pipeline wrapper)...")
        full_pipeline = FullPipelineWrapper(model, input_batch, cfg, generate_motion_wrapper).to(model.device)
        _ = full_pipeline(dummy_input)
        macs, params = profile(full_pipeline, inputs=(dummy_input,), verbose=False)
        flops = macs * 2
        profiling_backend = "thop"
    except Exception as thop_err:
        print(f"  ⚠️ thop profiling failed: {thop_err}")
        print("  Trying fvcore.nn.FlopCountAnalysis instead...")
        try:
            full_pipeline = FullPipelineWrapper(model, input_batch, cfg, generate_motion_wrapper).to(model.device)
            _ = full_pipeline(dummy_input)
            flops = FlopCountAnalysis(full_pipeline, (dummy_input,)).total()
            params = count_parameters(full_pipeline)[0]
            macs = flops / 2
            profiling_backend = "fvcore"
        except Exception as fvcore_err:
            print(f"\n❌ FLOPs calculation failed for both backends.")
            print(f"  thop error:   {thop_err}")
            print(f"  fvcore error: {fvcore_err}")
            print("  Please check custom modules or consider simplifying the pipeline.")
            profiling_backend = None
            macs = flops = params = None

    if profiling_backend is not None:
        print(f"\n✅ Profiling completed using {profiling_backend}!")
        print(f"\n  MACs (Multiply-Accumulate):  {format_number(macs)} ({macs:,})")
        print(f"  FLOPs (approximate):         {format_number(flops)} ({flops:,})")
        print(f"  Parameters detected:         {format_number(params)} ({params:,})")

        num_steps = cfg.model.diff_params.num_inference_timesteps
        flops_per_step = flops / num_steps if num_steps > 0 else 0
        print(f"\n  Inference steps:             {num_steps}")
        print(f"  FLOPs per step:              {format_number(flops_per_step)}")


    print("\n" + "="*70)
    print("ℹ️  Model Configuration")
    print("="*70)
    print(f"  Model type:           {cfg.model.modelname}")
    print(f"  Use VAE:              {use_vae}")

    if use_vae:
        print(f"  VAE latent size:      {model.vae.latent_size}")
        print(f"  VAE latent dim:       {model.vae.latent_dim}")
        print(f"  Diffusion input:      ({model.vae.latent_size}, {model.vae.latent_dim})")
    else:
        feat_dim = sum(model.input_feats_dims) if hasattr(model, 'input_feats_dims') else 263
        print(f"  Feature dimension:    {feat_dim}")
        print(f"  Diffusion input:      (seq_len, {feat_dim})")

    print(f"  Text encoder:         {cfg.model.text_encoder._target_.split('.')[-1]}")
    print(f"  Denoiser type:        {cfg.model.denoiser._target_.split('.')[-1]}")

    print("\n" + "="*70)
    print("\n✅ Analysis completed!")
    print("\n💡 Note: This script profiles the full generation pipeline,")
    print("   including text encoding, VAE encoding/decoding (if applicable),")
    print("   and the denoiser with all diffusion steps.")
    print("="*70 + "\n")


if __name__ == '__main__':
    compute_flops()
