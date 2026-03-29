"""MotionFix evaluation and sample generation.

Loads a trained checkpoint, runs motion editing / generation on the test set
(with optional rendering). Paper: use this script to generate samples, then
compute_metrics.py for metrics.

Usage:
    python motionfix_evaluate.py folder=<experiment_folder>
"""

import os
import logging
import hydra
import joblib
from omegaconf import DictConfig
from omegaconf import OmegaConf
from src import data
from src.render.mesh_viz import render_motion
from torch import Tensor
from src.render.video import save_video_samples
import src.launch.prepare  # noqa
from tqdm import tqdm
import torch
import itertools
from src.model.utils.tools import pack_to_render
import time

logger = logging.getLogger(__name__)
import numpy as np


def count_parameters(model):
    """Count model parameters (including frozen)."""
    return sum(p.numel() for p in model.parameters())


def calculate_flops_estimate(model, use_vae, latent_shape_tuple, feature_dim, seq_len):
    """Rough FLOPs complexity estimate (attention ~ O(L^2) style)."""
    if use_vae:
        L, d = latent_shape_tuple
        return L, d, L * L
    else:
        return seq_len, feature_dim, seq_len * seq_len


@hydra.main(config_path="configs", config_name="motionfix_eval")
def _render_vids(cfg: DictConfig) -> None:
    """Hydra entry: calls render_vids."""
    return render_vids(cfg)

def chunker(seq, size):
    """Yield fixed-size chunks from seq."""
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def prepare_test_batch(model, batch):
    """Move batch tensors to model device and build norm_and_cat motion fields."""
    batch = { k: v.to(model.device) if torch.is_tensor(v) else v
                for k, v in batch.items() }

    input_batch = model.norm_and_cat(batch, model.input_feats)
    for k, v in input_batch.items():
        batch[f'{k}_motion'] = v

    return batch

def cleanup_files(lo_fls):
    """Remove temporary files."""
    for fl in lo_fls:
        os.remove(fl)

def get_folder_name(config):
    """Build output folder name string from config (legacy helper)."""
    sched_name = config.model.infer_scheduler._target_.split('.')[-1]
    sched_name = sched_name.replace('Scheduler', '').lower()
    mot_guid = config.model.diff_params.guidance_scale_motion
    text_guid = config.model.diff_params.guidance_scale_text
    infer_steps = config.model.diff_params.num_inference_timesteps
    
    if config.init_from == 'source':
        init_from = '_src_init_'
    else:
        init_from = ''
    
    if config.ckpt_name == 'last':
        ckpt_n = ''
    else:
        ckpt_n = f'ckpt-{config.ckpt_name}_'

    return f'{ckpt_n}{init_from}{sched_name}_steps{infer_steps}'


def render_vids(newcfg: DictConfig) -> None:
    """Load checkpoint, run generation over the test loader, save npy samples."""
    from pathlib import Path
    exp_folder = Path(hydra.utils.to_absolute_path(newcfg.folder))
    last_ckpt_path = newcfg.last_ckpt_path
    prevcfg = OmegaConf.load(exp_folder / ".hydra/config.yaml")

    cfg = OmegaConf.merge(prevcfg, newcfg)
    from src.diffusion import create_diffusion

    from src.diffusion.gaussian_diffusion import ModelMeanType, ModelVarType
    from src.diffusion.gaussian_diffusion import LossType
    if cfg.num_sampling_steps is not None:
        if cfg.num_sampling_steps <= cfg.model.diff_params.num_train_timesteps:
            num_infer_steps = cfg.num_sampling_steps
        else:
            num_infer_steps = cfg.model.diff_params.num_train_timesteps
            logger.info('More sampling steps than the training ones! Sampling with maximum')
            logger.info(f'Number of steps: {num_infer_steps}')
    else:
        num_infer_steps = cfg.model.diff_params.num_train_timesteps
    init_diff_from = cfg.init_from
    if init_diff_from == 'source':
        num_infer_steps //= 1
    if cfg.linear_gd:
        use_linear_guid = True
        gd_str = 'lingd_'

    else:
        use_linear_guid = False
        gd_str = ''

    if hasattr(cfg, 'test_infer_scheduler') and cfg.test_infer_scheduler is not None:
        if isinstance(cfg.test_infer_scheduler, str):
            from pathlib import Path
            try:
                scheduler_config_path = Path("configs") / "model" / "infer_scheduler" / f"{cfg.test_infer_scheduler}.yaml"
                if scheduler_config_path.exists():
                    scheduler_cfg = OmegaConf.load(scheduler_config_path)
                    infer_scheduler_to_use = scheduler_cfg
                    infer_sched_target = scheduler_cfg._target_ if hasattr(scheduler_cfg, '_target_') else str(scheduler_cfg)
                else:
                    raise FileNotFoundError(f"Scheduler config not found: {scheduler_config_path}")
            except Exception as e:
                logger.warning(f"Could not load test_infer_scheduler '{cfg.test_infer_scheduler}', falling back to model config: {e}")
                infer_scheduler_to_use = cfg.model.infer_scheduler
                infer_sched_target = cfg.model.infer_scheduler._target_ if hasattr(cfg.model.infer_scheduler, '_target_') else str(cfg.model.infer_scheduler)
        else:
            from hydra.utils import instantiate
            infer_scheduler_to_use = instantiate(cfg.test_infer_scheduler) if hasattr(cfg.test_infer_scheduler, '_target_') else cfg.test_infer_scheduler
            infer_sched_target = infer_scheduler_to_use._target_ if hasattr(infer_scheduler_to_use, '_target_') else str(infer_scheduler_to_use)

        sampler_name = 'DDIM' if 'DDIM' in str(infer_sched_target) else 'DDPM'
        logger.info(f"Using test-time infer scheduler: {sampler_name}")
    else:
        infer_scheduler_to_use = cfg.model.infer_scheduler
        infer_sched_target = cfg.model.infer_scheduler._target_ if hasattr(cfg.model.infer_scheduler, '_target_') else str(cfg.model.infer_scheduler)
        sampler_name = 'DDIM' if 'DDIM' in str(infer_sched_target) else 'DDPM'
        logger.info(f"Using model infer_scheduler (same as validation): {sampler_name}")

    ddim_eta = getattr(cfg.model.diff_params, 'ddim_eta', 0.0)
    logger.info(f"[TEST] sampler={sampler_name}, steps={num_infer_steps}, DDIM eta={ddim_eta}")

    diffusion_process = create_diffusion(timestep_respacing=None,
                                    learn_sigma=False,
                                    sigma_small=True,
                                    diffusion_steps=num_infer_steps,
                                    noise_schedule=cfg.model.diff_params.noise_schedule,
                                    predict_xstart=False if cfg.model.diff_params.predict_type == 'noise' else True)
    # cfg.model.infer_scheduler = newcfg.model.infer_scheduler
    # cfg.model.diff_params.num_inference_timesteps = newcfg.steps
    # cfg.model.diff_params.guidance_scale_motion = newcfg.guidance_scale_motion
    # cfg.model.diff_params.guidance_scale_text = newcfg.guidance_scale_text
    if cfg.inpaint:
        assert cfg.data.dataname == 'motionfix'

    # init_diff_from = 'noise'
    # TODO pUT THIS BACK    
    # fd_name = get_folder_name(cfg)
    sampler_name_lower = sampler_name.lower()
    fd_name = f'{sampler_name_lower}_steps_{num_infer_steps}'
    if cfg.inpaint:
        output_path = exp_folder / f'{cfg.prob_way}_{gd_str}{fd_name}_{cfg.data.dataname}_{cfg.init_from}_{cfg.ckpt_name}_inpaint_bsl'
    else:
        output_path = exp_folder / f'{cfg.prob_way}_{gd_str}{fd_name}_{cfg.data.dataname}_{cfg.init_from}_{cfg.ckpt_name}'

    output_path.mkdir(exist_ok=True, parents=True)
    logger.info(f"-------Output path:{output_path}------")
    import pytorch_lightning as pl
    import numpy as np
    from hydra.utils import instantiate
    from src.render.video import put_text
    from src.render.video import stack_vids
    from tqdm import tqdm

    seed_logger = logging.getLogger("pytorch_lightning.utilities.seed")
    seed_logger.setLevel(logging.WARNING)

    pl.seed_everything(cfg.seed)
    # import wandb
    # wandb.init(project="pose-edit-eval", job_type="evaluate",
    #            name=log_name, dir=output_path)
    aitrenderer = None
    logger.info("Loading model")
    from src.model.base_diffusion import MD
    model = MD.load_from_checkpoint(
        last_ckpt_path,
        renderer=aitrenderer,
        infer_scheduler=infer_scheduler_to_use,
        strict=False,
        map_location="cpu",
    )

    actual_sampler_name = 'DDIM' if model.use_ddim else 'DDPM'
    actual_ddim_eta = model.ddim_eta
    if actual_sampler_name != sampler_name:
        logger.warning(f"Config sampler ({sampler_name}) != model sampler ({actual_sampler_name}); using model value")
        sampler_name = actual_sampler_name
        ddim_eta = actual_ddim_eta
    logger.info(f"Model loaded; sampler={sampler_name}, DDIM eta={ddim_eta}")

    if cfg.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif cfg.device == 'cuda':
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            device = torch.device('cpu')
        else:
            gpu_id = getattr(cfg, 'gpu_id', None)
            if gpu_id is not None:
                device = torch.device(f'cuda:{int(gpu_id)}')
            else:
                device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    logger.info(f"Moving model to device: {device}")
    model = model.to(device)
    
    model.eval()
    model.freeze()
    logger.info(f"Model '{cfg.model.modelname}' loaded on {device}")
    # logger.info('------Generating using Scheduler------\n\n'\
    #             f'{model.infer_scheduler}')
    logger.info('------Diffusion Parameters------\n\n'\
                f'{model.diff_params}')
    
    if hasattr(model.denoiser, 'use_skip_transformer'):
        if model.denoiser.use_skip_transformer:
            logger.info(f"✅ Skip Transformer: ENABLED [layers={model.denoiser.encoder.num_layers}]")
        else:
            logger.info(f"⚠️  Skip Transformer: DISABLED (using standard Transformer)")
    
    logger.info(f"Initialization method: '{cfg.init_from}'")
    if cfg.init_from == 'source':
        logger.info(f"   → The diffusion process will be initiated from the source action")
    else:
        logger.info(f"   → The diffusion process will be initialized from noise.")
    
    use_vae = hasattr(model, 'vae') and model.vae is not None
    latent_shape = None
    latent_shape_tuple = None
    diffusion_input_size = None
    
    if use_vae:
        latent_shape_tuple = (model.vae.latent_size, model.vae.latent_dim)
        latent_shape = f"({model.vae.latent_size}, {model.vae.latent_dim})"
        diffusion_input_size = model.vae.latent_size * model.vae.latent_dim
    else:
        feat_dim = sum(model.input_feats_dims) if hasattr(model, 'input_feats_dims') else 263
        diffusion_input_size = f"L × {feat_dim}"
    
    model_stats = {}
    model_stats['total_params'] = count_parameters(model)
    
    if hasattr(model, 'denoiser'):
        model_stats['denoiser_params'] = count_parameters(model.denoiser)
    if use_vae and hasattr(model, 'vae'):
        model_stats['vae_params'] = count_parameters(model.vae)
    
    avg_seq_len = getattr(cfg, 'avg_target_len_for_flops', 196)
    if hasattr(model, 'input_feats_dims'):
        feat_dim = sum(model.input_feats_dims)
    elif hasattr(model, 'input_feats'):
        feat_dim = 263
    else:
        feat_dim = 263
    
    L_diff, d_diff, complexity_coef = calculate_flops_estimate(
        model, use_vae, latent_shape_tuple, feat_dim, avg_seq_len
    )
    
    if use_vae:
        tmed_complexity = avg_seq_len * avg_seq_len
        vae_tmed_complexity = complexity_coef
        complexity_reduction = tmed_complexity / vae_tmed_complexity if vae_tmed_complexity > 0 else 1
    else:
        complexity_reduction = 1
    
    timing_stats = {
        'total_samples': 0,
        'total_time': 0.0,
    }
    
    flops_computed = False
    flops_value = None


    data_module = instantiate(cfg.data, amt_only=True)

    transl_feats = [x for x in model.input_feats if 'transl' in x]
    if set(transl_feats).issubset(["body_transl_delta", "body_transl_delta_pelv",
                                   "body_transl_delta_pelv_xy"]):
        model.using_deltas_transl = True
    
    split_to_load = cfg.get('split_to_load', ['test'])
    if isinstance(split_to_load, str):
        split_to_load = [split_to_load]
    elif not isinstance(split_to_load, (list, tuple)):
        split_to_load = ['test']

    valid_splits = ['train', 'val', 'test']
    split_to_load = [s for s in split_to_load if s in valid_splits]
    if not split_to_load:
        logger.warning("No valid splits in split_to_load; defaulting to 'test'")
        split_to_load = ['test']
    
    logger.info(f'Generating samples for splits: {split_to_load}')

    datasets_to_merge = []
    for split in split_to_load:
        if split in data_module.dataset:
            datasets_to_merge.append(data_module.dataset[split])
            logger.info(f'  - {split}: {len(data_module.dataset[split])} samples')
        else:
            logger.warning(f'  - {split}: split missing, skipped')
    
    if not datasets_to_merge:
        raise ValueError(f"No dataset splits available. Check split_to_load: {split_to_load}")

    if len(datasets_to_merge) == 1:
        test_dataset = datasets_to_merge[0]
    else:
        test_dataset = datasets_to_merge[0]
        for ds in datasets_to_merge[1:]:
            test_dataset = test_dataset + ds
    
    features_to_load = datasets_to_merge[0].load_feats

    from src.data.tools.collate import collate_batch_last_padding
    collate_fn = lambda b: collate_batch_last_padding(b, features_to_load)

    subset = []
    testloader = torch.utils.data.DataLoader(test_dataset,
                                             shuffle=False,
                                             num_workers=32,
                                             batch_size=128,
                                             collate_fn=collate_fn)
    ds_iterator = testloader 

    from src.utils.art_utils import color_map
    
    mode_cond = cfg.condition_mode
    if cfg.model.motion_condition is None:
        mode_cond = 'text_cond'
    else:
        mode_cond = cfg.condition_mode

    tot_pkls = []
    
    if hasattr(cfg, 'guidance_combinations') and cfg.guidance_combinations is not None:
        comb_val = OmegaConf.to_container(cfg.guidance_combinations, resolve=True)
        if isinstance(comb_val, (list, tuple)) and len(comb_val) > 0:
            guidances_mix = []
            for comb in comb_val:
                if isinstance(comb, (list, tuple)) and len(comb) == 2:
                    guidances_mix.append((float(comb[0]), float(comb[1])))
                else:
                    raise ValueError(f"Invalid guidance pair {comb}; expected [text_scale, motion_scale]")
            logger.info(f'Using guidance_combinations: {guidances_mix}')
        else:
            raise ValueError(f"Invalid guidance_combinations: {comb_val}; expected list of pairs")
    else:
        if cfg.guidance_scale_text_n_motion is None:
            gd_text = [2.0,3.0]
        else:
            text_val = OmegaConf.to_container(cfg.guidance_scale_text_n_motion, resolve=True)
            if isinstance(text_val, (list, tuple)):
                gd_text = [float(x) for x in text_val]
            else:
                gd_text = [float(text_val)]
        if cfg.guidance_scale_motion is None:
            gd_motion = [2.0,3.0,4.0,5.0,6.0,7.0]
        else:
            motion_val = OmegaConf.to_container(cfg.guidance_scale_motion, resolve=True)
            if isinstance(motion_val, (list, tuple)):
                gd_motion = [float(x) for x in motion_val]
            else:
                gd_motion = [float(motion_val)]

        guidances_mix = [(x, y) for x in gd_text for y in gd_motion]
        logger.info(f'Cartesian product guidance grid: {guidances_mix}')

    if cfg.model.motion_condition is None:
        mode_cond = 'text_cond'
    else:
        mode_cond = 'full_cond'
    logger.info(f'Evaluation Set length:{len(test_dataset)}')
    if cfg.inpaint:
        model.motion_condition = None
    if cfg.save_gt:
        save_data_sample = True
    else:
        save_data_sample = False
    with torch.no_grad():
        for guid_text, guid_motion in guidances_mix:
            assert not isinstance(guid_text, (list, tuple)), f"guid_text must be scalar, got: {guid_text}"
            assert not isinstance(guid_motion, (list, tuple)), f"guid_motion must be scalar, got: {guid_motion}"
            cur_guid_comb = f'ld_txt-{guid_text}_ld_mot-{guid_motion}'
            cur_outpath = output_path / cur_guid_comb
            cur_outpath.mkdir(exist_ok=True, parents=True)
            logger.info(f"Sample MotionFix test set\n in:{cur_outpath}")

            for batch in tqdm(ds_iterator):
                batch_start_time = time.time()

                text_diff = batch['text']
                target_lens = batch['length_target']
                keyids = batch['id']
                source_lens = batch['length_source']
                no_of_motions = len(keyids)
                if save_data_sample:
                    dataset_motions = prepare_test_batch(model, batch)
                    src_mot_cond, tgt_mot = model.batch2motion(dataset_motions, pack_to_dict=False)
                input_batch = prepare_test_batch(model, batch)
                if cfg.inpaint:
                    from src.model.utils.body_parts import get_mask_from_texts, get_mask_from_bps
                    parts_to_keep = text_diff

                    try:
                        jts_ids = get_mask_from_texts(parts_to_keep)
                    except:
                        import ipdb;ipdb.set_trace()
                    mask_features = get_mask_from_bps(jts_ids, device=model.device,
                                                    feat_dim=sum(model.input_feats_dims))
                    inpaint_dict = {'mask': mask_features,
                                    'start_motion': input_batch['source_motion'].clone() }
                else:
                    inpaint_dict = None
                text_diff = [el.lower() for el in batch['text']]

                if model.motion_condition == 'source' or init_diff_from!='noise':
                    source_mot_pad = input_batch['source_motion'].clone()
                else:
                    source_mot_pad = None

                if model.motion_condition == 'source' or init_diff_from == 'source':
                    source_lens = batch['length_source']
                    if model.pad_inputs:
                        mask_source, mask_target = model.prepare_mot_masks(source_lens,
                                                                        target_lens,
                                                                        max_len=300)
                    else:
                        mask_source, mask_target = model.prepare_mot_masks(source_lens,
                                                                        target_lens,
                                                                        max_len=None)

                else:
                    from src.data.tools.tensors import lengths_to_mask
                    mask_target = lengths_to_mask(target_lens,
                                                model.device)
                    batch['source_motion'] = None
                    mask_source = None


                if init_diff_from == 'source':
                    source_init = source_mot_pad
                else:
                    source_init = None
                diffout = model.generate_motion(text_diff,
                                                source_mot_pad,
                                                mask_source,
                                                mask_target,
                                                diffusion_process,
                                                init_vec=source_init,
                                                init_vec_method=init_diff_from,
                                                condition_mode=mode_cond,
                                                gd_motion=guid_motion,
                                                gd_text=guid_text,
                                                num_diff_steps=num_infer_steps,
                                                inpaint_dict=inpaint_dict,
                                                use_linear=use_linear_guid, 
                                                prob_way=cfg.prob_way 
                                                )
                gen_mo = model.diffout2motion(diffout)
                
                if not flops_computed:
                    try:
                        from thop import profile

                        with torch.no_grad():
                            if use_vae:
                                dummy_latent = torch.randn(1, model.vae.latent_size, model.vae.latent_dim).to(model.device)
                                dummy_mask = torch.ones(1, model.vae.latent_size, dtype=torch.bool).to(model.device)
                            else:
                                dummy_seq_len = int(getattr(cfg, 'avg_target_len_for_flops', 196))
                                dummy_latent = torch.randn(1, dummy_seq_len, feat_dim).to(model.device)
                                dummy_mask = torch.ones(1, dummy_seq_len, dtype=torch.bool).to(model.device)

                            dummy_text_emb, dummy_text_mask = model.text_encoder([text_diff[0]])

                            if model.motion_condition == 'source':
                                if use_vae:
                                    source_mask_len = model.vae.latent_size
                                else:
                                    source_mask_len = dummy_seq_len

                                dummy_cond_mask = torch.cat([
                                    dummy_text_mask,
                                    torch.ones(1, source_mask_len, dtype=torch.bool).to(model.device)
                                ], dim=1)
                            else:
                                dummy_cond_mask = dummy_text_mask

                            dummy_timestep = torch.zeros(1, dtype=torch.long).to(model.device)

                            if model.motion_condition == 'source':
                                if use_vae:
                                    dummy_motion_emb = torch.randn(model.vae.latent_size, 1, model.vae.latent_dim).to(model.device)
                                else:
                                    dummy_motion_emb = torch.randn(dummy_seq_len, 1, feat_dim).to(model.device)
                            else:
                                dummy_motion_emb = None

                            macs, params = profile(
                                model.denoiser,
                                inputs=(
                                    dummy_latent,
                                    dummy_timestep,
                                    dummy_mask,
                                    dummy_text_emb,
                                    dummy_cond_mask,
                                    dummy_motion_emb
                                ),
                                verbose=False
                            )
                            
                            flops_value = macs * num_infer_steps / 1e9
                            flops_computed = True
                            logger.info(f"FLOPs per sample: {flops_value:.2f}G")
                    except Exception as e:
                        logger.warning(f"FLOPs calculation failed: {e}")
                        flops_computed = True

                batch_end_time = time.time()
                batch_time = batch_end_time - batch_start_time
                
                timing_stats['total_samples'] += no_of_motions
                timing_stats['total_time'] += batch_time
                
                from src.tools.transforms3d import transform_body_pose
                for i in range(gen_mo.shape[0]):
                    dict_to_save = {'pose': gen_mo[i, 
                                                   :target_lens[i]].cpu().numpy() 
                                    }
                    np.save(cur_outpath / f"{str(batch['id'][i]).zfill(6)}.npy",
                            dict_to_save)
                    if save_data_sample:
                        dict_to_save = {'pose': src_mot_cond[i,
                                                   :source_lens[i]].cpu().numpy()
                                    }
                        np.save(cur_outpath / f"{str(batch['id'][i]).zfill(6)}_source.npy",
                            dict_to_save)
                        dict_to_save = {'pose': tgt_mot[i,
                                                   :target_lens[i]].cpu().numpy()
                                    }
                        np.save(cur_outpath / f"{str(batch['id'][i]).zfill(6)}_target.npy",
                            dict_to_save)
                    # np.load(output_path / f"{str(batch['id'][i]).zfill(6)}.npy")
                # output_path = Path('/home/nathanasiou/Desktop/conditional_action_gen/modilex')
        logger.info(f"Sample script. The outputs are stored in:{cur_outpath}")
        
        import json
        speed_stats = {
            'inference_config': {
                'num_inference_steps': num_infer_steps,
                'use_vae': use_vae,
                'latent_shape': latent_shape,
                'diffusion_input_size': diffusion_input_size
            },
            'model_complexity': {
                'total_params': model_stats['total_params'],
                'denoiser_params': model_stats.get('denoiser_params'),
                'vae_params': model_stats.get('vae_params'),
                'flops_per_sample': flops_value
            },
            'timing_stats': timing_stats
        }
        
        speed_stats_file = cur_outpath / 'speed_statistics.json'
        with open(speed_stats_file, 'w') as f:
            json.dump(speed_stats, f, indent=2)
        logger.info(f"Speed statistics saved to: {speed_stats_file}")
        
        print("\n" + "="*70)
        print("===== GENERATION SPEED STATISTICS =====")
        print("="*70)
        
        print("\n⚙️  Inference Configuration:")
        print(f"  Inference steps:    {num_infer_steps}")
        print(f"  Use VAE:            {use_vae}")
        if latent_shape:
            print(f"  Latent shape:       {latent_shape}")
            print(f"  Diffusion input:    {diffusion_input_size} dims")
        else:
            print(f"  Diffusion input:    {diffusion_input_size}")
        
        print("\n📊 Model Complexity:")
        print(f"  Total parameters:   {model_stats['total_params']/1e6:.2f}M")
        if 'denoiser_params' in model_stats:
            print(f"  Denoiser params:    {model_stats['denoiser_params']/1e6:.2f}M")
        if 'vae_params' in model_stats:
            print(f"  VAE params:         {model_stats['vae_params']/1e6:.2f}M")
        
        if flops_value is not None:
            print(f"  FLOPs per sample:   {flops_value:.2f}G")
        
        if timing_stats['total_samples'] > 0:
            avg_time_per_sample = timing_stats['total_time'] / timing_stats['total_samples']
            print("\n⏱️  Timing Statistics:")
            print(f"  Total samples:      {timing_stats['total_samples']}")
            print(f"  Total time:         {timing_stats['total_time']:.2f} seconds")
            print(f"  Avg time/sample:    {avg_time_per_sample:.4f} seconds")
            print(f"  Throughput:         {1.0/avg_time_per_sample:.2f} samples/sec")
        
        print("\n" + "="*70)
        print(f"📁 Speed statistics saved to: {speed_stats_file}")
        
        print("\n💡 To compute evaluation metrics (FID, Retrieval, etc.), run:")
        print(f"   python compute_metrics.py folder={cur_outpath}")
        print("   (Speed statistics will be displayed together with metrics)")
        print("="*70)

if __name__ == '__main__':

    _render_vids()
