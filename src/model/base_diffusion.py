

import os
from os import times
from pathlib import Path
from contextlib import nullcontext
from typing import List, Optional, Union
from matplotlib.pylab import cond
import numpy as np
import torch
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig
from torch import Tensor, mode
from torch.distributions.distribution import Distribution
from torch.nn import ModuleDict
from src.data.tools.collate import collate_tensor_with_padding
from torch.nn import functional as F
from src.data.tools.tensors import lengths_to_mask
from src.model.base import BaseModel
from src.model.utils.tools import remove_padding
from src.model.losses.utils import LossTracker
from src.data.tools import lengths_to_mask_njoints
import inspect
from src.model.utils.tools import remove_padding, pack_to_render
from src.render.mesh_viz import render_motion
from src.tools.transforms3d import change_for, transform_body_pose, get_z_rot
from src.tools.transforms3d import apply_rot_delta
from einops import rearrange, reduce
from torch.nn.functional import l1_loss, mse_loss, smooth_l1_loss
from src.utils.genutils import dict_to_device
from src.utils.art_utils import color_map
import torch
import torch.distributions as dist
import logging
import wandb
from src.diffusion import create_diffusion

log = logging.getLogger(__name__)

class MD(BaseModel):

    def __init__(self,
                 text_encoder: DictConfig,
                 motion_condition_encoder: DictConfig,
                 denoiser: DictConfig,
                 losses: DictConfig,
                 diff_params: DictConfig,
                 latent_dim: int,
                 nfeats: int,
                 input_feats: List[str],
                 statistics_path: str,
                 dim_per_feat: List[int],
                 norm_type: str,
                 smpl_path: str,
                 render_vids_every_n_epochs: Optional[int] = None,
                 num_vids_to_render: Optional[int] = None,
                 reduce_latents: Optional[str] = None,
                 condition: Optional[str] = "text",
                 motion_condition: Optional[str] = "source",
                 loss_func_pos: str = 'mse', # l1 mse
                 loss_func_feats: str = 'mse',
                 loss_func_feats_vae: str = 'sl1',
                 renderer = None,
                 pad_inputs = False,
                 source_encoder: str = 'trans_enc',
                 zero_len_source: bool = True,
                 copy_target: bool = False,
                 old_way: bool = False,
                 motion_vae: Optional[DictConfig] = None,
                 stage: Optional[str] = None,
                 encode_source_motion: bool = True,
                 encode_target_motion: bool = True,
                 **kwargs):

        super().__init__(statistics_path, nfeats, norm_type, input_feats,
                         dim_per_feat, smpl_path, num_vids_to_render,
                         renderer=renderer)


        if set(["body_transl_delta_pelv_xy", "body_orient_delta",
                "body_pose_delta"]).issubset(self.input_feats):
            self.using_deltas = True
        else:
            self.using_deltas = False


        transl_feats = [x for x in self.input_feats if 'transl' in x]

        if set(transl_feats).issubset(["body_transl_delta", "body_transl_delta_pelv",
                                  "body_transl_delta_pelv_xy"]):
            self.using_deltas_transl = True
        else:
            self.using_deltas_transl = False

        self.zero_len_source = zero_len_source
        self.copy_target = copy_target
        self.old_way = old_way
        self.encode_source_motion = encode_source_motion
        self.encode_target_motion = encode_target_motion
        self.smpl_path = smpl_path
        self.condition = condition
        self.motion_condition = motion_condition

        if self.motion_condition == 'source':
            if source_encoder == 'trans_enc':
                self.motion_cond_encoder = instantiate(motion_condition_encoder)
            else:
                self.motion_cond_encoder = None

        self.pad_inputs = pad_inputs
        self.text_encoder = instantiate(text_encoder)




        self.input_feats = input_feats
        self.render_vids_every_n_epochs = render_vids_every_n_epochs
        self.num_vids_to_render = num_vids_to_render
        self.renderer = renderer


        self.reduce_latents = reduce_latents
        self.latent_dim = latent_dim
        self.diff_params = diff_params




        if stage is not None:
            self.stage = stage
        else:
            self.stage = 'diffusion'
        log.info(f"Training stage: {self.stage}")


        vae_pretrained_ckpt = None
        vae_pretrained_state_key = 'vae'
        vae_pretrained_strict = True
        if motion_vae is not None:
            vae_pretrained_ckpt = motion_vae.get('pretrained_ckpt', None)
            vae_pretrained_state_key = motion_vae.get('pretrained_state_key', 'vae')
            vae_pretrained_strict = motion_vae.get('pretrained_strict', True)


        if motion_vae is not None:
            self.vae = instantiate(motion_vae)
            log.info("VAE initialized successfully")
            if vae_pretrained_ckpt:




                try:
                    self._load_pretrained_vae(
                        ckpt_path=vae_pretrained_ckpt,
                        state_key=vae_pretrained_state_key,
                        strict=vae_pretrained_strict
                    )
                except FileNotFoundError as e:
                    log.warning(
                        f"No separate VAE pretrained checkpoint at '{vae_pretrained_ckpt}'; "
                        "skipping extra load, using VAE weights from the current model ckpt if present."
                    )
                    log.warning(f"[details] {e}")
        else:
            self.vae = None
            log.info("VAE not configured, using traditional TMED mode")


        denoiser.motion_condition = self.motion_condition


        if self.vae is not None:


            denoiser.use_latent_input = encode_target_motion


            vae_latent_dim = self.vae.latent_dim
            if hasattr(denoiser, 'latent_dim'):

                if isinstance(denoiser.latent_dim, list):
                    denoiser_latent_dim = denoiser.latent_dim[-1]
                else:
                    denoiser_latent_dim = denoiser.latent_dim

                if denoiser_latent_dim != vae_latent_dim:
                    log.warning(f"VAE latent_dim ({vae_latent_dim}) != denoiser latent_dim ({denoiser_latent_dim})")
                    log.warning("Setting denoiser latent_dim to match VAE latent_dim")
                    denoiser.latent_dim = vae_latent_dim

        self.denoiser = instantiate(denoiser)


        from src.diffusion import create_diffusion
        from src.diffusion.gaussian_diffusion import ModelMeanType, ModelVarType
        from src.diffusion.gaussian_diffusion import LossType











        self.diffusion_process = create_diffusion(
            timestep_respacing=None,
            learn_sigma=False,
            sigma_small=True,
            diffusion_steps=self.diff_params.num_train_timesteps,
            noise_schedule=self.diff_params.noise_schedule,
            predict_xstart=False if self.diff_params.predict_type == 'noise' else True
        )




        if 'infer_scheduler' in kwargs:
            scheduler_config = kwargs['infer_scheduler']
            scheduler_target = scheduler_config._target_ if hasattr(scheduler_config, '_target_') else None
            self.use_ddim = 'DDIM' in str(scheduler_target) if scheduler_target else False
        else:

            self.use_ddim = False


        self.ddim_eta = getattr(self.diff_params, 'ddim_eta', 0.0)

        if self.use_ddim:
            log.info(f"✅ DDIM launched [eta={self.ddim_eta}]")


        shape = 2.0
        scale = 1.0
        self.tsteps_distr = dist.Gamma(torch.tensor(shape),
                                       torch.tensor(scale))
        self.loss_params = losses
        self.enable_gen_consistency = getattr(self.loss_params, 'enable_gen_consistency', False)


        if loss_func_pos == 'l1':
            self.loss_func_pos = l1_loss
        elif loss_func_pos in ['mse', 'l2']:
            self.loss_func_pos = mse_loss
        elif loss_func_pos in ['sl1']:
            self.loss_func_pos = smooth_l1_loss


        if loss_func_feats == 'l1':
            self.loss_func_feats = l1_loss
        elif loss_func_feats in ['mse', 'l2']:
            self.loss_func_feats = mse_loss
        elif loss_func_feats in ['sl1']:
            self.loss_func_feats = smooth_l1_loss


        if loss_func_feats_vae == 'l1':
            self.loss_func_feats_vae = l1_loss
        elif loss_func_feats_vae in ['mse', 'l2']:
            self.loss_func_feats_vae = mse_loss
        elif loss_func_feats_vae in ['sl1']:
            self.loss_func_feats_vae = smooth_l1_loss
        else:

            self.loss_func_feats_vae = smooth_l1_loss

        self.validation_step_outputs = {}

        self.__post_init__()

    def _load_pretrained_vae(self,
                              ckpt_path: str,
                              state_key: Optional[str] = 'vae',
                              strict: bool = True) -> None:

        ckpt_str = os.path.expanduser(str(ckpt_path))
        try:
            ckpt_full = Path(to_absolute_path(ckpt_str))
        except ValueError:
            ckpt_full = Path(ckpt_str).resolve()

        if not ckpt_full.exists():
            raise FileNotFoundError(f"VAE pretrained checkpoint not found: {ckpt_full}")

        log.info(f"Loading pretrained VAE weights from {ckpt_full}")
        checkpoint = torch.load(ckpt_full, map_location='cpu')
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            raise ValueError(f"Unrecognized VAE checkpoint format: {ckpt_full}")

        prefix = f"{state_key}." if state_key else ""
        if prefix:
            vae_state = {
                k[len(prefix):]: v
                for k, v in state_dict.items()
                if k.startswith(prefix)
            }
        else:
            vae_state = state_dict

        if not vae_state:
            raise ValueError(
                f"No VAE weights with prefix '{state_key}' in checkpoint {ckpt_full}"
            )

        load_result = self.vae.load_state_dict(vae_state, strict=strict)
        missing, unexpected = load_result.missing_keys, load_result.unexpected_keys

        if missing:
            log.warning(f"Missing VAE state_dict keys: {missing}")
        if unexpected:
            log.warning(f"Unexpected VAE state_dict keys: {unexpected}")

        log.info("Pretrained VAE weights loaded successfully")

    def on_train_start(self):


        log.info(f"[Train Stage] {self.stage}")

        if self.vae is not None:
            log.info(f"encode_source_motion = {self.encode_source_motion}")
            log.info(f"encode_target_motion = {self.encode_target_motion}")

        if self.stage == 'diffusion' and self.vae is not None:

            self.vae.training = False
            for param in self.vae.parameters():
                param.requires_grad = False
            log.info("VAE parameters frozen for diffusion training")
        elif self.stage == 'vae' and self.vae is not None:

            self.denoiser.training = False
            for param in self.denoiser.parameters():
                param.requires_grad = False
            log.info("Denoiser parameters frozen for VAE training")


            from src.model.metrics.mr import MRMetrics
            self.mr_metrics = MRMetrics(
                njoints=22,
                jointstype="smplnh",
                force_in_meter=True,
                align_root=True,
                dist_sync_on_step=True
            )
            log.info("MRMetrics initialized for VAE training")


            from src.model.metrics.vae_fid import VAEFIDMetrics
            self.vae_fid_metrics = VAEFIDMetrics(dist_sync_on_step=True)
            log.info("VAEFIDMetrics initialized for VAE training")
        elif self.stage == 'vae_diffusion' and self.vae is not None:

            log.info("VAE and denoiser parameters active for joint training")


            from src.model.metrics.mr import MRMetrics
            self.mr_metrics = MRMetrics(
                njoints=22,
                jointstype="smplnh",
                force_in_meter=True,
                align_root=True,
                dist_sync_on_step=True
            )
            log.info("MRMetrics initialized for joint training")


            from src.model.metrics.vae_fid import VAEFIDMetrics
            self.vae_fid_metrics = VAEFIDMetrics(dist_sync_on_step=True)
            log.info("VAEFIDMetrics initialized for joint training")
        else:
            self.mr_metrics = None
            self.vae_fid_metrics = None

    def sample_from_distribution(
        self,
        dist,
        *,
        fact=None,
        sample_mean=False,
    ) -> Tensor:

        fact = fact if fact is not None else self.fact
        sample_mean = sample_mean if sample_mean is not None else self.sample_mean

        if sample_mean:

            return dist.loc.unsqueeze(0)


        if fact is None:
            return dist.rsample().unsqueeze(0)



        eps = dist.rsample() - dist.loc
        z = dist.loc + fact * eps


        z = z.unsqueeze(0)
        return z

    def _diffusion_reverse(self,
                           text_embeds, text_masks_from_enc,
                           motion_embeds, cond_motion_masks,
                           inp_motion_mask, diff_process,
                           init_vec=None,
                           init_from='noise',
                           gd_text=None, gd_motion=None,
                           mode='full_cond',
                           return_init_noise=False,
                           steps_num=None,
                           inpaint_dict=None,
                           use_linear=False,
                           prob_way='3way',
                           show_progress=True,
                           lengths=None,
                           return_latent=False,
                           use_ddim=None,
                           ddim_eta=None):

        bsz = inp_motion_mask.shape[0]
        assert mode in ['full_cond', 'text_cond', 'mot_cond']
        assert inp_motion_mask is not None


        use_ddim_sampling = self.use_ddim if use_ddim is None else use_ddim
        ddim_eta_value = self.ddim_eta if ddim_eta is None else ddim_eta


        if self.vae is None:


            if init_vec is None:
                initial_latents = torch.randn(
                    (bsz, inp_motion_mask.shape[1], self.nfeats),
                    device=inp_motion_mask.device,
                    dtype=torch.float,
                )
            else:
                initial_latents = init_vec
        else:


            latent_size = self.vae.latent_size
            latent_dim = self.vae.latent_dim


            if self.encode_target_motion:


                if lengths is None:
                    if latent_size == 1:

                        lengths = [1] * bsz
                    else:

                        if inp_motion_mask.shape[1] == 1:
                            lengths = [1] * bsz
                        else:

                            lengths = inp_motion_mask.sum(dim=1).cpu().tolist()


                if init_vec is None:
                    if init_from == 'noise':

                        initial_latents = torch.randn(
                            (bsz, latent_size, latent_dim),
                            device=inp_motion_mask.device,
                            dtype=torch.float,
                        )
                    elif init_from == 'source':

                        if motion_embeds is not None:


                            initial_latents = motion_embeds.permute(1, 0, 2)  # [B, latent_size, latent_dim]
                        else:

                            initial_latents = torch.randn(
                                (bsz, latent_size, latent_dim),
                                device=inp_motion_mask.device,
                                dtype=torch.float,
                            )
                    else:
                        initial_latents = torch.randn(
                            (bsz, latent_size, latent_dim),
                            device=inp_motion_mask.device,
                            dtype=torch.float,
                        )
                else:


                    if len(init_vec.shape) == 3 and init_vec.shape[-1] == latent_dim and init_vec.shape[1] == latent_size:

                        initial_latents = init_vec
                    elif init_from == 'source' and motion_embeds is not None:


                        initial_latents = motion_embeds.permute(1, 0, 2)  # [B, latent_size, latent_dim]
                    else:


                        initial_latents = torch.randn(
                            (bsz, latent_size, latent_dim),
                            device=inp_motion_mask.device,
                            dtype=torch.float,
                        )


                if latent_size == 1:
                    inp_motion_mask_latent = torch.ones(
                        (bsz, latent_size),
                        dtype=torch.bool,
                        device=inp_motion_mask.device
                    )
                else:

                    inp_motion_mask_latent = torch.ones(
                        (bsz, latent_size),
                        dtype=torch.bool,
                        device=inp_motion_mask.device
                    )
            else:


                if lengths is None:
                    lengths = inp_motion_mask.sum(dim=1).cpu().tolist()


                if init_vec is None:
                    if init_from == 'noise':

                        seqlen_tgt = inp_motion_mask.shape[1]
                        initial_latents = torch.randn(
                            (bsz, seqlen_tgt, self.nfeats),
                            device=inp_motion_mask.device,
                            dtype=torch.float,
                        )
                    elif init_from == 'source':

                        if motion_embeds is not None:

                            if motion_embeds.shape[-1] == latent_dim and motion_embeds.shape[0] == latent_size:


                                source_lengths = cond_motion_masks.sum(dim=1).cpu().tolist() if cond_motion_masks is not None else [motion_embeds.shape[0]] * bsz
                                motion_embeds_feat = self.decode_with_vae(motion_embeds, source_lengths)  # [B, T_src, nfeats]
                                motion_embeds_feat = motion_embeds_feat.permute(1, 0, 2)  # [B, T_src, nfeats] → [T_src, B, nfeats]
                            else:

                                motion_embeds_feat = motion_embeds  # [T_src, B, nfeats]


                            seqlen_tgt = inp_motion_mask.shape[1]
                            seqlen_src = motion_embeds_feat.shape[0]
                            if seqlen_tgt > seqlen_src:

                                padding = torch.zeros(
                                    (seqlen_tgt - seqlen_src, bsz, self.nfeats),
                                    device=motion_embeds_feat.device,
                                    dtype=motion_embeds_feat.dtype
                                )
                                initial_latents = torch.cat([motion_embeds_feat, padding], dim=0).permute(1, 0, 2)  # [B, T, nfeats]
                            else:

                                initial_latents = motion_embeds_feat[:seqlen_tgt].permute(1, 0, 2)  # [B, T, nfeats]
                        else:

                            seqlen_tgt = inp_motion_mask.shape[1]
                            initial_latents = torch.randn(
                                (bsz, seqlen_tgt, self.nfeats),
                                device=inp_motion_mask.device,
                                dtype=torch.float,
                            )
                    else:
                        seqlen_tgt = inp_motion_mask.shape[1]
                        initial_latents = torch.randn(
                            (bsz, seqlen_tgt, self.nfeats),
                            device=inp_motion_mask.device,
                            dtype=torch.float,
                        )
                else:

                    initial_latents = init_vec


                inp_motion_mask_latent = inp_motion_mask  # [B, T]






            if motion_embeds is not None:

                if motion_embeds.shape[-1] == latent_dim and motion_embeds.shape[0] == latent_size:


                    if not self.encode_target_motion:

                        if cond_motion_masks is not None:
                            source_lengths = cond_motion_masks.sum(dim=1).cpu().tolist()
                        else:
                            source_lengths = [motion_embeds.shape[0]] * bsz
                        source_feats = self.decode_with_vae(motion_embeds, source_lengths)  # [B, T_src, nfeats]
                        motion_embeds_latent = source_feats.permute(1, 0, 2)  # [B, T_src, nfeats] → [T_src, B, nfeats]
                    else:

                        motion_embeds_latent = motion_embeds
                elif self.encode_source_motion:

                    if cond_motion_masks is not None:
                        source_lengths = cond_motion_masks.sum(dim=1).cpu().tolist()
                    else:
                        source_lengths = [motion_embeds.shape[0]] * bsz

                    motion_embeds_latent, _ = self.encode_with_vae(motion_embeds, source_lengths)
                    # [latent_size, B, latent_dim]


                    if not self.encode_target_motion:
                        source_feats = self.decode_with_vae(motion_embeds_latent, source_lengths)  # [B, T_src, nfeats]
                        motion_embeds_latent = source_feats.permute(1, 0, 2)  # [B, T_src, nfeats] → [T_src, B, nfeats]
                else:

                    motion_embeds_latent = motion_embeds  # [T, B, nfeats]
            else:
                motion_embeds_latent = None




            if motion_embeds_latent is not None:

                source_is_latent = self.encode_source_motion and self.encode_target_motion

                if source_is_latent:

                    if latent_size == 1:
                        cond_motion_masks_latent = torch.ones(
                            (bsz, latent_size),
                            dtype=torch.bool,
                            device=inp_motion_mask.device
                        )
                    else:
                        cond_motion_masks_latent = torch.ones(
                            (bsz, latent_size),
                            dtype=torch.bool,
                            device=inp_motion_mask.device
                        )
                else:

                    if cond_motion_masks is not None:
                        cond_motion_masks_latent = cond_motion_masks  # [B, T]
                    else:
                        seq_len = motion_embeds_latent.shape[0]
                        cond_motion_masks_latent = torch.ones(
                            (bsz, seq_len),
                            dtype=torch.bool,
                            device=inp_motion_mask.device
                        )
            else:
                cond_motion_masks_latent = None


            if gd_text is None:
                gd_scale_text = self.diff_params.guidance_scale_text
            else:
                gd_scale_text = gd_text

            if gd_motion is None:
                gd_scale_motion = self.diff_params.guidance_scale_motion
            else:
                gd_scale_motion = gd_motion

            if text_embeds is not None:
                max_text_len = text_embeds.shape[1]
            else:
                max_text_len = 0


            if self.motion_condition == 'source' and motion_embeds_latent is not None:
                max_motion_len = cond_motion_masks_latent.shape[1]
                text_masks = text_masks_from_enc.clone()
                if self.zero_len_source or self.old_way:
                    nomotion_mask = torch.zeros(bsz, max_motion_len,
                                dtype=torch.bool).to(self.device)
                else:
                    nomotion_mask = torch.ones(bsz, max_motion_len,
                                dtype=torch.bool).to(self.device)
                motion_masks = torch.cat([nomotion_mask,
                                          cond_motion_masks_latent,
                                          cond_motion_masks_latent],
                                        dim=0)
                aug_mask = torch.cat([text_masks,
                                      motion_masks],
                                     dim=1)
            else:
                if max_text_len > 1:
                    aug_mask = text_masks_from_enc
                else:
                    aug_mask = torch.ones(2*bsz, max_text_len,
                                dtype=torch.bool).to(self.device)


            if motion_embeds_latent is not None:

                source_is_latent = self.encode_source_motion and self.encode_target_motion

                if source_is_latent:


                    motion_embeds_batch = motion_embeds_latent.permute(1, 0, 2)  # [latent_size, B, latent_dim] → [B, latent_size, latent_dim]
                else:



                    motion_embeds_batch = motion_embeds_latent.permute(1, 0, 2)  # [T, B, nfeats] → [B, T, nfeats]


                z = torch.cat([initial_latents, initial_latents, initial_latents], 0)

                model_kwargs = dict(
                    in_motion_mask=torch.cat([inp_motion_mask_latent,
                                            inp_motion_mask_latent,
                                            inp_motion_mask_latent], 0),
                    text_embeds=text_embeds,
                    condition_mask=aug_mask,
                    motion_embeds=torch.cat([torch.zeros_like(motion_embeds_batch),
                                            motion_embeds_batch,
                                            motion_embeds_batch], 0),
                    guidance_motion=gd_motion,
                    guidance_text_n_motion=gd_text,
                    inpaint_dict=inpaint_dict,
                    max_steps=max_steps_diff if use_linear else None,
                    prob_way=prob_way
                )
            else:

                z = torch.cat([initial_latents, initial_latents], 0)

                model_kwargs = dict(
                    in_motion_mask=torch.cat([inp_motion_mask_latent,
                                            inp_motion_mask_latent], 0),
                    text_embeds=text_embeds,
                    condition_mask=aug_mask,
                    motion_embeds=None,
                    guidance_motion=gd_motion,
                    guidance_text_n_motion=gd_text,
                    inpaint_dict=inpaint_dict,
                    max_steps=max_steps_diff if use_linear else None
                )


            if use_ddim_sampling:

                samples = diff_process.ddim_sample_loop(
                    self.denoiser.forward_with_guidance,
                    z.shape, z,
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                    progress=show_progress,
                    device=initial_latents.device,
                    eta=ddim_eta_value,
                )
            else:

                samples = diff_process.p_sample_loop(
                    self.denoiser.forward_with_guidance,
                    z.shape, z,
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                    progress=show_progress,
                    device=initial_latents.device,
                )


            if motion_embeds_latent is not None:
                _, _, samples = samples.chunk(3, dim=0)
            else:
                _, samples = samples.chunk(2, dim=0)


            if return_latent:
                if self.encode_target_motion:


                    return samples.permute(1, 0, 2)
                else:

                    raise ValueError("return_latent=True but encode_target_motion=False: cannot return latent when target is in feature space")


            if self.encode_target_motion:


                sampled_latent = samples.permute(1, 0, 2)  # [latent_size, B, latent_dim]
                sampled_feats = self.decode_with_vae(sampled_latent, lengths)  # [B, T, nfeats]


                sampled_feats_seq = sampled_feats.permute(1, 0, 2)  # [T, B, nfeats]
            else:


                sampled_feats_seq = samples.permute(1, 0, 2)  # [T, B, nfeats]

            if return_init_noise:
                return initial_latents, sampled_feats_seq
            else:
                return sampled_feats_seq


        if gd_text is None:
            gd_scale_text = self.diff_params.guidance_scale_text
        else:
            gd_scale_text = gd_text

        if gd_motion is None:
            gd_scale_motion = self.diff_params.guidance_scale_motion
        else:
            gd_scale_motion = gd_motion

        if text_embeds is not None:
            max_text_len = text_embeds.shape[1]
        else:
            max_text_len = 0
        if self.motion_condition == 'source' and motion_embeds is not None:
            max_motion_len = cond_motion_masks.shape[1]
            text_masks = text_masks_from_enc.clone()
            if self.zero_len_source or self.old_way:
                nomotion_mask = torch.zeros(bsz, max_motion_len,
                            dtype=torch.bool).to(self.device)
            else:
                nomotion_mask = torch.ones(bsz, max_motion_len,
                            dtype=torch.bool).to(self.device)
            motion_masks = torch.cat([nomotion_mask,
                                      cond_motion_masks,
                                      cond_motion_masks],
                                    dim=0)
            aug_mask = torch.cat([text_masks,
                                  motion_masks],
                                 dim=1)

        else:
            if max_text_len > 1:
                # aug_mask = text_mask
                # text_mask_aux = torch.ones(2*bsz, max_text_len,
                #             dtype=torch.bool).to(self.device)
                aug_mask = text_masks_from_enc
            else:
                aug_mask = torch.ones(2*bsz, max_text_len,
                            dtype=torch.bool).to(self.device)


        if motion_embeds is not None:
            z = torch.cat([initial_latents, initial_latents, initial_latents], 0)
        else:
            z = torch.cat([initial_latents, initial_latents], 0)


        # y = torch.cat([y, y_null], 0)
        if use_linear:
            max_steps_diff = diff_process.num_timesteps
        else:
            max_steps_diff = None
        if motion_embeds is not None:
            model_kwargs = dict(# noised_motion=latent_model_input,
                                # timestep=t,
                                in_motion_mask=torch.cat([inp_motion_mask,
                                                        inp_motion_mask,
                                                        inp_motion_mask], 0),
                                text_embeds=text_embeds,
                                condition_mask=aug_mask,
                                motion_embeds=torch.cat([torch.zeros_like(motion_embeds),
                                                        motion_embeds,
                                                        motion_embeds], 1),
                                guidance_motion=gd_motion,
                                guidance_text_n_motion=gd_text,
                                inpaint_dict=inpaint_dict,
                                max_steps=max_steps_diff,
                                prob_way=prob_way)
        else:
            model_kwargs = dict(# noised_motion=latent_model_input,
                    # timestep=t,
                    in_motion_mask=torch.cat([inp_motion_mask,
                                            inp_motion_mask], 0),
                    text_embeds=text_embeds,
                    condition_mask=aug_mask,
                    motion_embeds=None,
                    guidance_motion=gd_motion,
                    guidance_text_n_motion=gd_text,
                    inpaint_dict=inpaint_dict,
                    max_steps=max_steps_diff)



        if use_ddim_sampling:

            samples = diff_process.ddim_sample_loop(
                self.denoiser.forward_with_guidance,
                z.shape, z,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=show_progress,
                device=initial_latents.device,
                eta=ddim_eta_value,
            )
        else:

            samples = diff_process.p_sample_loop(
                self.denoiser.forward_with_guidance,
                z.shape, z,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=show_progress,
                device=initial_latents.device,
            )
        if motion_embeds is not None:
            _, _, samples = samples.chunk(3, dim=0)
        else:

            _, samples = samples.chunk(2, dim=0)

        # [batch_size, 1, latent_dim] -> [1, batch_size, latent_dim]
        final_diffout = samples.permute(1, 0, 2)
        if return_init_noise:
            return initial_latents, final_diffout
        else:
            return final_diffout

    def sample_timesteps(self, samples: int, sample_mode=None):

        if sample_mode is None:


            if self.trainer.current_epoch / self.trainer.max_epochs > 0.5:
                gamma_samples = self.tsteps_distr.sample((samples,))
                lower_bound = 0
                upper_bound = self.diffusion_process.num_timesteps

                scaled_samples = upper_bound * (gamma_samples / gamma_samples.max())

                timesteps_sampled = scaled_samples.floor().int().to(self.device)
            else:

                timesteps_sampled = torch.randint(0,
                                    self.diffusion_process.num_timesteps,
                                     (samples, ),
                                    device=self.device)
        else:
            if sample_mode == 'uniform':

                timesteps_sampled = torch.randint(0,
                                        self.diffusion_process.num_timesteps,
                                        (samples, ),
                                        device=self.device)
        return timesteps_sampled

    def _diffusion_process(self, input_motion_feats,
                           mask_in_mot,
                           text_encoded,
                           mask_for_condition,
                           motion_encoded=None,
                           sample=None,
                           lengths=None):


        if self.vae is not None:



            input_motion_feats_latent = input_motion_feats
        else:


            input_motion_feats_latent = input_motion_feats.permute(1, 0, 2)  # [T, B, nfeats] → [B, seq_len, nfeats]

        bsz = input_motion_feats_latent.shape[0]


        timesteps = self.sample_timesteps(samples=bsz, sample_mode='uniform')
        timesteps = timesteps.long()


        model_args = dict(
            in_motion_mask=mask_in_mot,
            text_embeds=text_encoded,
            condition_mask=mask_for_condition,
            motion_embeds=motion_encoded
        )






        diff_outs = self.diffusion_process.training_losses(
            self.denoiser,
            input_motion_feats_latent,
            timesteps,
            model_args
        )
        return diff_outs

    def train_diffusion_forward(self, batch, mask_source_motion,
                                mask_target_motion):


        if self.vae is None:


            cond_emb_motion = None
            batch_size = len(batch["text"])

            if self.motion_condition == 'source':
                source_motion_condition = batch['source_motion']
                if self.motion_cond_encoder is not None:

                    cond_emb_motion = self.motion_cond_encoder(source_motion_condition,
                                                               mask_source_motion)

                    cond_emb_motion = cond_emb_motion.unsqueeze(0)

                    mask_source_motion = torch.ones((batch_size, 1),
                                                     dtype=bool,
                                                     device=self.device)
                else:

                    cond_emb_motion = source_motion_condition

            feats_for_denois = batch['target_motion']
            target_lens = batch['length_target']
        else:

            cond_emb_motion = None
            batch_size = len(batch["text"])


            latent_size = self.vae.latent_size
            latent_dim = self.vae.latent_dim


            if self.motion_condition == 'source':
                source_motion_condition = batch['source_motion']  # [T, B, nfeats]
                source_lengths = batch.get('length_source', batch['length_target'])

                if self.encode_source_motion:

                    encode_ctx = nullcontext() if self.stage == 'vae_diffusion' else torch.no_grad()
                    with encode_ctx:
                        source_z, _ = self.encode_with_vae(source_motion_condition, source_lengths)



                    if not self.encode_target_motion:


                        source_feats = self.decode_with_vae(source_z, source_lengths)  # [B, T_src, nfeats]
                        source_feats = source_feats.permute(1, 0, 2)  # [B, T_src, nfeats] → [T_src, B, nfeats]
                        cond_emb_motion = source_feats  # [T_src, B, nfeats]

                        mask_source_motion = mask_source_motion
                    else:


                        mask_source_motion_latent = torch.ones(
                            (batch_size, latent_size),
                            dtype=torch.bool,
                            device=self.device
                        )
                        cond_emb_motion = source_z  # [latent_size, B, latent_dim]
                        mask_source_motion = mask_source_motion_latent
                else:

                    cond_emb_motion = source_motion_condition  # [T, B, nfeats]

                    mask_source_motion = mask_source_motion


            feats_for_denois = batch['target_motion']  # [T, B, nfeats]
            target_lens = batch['length_target']

            if self.encode_target_motion:

                encode_ctx = nullcontext() if self.stage == 'vae_diffusion' else torch.no_grad()
                with encode_ctx:
                    target_z, _ = self.encode_with_vae(feats_for_denois, target_lens)



                mask_target_motion_latent = torch.ones(
                    (batch_size, latent_size),
                    dtype=torch.bool,
                    device=self.device
                )


                feats_for_denois = target_z.permute(1, 0, 2)  # [latent_size, B, latent_dim] → [B, latent_size, latent_dim]
                mask_target_motion = mask_target_motion_latent
            else:



                feats_for_denois = feats_for_denois.permute(1, 0, 2)  # [T, B, nfeats] → [B, T, nfeats]

                mask_target_motion = mask_target_motion
            target_lens = batch['length_target']


        text_list = batch["text"]
        perc_uncondp = self.diff_params.prob_uncondp
        perc_drop_text = self.diff_params.prob_drop_text
        perc_drop_motion = self.diff_params.prob_drop_motion
        perc_keep_both = 1 - perc_uncondp - perc_drop_motion - perc_drop_text


        if self.vae is None:

            bs_cond = feats_for_denois.shape[1]
            if cond_emb_motion is not None:
                max_motion_len = cond_emb_motion.shape[0]

            if self.motion_condition == 'source':


                mask = (torch.rand(bs_cond, 1, 1, device=cond_emb_motion.device) > perc_drop_motion).float()
                cond_emb_motion = cond_emb_motion.permute(1, 0, 2) * mask
                cond_emb_motion = cond_emb_motion.permute(1, 0, 2)
                mask_source_motion = (mask_source_motion * mask.squeeze(-1)).bool()


                text_list = [
                    "" if np.random.rand(1) < perc_drop_text else i
                    for i in text_list
                ]


                mask_both = (torch.rand(bs_cond, 1, 1, device=cond_emb_motion.device) > (1-perc_keep_both)).float()
                zeroed_rows_indices = torch.nonzero(mask_both.squeeze() == 0).view(-1).tolist()
                for idx in zeroed_rows_indices:
                    text_list[idx] = ""
                cond_emb_motion = cond_emb_motion.permute(1, 0, 2) * mask_both
                cond_emb_motion = cond_emb_motion.permute(1, 0, 2)
                if not self.old_way:
                    mask_source_motion = (mask_source_motion * mask_both.squeeze(-1)).bool()
            else:
                text_list = [
                    "" if np.random.rand(1) < self.diff_params.prob_uncondp else i
                    for i in text_list
                ]
        else:

            bs_cond = feats_for_denois.shape[0]
            if self.motion_condition == 'source' and cond_emb_motion is not None:

                source_is_latent = self.encode_source_motion and self.encode_target_motion

                if source_is_latent:


                    mask_drop_motion = (torch.rand(bs_cond, 1, 1, device=cond_emb_motion.device) > perc_drop_motion).float()
                    mask_both = (torch.rand(bs_cond, 1, 1, device=cond_emb_motion.device) > (1-perc_keep_both)).float()

                    combined_mask = mask_drop_motion * mask_both

                    cond_emb_motion_batch = cond_emb_motion.permute(1, 0, 2)  # [latent_size, B, latent_dim] → [B, latent_size, latent_dim]
                    cond_emb_motion_batch = cond_emb_motion_batch * combined_mask
                    cond_emb_motion = cond_emb_motion_batch.permute(1, 0, 2)  # [B, latent_size, latent_dim] → [latent_size, B, latent_dim]

                    if not self.old_way:
                        mask_source_motion = (mask_source_motion * combined_mask.squeeze(-1)).bool()
                    else:
                        mask_source_motion = (mask_source_motion * mask_drop_motion.squeeze(-1)).bool()
                else:

                    mask_drop_motion = (torch.rand(bs_cond, 1, 1, device=cond_emb_motion.device) > perc_drop_motion).float()
                    mask_both = (torch.rand(bs_cond, 1, 1, device=cond_emb_motion.device) > (1-perc_keep_both)).float()
                    combined_mask = mask_drop_motion * mask_both  # [B,1,1]


                    cond_emb_motion_batch = cond_emb_motion.permute(1, 0, 2)  # [T, B, nfeats] → [B, T, nfeats]
                    combined_mask_expanded = combined_mask.expand(-1, cond_emb_motion_batch.shape[1], -1)  # [B, T, 1]
                    cond_emb_motion_batch = cond_emb_motion_batch * combined_mask_expanded
                    cond_emb_motion = cond_emb_motion_batch.permute(1, 0, 2)  # [B, T, nfeats] → [T, B, nfeats]

                    combined_mask_2d = combined_mask.squeeze(-1)  # [B,1]→[B]

                    combined_mask_2d = combined_mask_2d.expand(-1, mask_source_motion.shape[1])  # [B, T]
                    mask_source_motion = (mask_source_motion * combined_mask_2d).bool()  # [B, T]


                text_list = [
                    "" if np.random.rand(1) < perc_drop_text else i
                    for i in text_list
                ]


                zeroed_rows_indices = torch.nonzero(mask_both.squeeze() == 0).view(-1).tolist()
                for idx in zeroed_rows_indices:
                    text_list[idx] = ""
            else:

                text_list = [
                    "" if np.random.rand(1) < self.diff_params.prob_uncondp else i
                    for i in text_list
                ]


        cond_emb_text, text_mask = self.text_encoder(text_list)
        max_text_len = cond_emb_text.shape[1]


        if self.motion_condition == 'source':
            aug_mask = torch.cat([
                text_mask if max_text_len > 1 else torch.ones_like(text_mask),
                mask_source_motion
            ], dim=1).to(self.device)
        else:
            if max_text_len > 1:
                aug_mask = text_mask
            else:
                aug_mask = torch.ones(batch_size, max_text_len,
                                      dtype=torch.bool, device=self.device)




        diff_outs = self._diffusion_process(
            input_motion_feats=feats_for_denois,
            mask_in_mot=mask_target_motion,
            text_encoded=cond_emb_text,
            motion_encoded=cond_emb_motion,
            mask_for_condition=aug_mask
        )

        diff_outs['motion_mask_target'] = mask_target_motion
        return diff_outs

    def training_step(self, batch, batch_idx):
        return self.allsplit_step("train", batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.allsplit_step("val", batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.allsplit_step("test", batch, batch_idx)

    def compute_losses(self, out_dict, dataset_names):

        from torch import nn
        from src.data.tools.tensors import lengths_to_mask

        pad_mask = out_dict['motion_mask_target']
        all_losses_dict = {}


        data_loss = self.loss_func_feats(
            out_dict['target'],
            out_dict['model_output'],
            reduction='none'
        )


        if self.vae is not None and self.stage in ['diffusion', 'vae_diffusion']:


            if self.encode_target_motion:




                # data_loss: [B, latent_size, latent_dim]
                # pad_mask: [B, latent_size]
                latent_loss_per_dim = data_loss.mean(-1)
                masked_loss = latent_loss_per_dim * pad_mask

                valid_count = pad_mask.sum()
                if valid_count > 0:
                    tot_loss = masked_loss.sum() / valid_count.float()
                else:
                    tot_loss = torch.tensor(0.0, device=self.device)
            else:

                f_rg = np.cumsum([0] + self.input_feats_dims)
                tot_loss = torch.tensor(0.0, device=self.device)
                full_feature_loss = data_loss
                valid_feat_counter = 0


                for i, _ in enumerate(f_rg[:-1]):
                    if 'delta' in self.input_feats[i]:
                        cur_mask = pad_mask[:, 1:]                       # [B, S-1]
                        cur_feat_loss = full_feature_loss[:, 1:, f_rg[i]:f_rg[i+1]].mean(-1) * cur_mask
                    else:
                        cur_mask = pad_mask                              # [B, S]
                        cur_feat_loss = full_feature_loss[..., f_rg[i]:f_rg[i+1]].mean(-1) * cur_mask

                    valid_count = cur_mask.sum()
                    if valid_count <= 0:
                        all_losses_dict.update({self.input_feats[i]: torch.tensor(0.0, device=self.device)})
                        continue

                    valid_feat_counter += 1
                    tot_feat_loss = cur_feat_loss.sum() / valid_count.float()


                    all_losses_dict.update({self.input_feats[i]: tot_feat_loss})

                    tot_loss += tot_feat_loss

                tot_loss /= max(valid_feat_counter, 1)


        else:
            f_rg = np.cumsum([0] + self.input_feats_dims)
            tot_loss = torch.tensor(0.0, device=self.device)
            full_feature_loss = data_loss
            valid_feat_counter = 0


            for i, _ in enumerate(f_rg[:-1]):
                if 'delta' in self.input_feats[i]:
                    cur_mask = pad_mask[:, 1:]                       # [B, S-1]
                    cur_feat_loss = full_feature_loss[:, 1:, f_rg[i]:f_rg[i+1]].mean(-1) * cur_mask
                else:
                    cur_mask = pad_mask                              # [B, S]
                    cur_feat_loss = full_feature_loss[..., f_rg[i]:f_rg[i+1]].mean(-1) * cur_mask

                valid_count = cur_mask.sum()
                if valid_count <= 0:
                    all_losses_dict.update({self.input_feats[i]: torch.tensor(0.0, device=self.device)})
                    continue

                valid_feat_counter += 1
                tot_feat_loss = cur_feat_loss.sum() / valid_count.float()


                all_losses_dict.update({self.input_feats[i]: tot_feat_loss})

                tot_loss += tot_feat_loss

            tot_loss /= max(valid_feat_counter, 1)

        all_losses_dict['total_loss'] = tot_loss
        return tot_loss, all_losses_dict

    def generate_motion(self, texts_cond, motions_cond,
                        mask_source, mask_target,
                        diffusion_process,
                        lengths=None,
                        init_vec_method='noise', init_vec=None,
                        gd_text=None, gd_motion=None,
                        return_init_noise=False,
                        condition_mode='full_cond',
                        num_diff_steps=None,
                        inpaint_dict=None,
                        use_linear=False,
                        prob_way='3way',
                        show_progress=True,
                        use_ddim=None,
                        ddim_eta=None,
                        allow_grad: bool = False
                        ):


        bsz, seqlen_tgt = mask_target.shape

        feat_sz = sum(self.input_feats_dims)


        if texts_cond is not None:

            no_of_texts = len(texts_cond)

            texts_cond = [''] * no_of_texts + texts_cond
            if self.motion_condition == 'source':

                texts_cond = [''] * no_of_texts + texts_cond

            text_emb, text_mask = self.text_encoder(texts_cond)


        cond_emb_motion = None
        cond_motion_mask = None
        if self.motion_condition == 'source':

            bsz, seqlen_src = mask_source.shape

            if condition_mode == 'full_cond' or condition_mode == 'mot_cond' :

                if motions_cond is not None and self.vae is not None:
                    if self.encode_source_motion:

                        source_lengths = mask_source.sum(dim=1).cpu().tolist()
                        motions_cond_latent, _ = self.encode_with_vae(motions_cond, source_lengths)

                        cond_emb_motion = motions_cond_latent
                        if self.vae.latent_size == 1:
                            cond_motion_mask = torch.ones((bsz, 1),
                                                          dtype=bool, device=self.device)
                        else:
                            cond_motion_mask = torch.ones((bsz, self.vae.latent_size),
                                                          dtype=bool, device=self.device)
                    else:

                        cond_emb_motion = motions_cond  # [T, B, nfeats]
                        cond_motion_mask = mask_source  # [B, T]
                elif self.motion_cond_encoder is not None:

                    source_motion_condition = motions_cond
                    cond_emb_motion = self.motion_cond_encoder(source_motion_condition, mask_source)

                    cond_emb_motion = cond_emb_motion.unsqueeze(0)

                    cond_motion_mask = torch.ones((mask_source.shape[0], 1),
                                                  dtype=bool, device=self.device)
                else:

                    source_motion_condition = motions_cond
                    cond_emb_motion = source_motion_condition
                    cond_motion_mask = mask_source
            else:

                if self.vae is not None:

                    if self.vae.latent_size == 1:
                        cond_emb_motion = torch.zeros(1, bsz,
                                                      self.vae.latent_dim,
                                                      device=self.device)
                        cond_motion_mask = torch.ones((bsz, 1),
                                                      dtype=bool, device=self.device)
                    else:
                        cond_emb_motion = torch.zeros(self.vae.latent_size, bsz,
                                                      self.vae.latent_dim,
                                                      device=self.device)
                        cond_motion_mask = torch.ones((bsz, self.vae.latent_size),
                                                      dtype=bool, device=self.device)
                elif self.motion_cond_encoder is not None:

                    cond_emb_motion = torch.zeros(1, bsz,
                                                  self.denoiser.latent_dim,
                                                   device=self.device)
                    cond_motion_mask = torch.ones((bsz, 1),
                                                  dtype=bool, device=self.device)
                else:

                    cond_emb_motion = torch.zeros(seqlen_src, bsz, feat_sz,
                                                  device=self.device)
                    cond_motion_mask = torch.ones((bsz, 1),
                                                dtype=bool, device=self.device)


        if init_vec_method == 'noise_prev':

            init_diff_rev = init_vec
        elif init_vec_method == 'source':

            if self.vae is not None:

                if cond_emb_motion is not None:
                    if self.encode_target_motion:

                        if self.encode_source_motion:

                            init_diff_rev = cond_emb_motion.permute(1, 0, 2)  # [B, latent_size, latent_dim]
                            logger = logging.getLogger(__name__)
                            logger.info("VAE-TMED: init from source motion latent (target latent space)")
                        else:

                            source_lengths = mask_source.sum(dim=1).cpu().tolist()
                            source_z, _ = self.encode_with_vae(motions_cond, source_lengths)
                            init_diff_rev = source_z.permute(1, 0, 2)  # [B, latent_size, latent_dim]
                            logger = logging.getLogger(__name__)
                            logger.info("VAE-TMED: init from encoded source features (target latent space)")
                    else:

                        if self.encode_source_motion:

                            source_lengths = mask_source.sum(dim=1).cpu().tolist()

                            source_feats = self.decode_with_vae(cond_emb_motion, source_lengths)  # [B, T_src, nfeats]
                            source_feats = source_feats.permute(1, 0, 2)  # [B, T_src, nfeats] → [T_src, B, nfeats]


                            tgt_len = mask_target.shape[-1]
                            src_len = source_feats.shape[0]
                            if tgt_len > src_len:

                                padding = torch.zeros(
                                    (tgt_len - src_len, bsz, self.nfeats),
                                    device=source_feats.device,
                                    dtype=source_feats.dtype
                                )
                                init_diff_rev = torch.cat([source_feats, padding], dim=0).permute(1, 0, 2)  # [B, T, nfeats]
                            else:

                                init_diff_rev = source_feats[:tgt_len].permute(1, 0, 2)  # [B, T, nfeats]
                            logger = logging.getLogger(__name__)
                            logger.info("VAE-TMED: init from decoded source latent (target feature space)")
                        else:

                            init_diff_rev = motions_cond
                            tgt_len = mask_target.shape[-1]
                            src_len = mask_source.shape[-1]
                            if tgt_len > src_len:

                                init_diff_rev = torch.cat([init_diff_rev,
                                                          torch.zeros((tgt_len-src_len,
                                                                       *init_diff_rev.shape[1:]),
                                                                      device=self.device)],
                                                         dim=0)
                                init_diff_rev = init_diff_rev.permute(1, 0, 2)  # [seq, batch, feat]→[batch, seq, feat]
                            else:

                                init_diff_rev = init_diff_rev[:tgt_len]
                                init_diff_rev = init_diff_rev.permute(1, 0, 2)
                            logger = logging.getLogger(__name__)
                            logger.info("VAE-TMED: init from source motion features (target feature space)")
                else:

                    logger = logging.getLogger(__name__)
                    logger.warning(
                        "VAE-TMED: init_vec_method='source' but cond_emb_motion unavailable; falling back to noise init"
                    )
                    init_diff_rev = None
            else:

                init_diff_rev = motions_cond
                tgt_len = mask_target.shape[-1]
                src_len = mask_source.shape[-1]
                if tgt_len > src_len:

                    init_diff_rev = torch.cat([init_diff_rev,
                                              torch.zeros((tgt_len-src_len,
                                                           *init_diff_rev.shape[1:]),
                                                          device=self.device)],
                                             dim=0)
                    init_diff_rev = init_diff_rev.permute(1, 0, 2)  # [seq, batch, feat]→[batch, seq, feat]
                else:

                    init_diff_rev = init_diff_rev[:tgt_len]
                    init_diff_rev = init_diff_rev.permute(1, 0, 2)

                logger = logging.getLogger(__name__)
                logger.info("TMED: init from source motion features")
        else:

            init_diff_rev = None



        if lengths is not None:
            target_lengths = lengths
        else:
            target_lengths = mask_target.sum(dim=1).cpu().tolist() if mask_target is not None else None


        ctx = nullcontext() if allow_grad else torch.no_grad()
        with ctx:
            if return_init_noise:

                init_noise, diff_out = self._diffusion_reverse(
                    text_emb,
                    text_mask,
                    cond_emb_motion,
                    cond_motion_mask,
                    mask_target,
                    diffusion_process,
                    init_vec=init_diff_rev,
                    init_from=init_vec_method,
                    gd_text=gd_text,
                    gd_motion=gd_motion,
                    return_init_noise=return_init_noise,
                    mode=condition_mode,
                    steps_num=num_diff_steps,
                    inpaint_dict=inpaint_dict,
                    use_linear=use_linear,
                    prob_way=prob_way,
                    show_progress=show_progress,
                    lengths=target_lengths,
                    return_latent=False,
                    use_ddim=use_ddim,
                    ddim_eta=ddim_eta
                )


                diff_out = diff_out.permute(1, 0, 2)
                return init_noise, diff_out
            else:

                diff_out = self._diffusion_reverse(
                    text_emb,
                    text_mask,
                    cond_emb_motion,
                    cond_motion_mask,
                    mask_target,
                    diffusion_process,
                    init_vec=init_diff_rev,
                    init_from=init_vec_method,
                    gd_text=gd_text,
                    gd_motion=gd_motion,
                    return_init_noise=return_init_noise,
                    mode=condition_mode,
                    steps_num=num_diff_steps,
                    inpaint_dict=inpaint_dict,
                    use_linear=use_linear,
                    show_progress=show_progress,
                    lengths=target_lengths,
                    return_latent=False,
                    use_ddim=use_ddim,
                    ddim_eta=ddim_eta
                )

            diff_out = diff_out.permute(1, 0, 2)
            return diff_out

    # def integrate_feats2motion(self, first_pose_norm, delta_motion_norm):
    #     """"
    #     Given a state [translation, orientation, pose] and state deltas,
    #     properly calculate the next state
    #     input and output are normalised features hence we first unnormalise,
    #     perform the calculatios and then normalise again
    #     """
    #     # unnorm features

    #     first_pose = self.unnorm_state(first_pose_norm)
    #     delta_motion = self.unnorm_delta(delta_motion_norm)

    #     # apply deltas
    #     # get velocity in global c.f. and add it to the state position
    #     assert 'body_transl_delta_pelv_xy' in self.input_feats
    #     pelvis_orient = first_pose[..., 3:9]
    #     R_z = get_z_rot(pelvis_orient, in_format="6d")

    #     # rotate R_z
    #     root_vel = change_for(delta_motion[..., :3],
    #                           R_z.squeeze(), forward=False)

    #     new_state_pos = first_pose[..., :3].squeeze() + root_vel

    #     # apply rotational deltas
    #     new_state_rot = apply_rot_delta(first_pose[..., 3:].squeeze(),
    #                                     delta_motion[..., 3:],
    #                                     in_format="6d", out_format="6d")

    #     # cat and normalise the result
    #     new_state = torch.cat((new_state_pos, new_state_rot), dim=-1)
    #     new_state_norm = self.norm_state(new_state)
    #     return new_state_norm


    # def integrate_translation(self, pelv_orient_norm, first_trans,
    #                           delta_transl_norm):
    #     """"
    #     Given a state [translation, orientation, pose] and state deltas,
    #     properly calculate the next state
    #     input and output are normalised features hence we first unnormalise,
    #     perform the calculatios and then normalise again
    #     """
    #     # B, S, 6d
    #     pelv_orient_unnorm = self.cat_inputs(self.unnorm_inputs(
    #                                             [pelv_orient_norm],
    #                                             ['body_orient'])
    #                                          )[0]
    #     # B, S, 3
    #     delta_trans_unnorm = self.cat_inputs(self.unnorm_inputs(
    #                                             [delta_transl_norm],
    #                                             ['body_transl_delta_pelv'])
    #                                             )[0]
    #     # B, 1, 3
    #     first_trans = self.cat_inputs(self.unnorm_inputs(
    #                                             [first_trans],
    #                                             ['body_transl'])
    #                                       )[0]

    #     # apply deltas
    #     # get velocity in global c.f. and add it to the state position
    #     assert 'body_transl_delta_pelv' in self.input_feats
    #     pelv_orient_unnorm_rotmat = transform_body_pose(pelv_orient_unnorm,
    #                                                     "6d->rot")
    #     trans_vel_pelv = change_for(delta_trans_unnorm,
    #                                 pelv_orient_unnorm_rotmat,
    #                                 forward=False)

    #     # new_state_pos = prev_trans_norm.squeeze() + trans_vel_pelv
    #     full_trans_unnorm = torch.cumsum(trans_vel_pelv,
    #                                       dim=1) + first_trans
    #     full_trans_unnorm = torch.cat([first_trans,
    #                                    full_trans_unnorm], dim=1)
    #     return full_trans_unnorm

    def diffout2motion(self, diffout):

        if diffout.shape[1] == 1:

            rots_unnorm = self.cat_inputs(
                self.unnorm_inputs(
                    self.uncat_inputs(
                        diffout,
                        self.input_feats_dims
                    ),
                    self.input_feats
                )
            )[0]

            full_motion_unnorm = rots_unnorm
        else:

            # - "body_transl_delta_pelv_xy_wo_z"
            # - "body_transl_z"
            # - "z_orient_delta"
            # - "body_orient_xy"
            # - "body_pose"
            # - "body_joints_local_wo_z_rot"

            feats_unnorm = self.cat_inputs(
                self.unnorm_inputs(
                    self.uncat_inputs(diffout, self.input_feats_dims),
                    self.input_feats
                )
            )[0]


            if "body_joints_local_wo_z_rot" in self.input_feats:
                idx = self.input_feats.index("body_joints_local_wo_z_rot")
                feats_unnorm = feats_unnorm[..., :-self.input_feats_dims[idx]]


            first_trans = torch.zeros(*diffout.shape[:-1], 3, device=self.device)[:, [0]]


            if 'z_orient_delta' in self.input_feats:

                first_orient_z = torch.eye(3, device=self.device).unsqueeze(0)
                first_orient_z = first_orient_z.repeat(feats_unnorm.shape[0], 1, 1)
                first_orient_z = transform_body_pose(first_orient_z, 'rot->6d')


                z_orient_delta = feats_unnorm[..., 9:15]

                from src.tools.transforms3d import apply_rot_delta, remove_z_rot, get_z_rot, change_for
                prev_z = first_orient_z
                full_z_angle = [first_orient_z[:, None]]  # (B,1,6)

                for i in range(1, z_orient_delta.shape[1]):
                    curr_z = apply_rot_delta(prev_z, z_orient_delta[:, i])
                    prev_z = curr_z.clone()
                    full_z_angle.append(curr_z[:, None])
                full_z_angle = torch.cat(full_z_angle, dim=1)  # (B, T, 6)
                full_z_angle_rotmat = get_z_rot(full_z_angle)

                xy_orient = feats_unnorm[..., 3:9]
                xy_orient_rotmat = transform_body_pose(xy_orient, '6d->rot')

                full_global_orient_rotmat = full_z_angle_rotmat @ xy_orient_rotmat
                full_global_orient = transform_body_pose(full_global_orient_rotmat, 'rot->6d')


                first_trans = self.cat_inputs(
                    self.unnorm_inputs([first_trans], ['body_transl'])
                )[0]


                assert 'body_transl_delta_pelv' in self.input_feats

                pelvis_delta = feats_unnorm[..., :3]

                trans_vel_pelv = change_for(
                    pelvis_delta[:, 1:],
                    full_global_orient_rotmat[:, :-1],
                    forward=False
                )

                full_trans = torch.cumsum(trans_vel_pelv, dim=1) + first_trans

                full_trans = torch.cat([first_trans, full_trans], dim=1)


                full_rots = torch.cat(
                    [full_global_orient, feats_unnorm[...,-21*6:]],
                    dim=-1
                )

                full_motion_unnorm = torch.cat([full_trans, full_rots], dim=-1)


            elif "body_orient_delta" in self.input_feats:

                delta_trans = diffout[..., 6:9]
                pelv_orient = diffout[..., 9:15]

                full_trans_unnorm = self.integrate_translation(
                    pelv_orient[:, :-1],
                    first_trans,
                    delta_trans[:, 1:]
                )

                rots_unnorm = self.cat_inputs(
                    self.unnorm_inputs(
                        self.uncat_inputs(
                            diffout[..., 9:],
                            self.input_feats_dims[2:]
                        ),
                        self.input_feats[2:]
                    )
                )[0]

                full_motion_unnorm = torch.cat([full_trans_unnorm, rots_unnorm], dim=-1)


            else:
                delta_trans = diffout[..., :3]
                pelv_orient = diffout[..., 3:9]

                full_trans_unnorm = self.integrate_translation(
                    pelv_orient[:, :-1],
                    first_trans,
                    delta_trans[:, 1:]
                )

                rots_unnorm = self.cat_inputs(
                    self.unnorm_inputs(
                        self.uncat_inputs(
                            diffout[..., 3:],
                            self.input_feats_dims[1:]
                        ),
                        self.input_feats[1:]
                    )
                )[0]
                full_motion_unnorm = torch.cat([full_trans_unnorm, rots_unnorm], dim=-1)

        return full_motion_unnorm

    def _ensure_batch_first(self, feats: Tensor, lengths: Optional[List[int]] = None) -> Tensor:

        if feats.dim() != 3:
            return feats


        if lengths is not None:
            batch_size = len(lengths)

            if feats.shape[1] == batch_size and feats.shape[0] != batch_size:
                return feats.permute(1, 0, 2)  # [B, T, nfeats]

            elif feats.shape[0] == batch_size:
                return feats

            elif feats.shape[0] > feats.shape[1] and feats.shape[1] == batch_size:
                return feats.permute(1, 0, 2)  # [B, T, nfeats]


        return feats

    def feats2joints(self, feats: Tensor, lengths: Optional[List[int]] = None):


        B, T = feats.shape[:2]



        full_motion_unnorm = self.diffout2motion(feats)  # [B, T, 3+22*6] = [B, T, 135]



        body_transl = full_motion_unnorm[..., :3]  # [B, T, 3]
        body_orient = full_motion_unnorm[..., 3:9]  # [B, T, 6]
        body_pose = full_motion_unnorm[..., 9:135]






        from src.tools.transforms3d import transform_body_pose
        from src.info.joints import smplh2smplnh_indexes


        B_actual, T_actual = B, T
        body_transl_flat = body_transl.reshape(B_actual * T_actual, 3)
        body_orient_flat = body_orient.reshape(B_actual * T_actual, 6)
        body_pose_flat = body_pose.reshape(B_actual * T_actual, 126)


        self.body_model.batch_size = B_actual * T_actual



        smpl_output = self.body_model.smpl_forward_fast(
            transl=body_transl_flat,
            body_pose=transform_body_pose(body_pose_flat, '6d->rot'),
            global_orient=transform_body_pose(body_orient_flat, '6d->rot'),
            return_verts=False
        )


        all_joints = smpl_output.joints  # [B*T, n_joints, 3]


        n_joints = all_joints.shape[1]
        import logging
        logger = logging.getLogger(__name__)

        if n_joints == 22:

            joints_22 = all_joints  # [B_actual*T_actual, 22, 3]
        elif n_joints == 24:

            joints_22 = all_joints[:, smplh2smplnh_indexes, :]  # [B_actual*T_actual, 22, 3]
        elif n_joints == 52:


            if max(smplh2smplnh_indexes) < 52:
                joints_22 = all_joints[:, smplh2smplnh_indexes, :]  # [B_actual*T_actual, 22, 3]
            else:

                logger.warning(
                    f"feats2joints: smplh2smplnh_indexes out of range for 52 joints. "
                    f"Using first 22 joints instead."
                )
                joints_22 = all_joints[:, :22, :]  # [B_actual*T_actual, 22, 3]
        elif n_joints >= 73:



            logger.warning(
                f"feats2joints: SMPL-H returned {n_joints} joints (expected 22, 24, or 52). "
                f"Using first 22 joints (base joints)."
            )

            if max(smplh2smplnh_indexes) < n_joints:
                joints_22 = all_joints[:, smplh2smplnh_indexes, :]  # [B_actual*T_actual, 22, 3]
            else:

                joints_22 = all_joints[:, :22, :]  # [B_actual*T_actual, 22, 3]
        else:

            logger.warning(
                f"feats2joints: Unexpected number of joints: {n_joints}, "
                f"expected 22, 24, 52, or >=73. Attempting to extract 22 joints."
            )

            if max(smplh2smplnh_indexes) < n_joints:
                joints_22 = all_joints[:, smplh2smplnh_indexes, :]  # [B_actual*T_actual, 22, 3]
            else:

                n_extract = min(22, n_joints)
                logger.warning(
                    f"smplh2smplnh_indexes out of range for {n_joints} joints. "
                    f"Using first {n_extract} joints instead."
                )
                joints_22 = all_joints[:, :n_extract, :]  # [B_actual*T_actual, n_extract, 3]

                if n_extract < 22:
                    raise RuntimeError(
                        f"Cannot extract 22 joints from {n_joints} joints. "
                        f"Only {n_extract} joints available."
                    )



        if len(joints_22.shape) != 3 or joints_22.shape[1] != 22 or joints_22.shape[2] != 3:
            raise RuntimeError(
                f"joints_22 shape error: expected [B*T, 22, 3], got {joints_22.shape}, "
                f"all_joints shape: {all_joints.shape}, "
                f"input feats shape: {feats.shape}"
            )


        total_elements = joints_22.numel()
        expected_elements = B_actual * T_actual * 22 * 3

        if total_elements != expected_elements:



            T_calculated = total_elements // (B_actual * 22 * 3)
            if T_calculated * B_actual * 22 * 3 == total_elements:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(
                    f"feats2joints: T dimension mismatch - "
                    f"vertices T={T_actual}, calculated T={T_calculated} from joints_22"
                )
                T_actual = T_calculated
            else:

                B_calculated = total_elements // (T_actual * 22 * 3)
                if B_calculated * T_actual * 22 * 3 == total_elements:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(
                        f"feats2joints: B dimension mismatch - "
                        f"vertices B={B_actual}, calculated B={B_calculated} from joints_22"
                    )
                    B_actual = B_calculated
                else:

                    import logging
                    logger = logging.getLogger(__name__)
                    logger.error(
                        f"feats2joints: Cannot reshape joints_22 - "
                        f"total_elements={total_elements}, "
                        f"expected={expected_elements}, "
                        f"B_actual={B_actual}, T_actual={T_actual}, "
                        f"all_joints shape={all_joints.shape}, "
                        f"joints_22 shape={joints_22.shape}, "
                        f"input feats shape={feats.shape}"
                    )
                    raise RuntimeError(
                        f"Cannot reshape joints_22: total_elements={total_elements}, "
                        f"expected for [B={B_actual}, T={T_actual}, 22, 3]={expected_elements}, "
                        f"input feats shape={feats.shape}, "
                        f"all_joints shape={all_joints.shape}, "
                        f"joints_22 shape={joints_22.shape}"
                    )

        joints_reshaped = joints_22.reshape(B_actual, T_actual, 22, 3)


        if B_actual != B or T_actual != T:

            if B_actual > B:
                joints_reshaped = joints_reshaped[:B, :, :, :]
                B_actual = B
            elif B_actual < B:

                padding = joints_reshaped[-1:, :, :, :].repeat(B - B_actual, 1, 1, 1)
                joints_reshaped = torch.cat([joints_reshaped, padding], dim=0)
                B_actual = B


            if T_actual > T:

                joints_reshaped = joints_reshaped[:, :T, :, :]
                T_actual = T
            elif T_actual < T:

                padding = joints_reshaped[:, -1:, :, :].repeat(1, T - T_actual, 1, 1)
                joints_reshaped = torch.cat([joints_reshaped, padding], dim=1)
                T_actual = T


        joints = joints_reshaped.reshape(B, T, 22, 3)

        return joints  # [B, T, 22, 3]

    def allsplit_step(self, split: str, batch, batch_idx):



        if split in ['val', 'test'] and batch_idx == 0:

            sampler_name = 'DDIM' if self.use_ddim else 'DDPM'

            infer_steps_cfg = getattr(self.diff_params, 'num_inference_timesteps', self.diffusion_process.num_timesteps)
            log.info(
                f"[{split.upper()}] sampler: {sampler_name}, inference steps: {infer_steps_cfg}, DDIM eta: {self.ddim_eta}"
            )

        if self.stage == 'vae':

            return self._allsplit_step_vae(split, batch, batch_idx)
        elif self.stage == 'diffusion':

            return self._allsplit_step_diffusion(split, batch, batch_idx)
        elif self.stage == 'vae_diffusion':

            return self._allsplit_step_vae_diffusion(split, batch, batch_idx)
        else:
            raise ValueError(f"Unknown training stage: {self.stage}")

    def _allsplit_step_vae_diffusion(self, split: str, batch, batch_idx):

        from src.data.tools.tensors import lengths_to_mask


        input_batch = self.norm_and_cat(batch, self.input_feats)
        for k, v in input_batch.items():
            batch[f'{k}_motion'] = v


        if self.motion_condition is not None:
            if self.pad_inputs:
                mask_source, mask_target = self.prepare_mot_masks(
                    batch['length_source'],
                    batch['length_target'],
                    max_len=300
                )
            else:
                mask_source, mask_target = self.prepare_mot_masks(
                    batch['length_source'],
                    batch['length_target'],
                    max_len=None
                )
        else:

            mask_target = lengths_to_mask(batch['length_target'], device=self.device)
            if self.pad_inputs:
                mask_target = F.pad(mask_target, (0, 300 - mask_target.size(1)), value=0)
            batch['length_source'] = None
            batch['source_motion'] = None
            mask_source = None


        vae_out = self.train_vae_step(batch, batch_idx, split=split)
        if split == 'val':
            vae_loss = vae_out['loss']
            target_feats_recon = vae_out['target_feats_recon']
            source_feats_recon = vae_out['source_feats_recon']
        else:
            vae_loss = vae_out
            target_feats_recon, source_feats_recon = None, None


        dif_dict = self.train_diffusion_forward(
            batch,
            mask_source,
            mask_target
        )
        diff_loss, diff_loss_dict = self.compute_losses(dif_dict, batch['dataset_name'])


        total_loss = vae_loss + diff_loss


        loss_dict_to_log = {
            f'losses/{split}/vae': vae_loss,
            f'losses/{split}/diffusion': diff_loss,
            f'losses/{split}/total': total_loss,
        }


        if self.enable_gen_consistency:

            texts_cond = [el.lower() for el in batch["text"]]
            motions_cond = batch.get('source_motion', None)

            gd_text = getattr(self.diff_params, 'guidance_scale_text', 1.0)
            gd_motion = getattr(self.diff_params, 'guidance_scale_motion', 1.0)

            infer_steps = getattr(self.diff_params, 'num_inference_timesteps', self.diffusion_process.num_timesteps)
            gen_diffout = self.generate_motion(
                texts_cond,
                motions_cond,
                mask_source,
                mask_target,
                self.diffusion_process,
                num_diff_steps=infer_steps,
                gd_text=gd_text,
                gd_motion=gd_motion,
                show_progress=False,
                lengths=batch['length_target'],
                allow_grad=(split == 'train')
            )

            gen_feats_bt = gen_diffout                     # [B, S, nfeats]


            tgt_feats_bt = batch['target_motion'].permute(1, 0, 2)

            min_len = min(gen_feats_bt.shape[1], tgt_feats_bt.shape[1])
            if min_len != gen_feats_bt.shape[1] or min_len != tgt_feats_bt.shape[1]:
                gen_feats_bt = gen_feats_bt[:, :min_len, :]
                tgt_feats_bt = tgt_feats_bt[:, :min_len, :]
            tgt_lengths = [min(l, min_len) for l in batch['length_target']]

            gen_feature_loss = self.loss_func_feats_vae(gen_feats_bt, tgt_feats_bt)

            joints_gen = self.feats2joints(gen_feats_bt, tgt_lengths)
            joints_ref = self.feats2joints(tgt_feats_bt, tgt_lengths)
            from torch.nn import SmoothL1Loss
            mask = lengths_to_mask(tgt_lengths, device=gen_feats_bt.device)
            B, T, nj, _ = joints_gen.shape
            joints_gen_f = joints_gen.reshape(B * T, nj, 3)
            joints_ref_f = joints_ref.reshape(B * T, nj, 3)
            joints_gen_f = joints_gen_f[mask.reshape(B * T)]
            joints_ref_f = joints_ref_f[mask.reshape(B * T)]
            gen_joints_loss = SmoothL1Loss(reduction='mean')(joints_gen_f, joints_ref_f)

            lmd_gen = getattr(self.loss_params, 'lmd_gen', 1.0)
            total_loss = total_loss + lmd_gen * (gen_feature_loss + gen_joints_loss)

            loss_dict_to_log.update({
                f'losses/{split}/gen_feature': gen_feature_loss,
                f'losses/{split}/gen_joints': gen_joints_loss,
            })


        for k, v in diff_loss_dict.items():
            loss_dict_to_log[f'diff/{split}/{k}'] = v
        self.log_dict(loss_dict_to_log,
                      on_epoch=True,
                      on_step=True,
                      batch_size=len(batch['length_target']),
                      sync_dist=True,
                      rank_zero_only=True)


        if split == 'val':
            from src.data.tools.tensors import lengths_to_mask

            infer_steps = getattr(self.diff_params, 'num_inference_timesteps', self.diffusion_process.num_timesteps)

            if infer_steps != self.diffusion_process.num_timesteps:
                from src.diffusion import create_diffusion
                inference_diffusion_process = create_diffusion(
                    timestep_respacing=None,
                    learn_sigma=False,
                    sigma_small=True,
                    diffusion_steps=infer_steps,
                    noise_schedule=self.diff_params.noise_schedule,
                    predict_xstart=False if self.diff_params.predict_type == 'noise' else True
                )
            else:
                inference_diffusion_process = self.diffusion_process

            guidances_mix = [(2.0, 2.0), (2.0, 4.5)]
            gt_keyids = batch.get('id', [f'val_{batch_idx}_{i}' for i in range(len(batch['length_target']))])
            gt_texts = [el.lower() for el in batch["text"]]


            if batch_idx == 0:
                self.validation_step_outputs = {
                    f'{s_t}txt_{s_m}mot': {} for s_t, s_m in guidances_mix
                }


            for guid_text, guid_motion in guidances_mix:
                diffout = self.generate_motion(
                    gt_texts,
                    batch.get('source_motion', None),
                    mask_source,
                    mask_target,
                    inference_diffusion_process,
                    gd_motion=guid_motion,
                    gd_text=guid_text,
                    num_diff_steps=infer_steps,
                    show_progress=False,
                    lengths=batch['length_target']
                )
                gen_mo = self.diffout2motion(diffout)    # [B, T, feat]
                for ii, kval in enumerate(gt_keyids):
                    self.validation_step_outputs[f'{guid_text}txt_{guid_motion}mot'][kval] = gen_mo.detach().cpu()[ii]


            if self.global_rank == 0:
                return {
                    'val_recon_motions': target_feats_recon,
                    'val_recon_motions_src': source_feats_recon,
                    'val_motions': gen_mo
                }
            else:
                return {'val_motions': gen_mo}

        return total_loss

    def _allsplit_step_diffusion(self, split: str, batch, batch_idx):

        from src.data.tools.tensors import lengths_to_mask

        input_batch = self.norm_and_cat(batch, self.input_feats)

        for k, v in input_batch.items():
            batch[f'{k}_motion'] = v

            if v.shape[0] > 1 and self.pad_inputs:

                batch[f'{k}_motion'] = torch.nn.functional.pad(
                    v, (0, 0, 0, 0, 0, 300 - v.size(0)), value=0
                )


        if self.motion_condition is not None:
            if self.pad_inputs:

                mask_source, mask_target = self.prepare_mot_masks(
                    batch['length_source'],
                    batch['length_target'],
                    max_len=300
                )
            else:

                mask_source, mask_target = self.prepare_mot_masks(
                    batch['length_source'],
                    batch['length_target'],
                    max_len=None
                )
        else:

            mask_target = lengths_to_mask(batch['length_target'], device=self.device)
            if v.shape[0] > 1 and self.pad_inputs:

                mask_target = F.pad(mask_target, (0, 300 - mask_target.size(1)), value=0)

            batch['length_source'] = None
            batch['source_motion'] = None
            mask_source = None

        actual_target_lens = batch['length_target']



        gt_lens_tgt = batch['length_target']
        gt_lens_src = batch['length_source']
        batch['text'] = [el.lower() for el in batch['text']]
        gt_texts = batch['text']
        gt_keyids = batch['id']
        self.batch_size = len(gt_texts)


        dif_dict = self.train_diffusion_forward(
            batch,
            mask_source,
            mask_target
        )


        total_loss, loss_dict = self.compute_losses(dif_dict, batch['dataset_name'])

        # if self.trainer.current_epoch % 100 == 0 and self.trainer.current_epoch != 0:
        #     if self.global_rank == 0 and split=='train' and batch_idx == 0:
        #         if self.renderer is not None:
        #             self.visualize_diffusion(dif_dict, actual_target_lens,
        #                                     gt_keyids, gt_texts,
        #                                     self.trainer.current_epoch)

        # self.losses[split](rs_set)
        # if loss is None:
        #     raise ValueError("Loss is None, this happend with torchmetrics > 0.7")


        loss_dict_to_log = {
            f'total_losses/{split}/{k}' if k not in self.input_feats
            else f'feature_losses/{split}/{k}': v
            for k, v in loss_dict.items()
        }

        # loss_dict_to_log = {f'losses/{split}/{k}': v for k, v in
        #                     loss_dict.items()}
        self.log_dict(loss_dict_to_log, on_epoch=True, batch_size=self.batch_size,sync_dist=True,rank_zero_only=True)

        import random
        if split == 'val':
            from tqdm import tqdm

            # gd_motion = [5.0] #, 3.0, 7.0]
            # guidances_mix = [(x, y) for x in gd_text for y in gd_motion]
            guidances_mix = [(2.0, 2.0),(2.0,4.5)]




            infer_steps = getattr(self.diff_params, 'num_inference_timesteps', self.diffusion_process.num_timesteps)



            if infer_steps != self.diffusion_process.num_timesteps:

                from src.diffusion import create_diffusion
                inference_diffusion_process = create_diffusion(
                    timestep_respacing=None,
                    learn_sigma=False,
                    sigma_small=True,
                    diffusion_steps=infer_steps,
                    noise_schedule=self.diff_params.noise_schedule,
                    predict_xstart=False if self.diff_params.predict_type == 'noise' else True
                )
            else:

                inference_diffusion_process = self.diffusion_process

            if batch_idx == 0:

                self.validation_step_outputs = {
                    f'{s_t}txt_{s_m}mot': {} for s_t, s_m in guidances_mix
                }
            # prepare the motions
            # compute the metrics

            for guid_text, guid_motion in guidances_mix:


                diffout = self.generate_motion(
                    gt_texts, batch['source_motion'],
                    mask_source, mask_target,
                    inference_diffusion_process,
                    gd_motion=guid_motion,
                    gd_text=guid_text,
                    num_diff_steps=infer_steps,
                    show_progress=False
                )
                gen_mo = self.diffout2motion(diffout)

                for ii, kval in enumerate(gt_keyids):

                    self.validation_step_outputs[f'{guid_text}txt_{guid_motion}mot'][kval] = gen_mo.detach().cpu()[ii]

            return {'val_motions': gen_mo}

        return total_loss

    def _allsplit_step_vae(self, split: str, batch, batch_idx):


        input_batch = self.norm_and_cat(batch, self.input_feats)


        for k, v in input_batch.items():
            batch[f'{k}_motion'] = v


        result = self.train_vae_step(batch, batch_idx, split=split)


        if split == 'val' and self.global_rank == 0:

            target_feats_recon = result['target_feats_recon']
            source_feats_recon = result['source_feats_recon']
            total_loss = result['loss']


            target_lengths = batch['length_target']
            gt_keyids = batch.get('id', [f'val_{batch_idx}_{i}' for i in range(len(target_lengths))])


            if batch_idx == 0:
                self.validation_step_outputs = {}



            for ii, kval in enumerate(gt_keyids):
                self.validation_step_outputs[f'target_recon_{kval}'] = target_feats_recon.detach().cpu()[ii]


            if source_feats_recon is not None:
                for ii, kval in enumerate(gt_keyids):
                    self.validation_step_outputs[f'source_recon_{kval}'] = source_feats_recon.detach().cpu()[ii]


            return {'val_recon_motions': target_feats_recon}


        return result

    def train_vae_step(self, batch, batch_idx, split='train'):


        target_feats = batch['target_motion']  # [T, B, nfeats]
        target_lengths = batch['length_target']


        source_feats = batch.get('source_motion', None)
        source_lengths = batch.get('length_source', None)


        recons_joints = torch.tensor(0.0, device=target_feats.device)
        lambda_joint = self.loss_params.get('lmd_joint', 1.0)




        if source_feats is None:


            target_z, target_dist = self.encode_with_vae(target_feats, target_lengths)


            target_feats_rst = self.decode_with_vae(target_z, target_lengths)  # [B, T, nfeats]



            target_feats_bt = self._ensure_batch_first(target_feats, target_lengths)  # [B, T, nfeats]
            recons_feature_target = self.loss_func_feats_vae(target_feats_rst, target_feats_bt)
            recons_feature_source = torch.tensor(0.0, device=target_feats.device)
            recons_feature = recons_feature_target


            prior_dist = torch.distributions.Normal(
                torch.zeros_like(target_dist.loc),
                torch.ones_like(target_dist.scale)
            )
            kl_target = torch.distributions.kl_divergence(target_dist, prior_dist).mean()
            kl_source = torch.tensor(0.0, device=target_feats.device)
            kl_motion = kl_target






            if split == 'val':
                with torch.no_grad():


                    joints_rst = self.feats2joints(target_feats_rst, target_lengths)  # [B, T, 22, 3]



                    joints_ref = self.feats2joints(target_feats_bt, target_lengths)  # [B, T, 22, 3]
            else:

                joints_rst = self.feats2joints(target_feats_rst, target_lengths)  # [B, T, 22, 3]
                joints_ref = self.feats2joints(target_feats_bt, target_lengths)  # [B, T, 22, 3]





            rs_set = {
                'm_ref': target_feats_bt,
                'm_rst': target_feats_rst,
                'joints_ref': joints_ref,
                'joints_rst': joints_rst,
                'dist_m': target_dist,
                'dist_ref': prior_dist,
                'lengths': target_lengths,
            }




            lambda_kl = self.loss_params.get('lmd_kl', 1e-4)




            if lambda_joint > 0.0:
                from src.data.tools.tensors import lengths_to_mask
                from torch.nn import SmoothL1Loss


                mask = lengths_to_mask(target_lengths, device=joints_rst.device)  # [B, T]


                B, T, njoints, _ = joints_rst.shape
                joints_rst_flat = joints_rst.reshape(B * T, njoints, 3)  # [B*T, 22, 3]
                joints_ref_flat = joints_ref.reshape(B * T, njoints, 3)  # [B*T, 22, 3]


                mask_flat = mask.reshape(B * T)  # [B*T]
                joints_rst_masked = joints_rst_flat[mask_flat]  # [N_valid, 22, 3]
                joints_ref_masked = joints_ref_flat[mask_flat]  # [N_valid, 22, 3]



                joint_loss_func = SmoothL1Loss(reduction='mean')
                recons_joints = joint_loss_func(joints_rst_masked, joints_ref_masked)
            else:
                recons_joints = torch.tensor(0.0, device=recons_feature.device)


            total_loss = recons_feature + lambda_kl * kl_motion + lambda_joint * recons_joints


            if split == 'val' and hasattr(self, 'mr_metrics') and self.mr_metrics is not None:
                self.mr_metrics.update(
                    joints_rst,
                    joints_ref,
                    target_lengths
                )



        else:


            target_z, target_dist = self.encode_with_vae(target_feats, target_lengths)
            source_z, source_dist = self.encode_with_vae(source_feats, source_lengths)


            target_feats_rst = self.decode_with_vae(target_z, target_lengths)  # [B, T, nfeats]
            source_feats_rst = self.decode_with_vae(source_z, source_lengths)  # [B, T, nfeats]


            target_feats_bt = self._ensure_batch_first(target_feats, target_lengths)  # [B, T, nfeats]
            source_feats_bt = self._ensure_batch_first(source_feats, source_lengths)  # [B, T, nfeats]




            if split == 'val':
                with torch.no_grad():

                    target_joints_rst = self.feats2joints(target_feats_rst, target_lengths)  # [B, T, 22, 3]
                    source_joints_rst = self.feats2joints(source_feats_rst, source_lengths)  # [B, T, 22, 3]


                    target_joints_ref = self.feats2joints(target_feats_bt, target_lengths)  # [B, T, 22, 3]
                    source_joints_ref = self.feats2joints(source_feats_bt, source_lengths)  # [B, T, 22, 3]
            else:

                target_joints_rst = self.feats2joints(target_feats_rst, target_lengths)  # [B, T, 22, 3]
                source_joints_rst = self.feats2joints(source_feats_rst, source_lengths)  # [B, T, 22, 3]
                target_joints_ref = self.feats2joints(target_feats_bt, target_lengths)  # [B, T, 22, 3]
                source_joints_ref = self.feats2joints(source_feats_bt, source_lengths)  # [B, T, 22, 3]


            prior_dist = torch.distributions.Normal(
                torch.zeros_like(target_dist.loc),
                torch.ones_like(target_dist.scale)
            )



            recons_feature_target = self.loss_func_feats_vae(target_feats_rst, target_feats_bt)
            recons_feature_source = self.loss_func_feats_vae(source_feats_rst, source_feats_bt)
            recons_feature = (recons_feature_target + recons_feature_source) / 2


            kl_target = torch.distributions.kl_divergence(target_dist, prior_dist).mean()
            kl_source = torch.distributions.kl_divergence(source_dist, prior_dist).mean()
            kl_motion = (kl_target + kl_source) / 2


            lambda_kl = self.loss_params.get('lmd_kl', 1e-5)



            if lambda_joint > 0.0:
                from src.data.tools.tensors import lengths_to_mask
                from torch.nn import SmoothL1Loss



                T_target = target_joints_rst.shape[1]
                T_source = source_joints_rst.shape[1]
                T_max = max(T_target, T_source)


                if T_target < T_max:
                    pad_size = T_max - T_target
                    target_joints_rst_padded = torch.nn.functional.pad(
                        target_joints_rst, (0, 0, 0, 0, 0, pad_size), mode='constant', value=0
                    )
                    target_joints_ref_padded = torch.nn.functional.pad(
                        target_joints_ref, (0, 0, 0, 0, 0, pad_size), mode='constant', value=0
                    )
                else:
                    target_joints_rst_padded = target_joints_rst
                    target_joints_ref_padded = target_joints_ref

                if T_source < T_max:
                    pad_size = T_max - T_source
                    source_joints_rst_padded = torch.nn.functional.pad(
                        source_joints_rst, (0, 0, 0, 0, 0, pad_size), mode='constant', value=0
                    )
                    source_joints_ref_padded = torch.nn.functional.pad(
                        source_joints_ref, (0, 0, 0, 0, 0, pad_size), mode='constant', value=0
                    )
                else:
                    source_joints_rst_padded = source_joints_rst
                    source_joints_ref_padded = source_joints_ref


                joints_rst_merged = torch.cat([target_joints_rst_padded, source_joints_rst_padded], dim=0)  # [2*B, T_max, 22, 3]
                joints_ref_merged = torch.cat([target_joints_ref_padded, source_joints_ref_padded], dim=0)  # [2*B, T_max, 22, 3]
                all_lengths = target_lengths + source_lengths


                mask = lengths_to_mask(all_lengths, device=joints_rst_merged.device)  # [2*B, T_max]


                B_merged, T_merged, njoints, _ = joints_rst_merged.shape
                joints_rst_flat = joints_rst_merged.reshape(B_merged * T_merged, njoints, 3)  # [2*B*T_max, 22, 3]
                joints_ref_flat = joints_ref_merged.reshape(B_merged * T_merged, njoints, 3)  # [2*B*T_max, 22, 3]


                mask_flat = mask.reshape(B_merged * T_merged)  # [2*B*T_max]
                joints_rst_masked = joints_rst_flat[mask_flat]  # [N_valid, 22, 3]
                joints_ref_masked = joints_ref_flat[mask_flat]  # [N_valid, 22, 3]



                joint_loss_func = SmoothL1Loss(reduction='mean')
                recons_joints = joint_loss_func(joints_rst_masked, joints_ref_masked)
            else:
                recons_joints = torch.tensor(0.0, device=recons_feature.device)


            total_loss = recons_feature + lambda_kl * kl_motion + lambda_joint * recons_joints



            if split == 'val' and hasattr(self, 'mr_metrics') and self.mr_metrics is not None:


                T_target = target_joints_rst.shape[1]
                T_source = source_joints_rst.shape[1]
                T_max = max(T_target, T_source)


                if T_target < T_max:
                    pad_size = T_max - T_target
                    target_joints_rst_padded = torch.nn.functional.pad(
                        target_joints_rst, (0, 0, 0, 0, 0, pad_size), mode='constant', value=0
                    )
                    target_joints_ref_padded = torch.nn.functional.pad(
                        target_joints_ref, (0, 0, 0, 0, 0, pad_size), mode='constant', value=0
                    )
                else:
                    target_joints_rst_padded = target_joints_rst
                    target_joints_ref_padded = target_joints_ref

                if T_source < T_max:
                    pad_size = T_max - T_source
                    source_joints_rst_padded = torch.nn.functional.pad(
                        source_joints_rst, (0, 0, 0, 0, 0, pad_size), mode='constant', value=0
                    )
                    source_joints_ref_padded = torch.nn.functional.pad(
                        source_joints_ref, (0, 0, 0, 0, 0, pad_size), mode='constant', value=0
                    )
                else:
                    source_joints_rst_padded = source_joints_rst
                    source_joints_ref_padded = source_joints_ref


                joints_rst_merged = torch.cat([target_joints_rst_padded, source_joints_rst_padded], dim=0)  # [2*B, T_max, 22, 3]
                joints_ref_merged = torch.cat([target_joints_ref_padded, source_joints_ref_padded], dim=0)  # [2*B, T_max, 22, 3]
                all_lengths = target_lengths + source_lengths

                self.mr_metrics.update(
                    joints_rst_merged,
                    joints_ref_merged,
                    all_lengths
                )








        loss_dict = {
            f'recons/feature/{split}': recons_feature,
            f'kl/motion/{split}': kl_motion,
            f'total/{split}': total_loss,
        }


        loss_dict[f'recons/joints/{split}'] = recons_joints


        # if source_feats is not None:
        #     loss_dict[f'recons/feature_target/{split}'] = recons_feature_target
        #     loss_dict[f'recons/feature_source/{split}'] = recons_feature_source


        self.log_dict(loss_dict, on_step=True, on_epoch=True, batch_size=len(target_lengths), sync_dist=True,rank_zero_only=True)


        if split == 'val':
            if source_feats is None:

                return {
                    'loss': total_loss,
                    'target_feats_recon': target_feats_rst,
                    'source_feats_recon': None
                }
            else:

                return {
                    'loss': total_loss,
                    'target_feats_recon': target_feats_rst,
                    'source_feats_recon': source_feats_rst
                }
        else:

            return total_loss

    def encode_with_vae(self, features: Tensor, lengths: Optional[List[int]] = None):

        if self.vae is None:
            raise ValueError("VAE is not initialized")
        return self.vae.encode(features, lengths)

    def decode_with_vae(self, z: Tensor, lengths: List[int]):

        if self.vae is None:
            raise ValueError("VAE is not initialized")

        feats_seq_first = self.vae.decode(z, lengths)  # [T, B, nfeats]

        feats_batch_first = feats_seq_first.permute(1, 0, 2)  # [B, T, nfeats]
        return feats_batch_first
