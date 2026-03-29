

from typing import Union
import logging
import torch
import torch.nn as nn
from src.model.utils.timestep_embed import TimestepEmbedding, Timesteps, TimestepEmbedderMDM
from src.model.utils.positional_encoding import PositionalEncoding
from src.model.utils.transf_utils import (
    SkipTransformerEncoder,
    TransformerEncoder,
    TransformerEncoderLayer,
)
from src.model.utils.all_positional_encodings import build_position_encoding
from src.data.tools.tensors import lengths_to_mask

logger = logging.getLogger(__name__)

class TMED_denoiser(nn.Module):


    def __init__(self,
                 nfeats: int = 263,
                 condition: str = "text",
                 motion_condition: str = None,
                 latent_dim: Union[int, list] = 512,
                 ff_size: int = 1024,
                 num_layers: int = 9,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 activation: str = "gelu",
                 text_encoded_dim: int = 768,
                 pred_delta_motion: bool = False,
                 use_sep: bool = True,
                 use_latent_input: bool = False,
                 use_skip_transformer: bool = False,
                 **kwargs) -> None:

        super().__init__()

        if isinstance(latent_dim, list):
            self.latent_dim = latent_dim[-1]
        else:
            self.latent_dim = latent_dim

        self.use_latent_input = use_latent_input
        self.pred_delta_motion = pred_delta_motion
        self.text_encoded_dim = text_encoded_dim
        self.condition = condition
        self.motion_condition = motion_condition
        self.use_skip_transformer = use_skip_transformer

        self.feat_comb_coeff = nn.Parameter(torch.tensor([1.0]))




        if use_latent_input:

            self.pose_proj_in_source = nn.Identity()
            self.pose_proj_in_target = nn.Identity()
            self.pose_proj_out = nn.Identity()
            self.first_pose_proj = nn.Identity()

            self.pose_proj_in_source_feat = nn.Linear(nfeats, self.latent_dim) if self.motion_condition == "source" else None
        else:

            self.pose_proj_in_source = nn.Linear(nfeats, self.latent_dim)
            self.pose_proj_in_target = nn.Linear(nfeats, self.latent_dim)
            self.pose_proj_out = nn.Linear(self.latent_dim, nfeats)
            self.first_pose_proj = nn.Linear(self.latent_dim, nfeats)
            self.pose_proj_in_source_feat = None






        if self.condition in ["text", "text_uncond"]:


            self.embed_timestep = TimestepEmbedderMDM(self.latent_dim)

            # FIXME me TODO this
            # self.time_embedding = TimestepEmbedderMDM(self.latent_dim)



            if text_encoded_dim != self.latent_dim:
                # todo 10.24 debug why relu
                self.emb_proj = nn.Linear(text_encoded_dim, self.latent_dim)
        else:
            raise TypeError(f"condition type {self.condition} not supported")


        self.use_sep = use_sep


        self.query_pos = PositionalEncoding(self.latent_dim, dropout)
        self.mem_pos = PositionalEncoding(self.latent_dim, dropout)




        if self.motion_condition == "source":
            if self.use_sep:
                self.sep_token = nn.Parameter(torch.randn(1, self.latent_dim))




        encoder_layer = TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation,
            normalize_before=False,
        )
        if self.use_skip_transformer:

            assert num_layers % 2 == 1, "SkipTransformerEncoder requires an odd number of layers (1, 3, 5, ...)"
            logger.info(
                f"✅ TMED Denoiser: skip Transformer launched "
                f"[layers={num_layers}, dim={self.latent_dim}, heads={num_heads}]"
            )
            self.encoder = SkipTransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=num_layers,
                norm=None,
            )
        else:

            self.encoder = TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=num_layers,
                norm=None,
            )

    def forward(self,
                noised_motion,
                timestep,
                in_motion_mask,
                text_embeds,
                condition_mask,
                motion_embeds=None,
                lengths=None,
                **kwargs):


        bs = noised_motion.shape[0]

        if self.use_latent_input:



            noised_motion = noised_motion.permute(1, 0, 2)  # [B, latent_size, latent_dim] → [latent_size, B, latent_dim]
            proj_noised_motion = self.pose_proj_in_target(noised_motion)

            if motion_embeds is not None:


                if motion_embeds.shape[1] != proj_noised_motion.shape[1]:
                    motion_embeds = motion_embeds.permute(1, 0, 2)


                if motion_embeds.shape[-1] != self.latent_dim and self.pose_proj_in_source_feat is not None:
                    motion_embeds_proj = self.pose_proj_in_source_feat(motion_embeds)
                    zeroes_mask = (motion_embeds == 0).all(dim=-1)
                    motion_embeds_proj[zeroes_mask] = 0
                else:

                    motion_embeds_proj = self.pose_proj_in_source(motion_embeds)
            else:
                motion_embeds_proj = None
        else:



            noised_motion = noised_motion.permute(1, 0, 2)  # [B, seq_len, nfeats] → [seq_len, B, nfeats]
            proj_noised_motion = self.pose_proj_in_target(noised_motion)

            if motion_embeds is not None:


                if motion_embeds.shape[1] != proj_noised_motion.shape[1]:

                    motion_embeds = motion_embeds.permute(1, 0, 2)  # [B, seq_len, nfeats] → [seq_len, B, nfeats]


                if motion_embeds.shape[-1] != self.latent_dim:
                    motion_embeds_proj = self.pose_proj_in_source(motion_embeds)  # [seq_len, B, nfeats] → [seq_len, B, latent_dim]

                    zeroes_mask = (motion_embeds == 0).all(dim=-1)
                    motion_embeds_proj[zeroes_mask] = 0
                else:
                    motion_embeds_proj = motion_embeds
            else:
                motion_embeds_proj = None

        motion_in_mask = in_motion_mask









        timesteps = timestep.expand(proj_noised_motion.shape[1]).clone()
        time_emb = self.embed_timestep(timesteps).to(dtype=proj_noised_motion.dtype)


        if self.condition in ["text", "text_uncond"]:

            text_embeds = text_embeds.permute(1, 0, 2)


            if self.text_encoded_dim != self.latent_dim:
                # [1 or 2, bs, latent_dim] <= [1 or 2, bs, text_encoded_dim]
                text_emb_latent = self.emb_proj(text_embeds)
            else:
                text_emb_latent = text_embeds


            emb_latent = torch.cat((time_emb, text_emb_latent), 0)

        else:
            raise TypeError(f"condition type {self.condition} not supported")



        if motion_embeds_proj is None:

            xseq = torch.cat((emb_latent, proj_noised_motion), axis=0)
        else:

            if self.use_sep:

                sep_token_batch = torch.tile(self.sep_token, (bs,)).reshape(bs, -1)
                xseq = torch.cat((emb_latent, motion_embeds_proj,
                                sep_token_batch[None],
                                proj_noised_motion), axis=0)
            else:

                xseq = torch.cat((emb_latent, motion_embeds_proj,
                                  proj_noised_motion), axis=0)



        xseq = self.query_pos(xseq)



        if motion_embeds is None:

            time_token_mask = torch.ones((bs, time_emb.shape[0]),
                                        dtype=bool, device=xseq.device)
            aug_mask = torch.cat((time_token_mask,
                                  condition_mask[:, :text_emb_latent.shape[0]],
                                  motion_in_mask), 1)
        else:

            time_token_mask = torch.ones((bs, time_emb.shape[0]),
                                        dtype=bool,
                                        device=xseq.device)
            if self.use_sep:
                sep_token_mask = torch.ones((bs, self.sep_token.shape[0]),
                                        dtype=bool,
                                        device=xseq.device)


            if self.use_sep:
                aug_mask = torch.cat((time_token_mask,
                                condition_mask[:, :text_emb_latent.shape[0]],
                                condition_mask[:, text_emb_latent.shape[0]:],
                                sep_token_mask,
                                motion_in_mask,
                                ), 1)
            else:
                aug_mask = torch.cat((time_token_mask,
                                condition_mask[:, :text_emb_latent.shape[0]],
                                condition_mask[:, text_emb_latent.shape[0]:],
                                motion_in_mask,
                                ), 1)




        tokens = self.encoder(xseq, src_key_padding_mask=~aug_mask)



        if motion_embeds is not None:

            denoised_motion_proj = tokens[emb_latent.shape[0]:]
            if self.use_sep:

                useful_tokens = motion_embeds_proj.shape[0]+1
            else:

                useful_tokens = motion_embeds_proj.shape[0]
            denoised_motion_proj = denoised_motion_proj[useful_tokens:]
        else:

            denoised_motion_proj = tokens[emb_latent.shape[0]:]



        denoised_motion = self.pose_proj_out(denoised_motion_proj)




        if self.pred_delta_motion and motion_embeds is not None:
            import torch.nn.functional as F
            tgt_size = len(denoised_motion)

            if len(denoised_motion) > len(motion_embeds):
                pad_for_src = tgt_size - len(motion_embeds)
                motion_embeds = F.pad(motion_embeds,
                                      (0, 0, 0, 0, 0, pad_for_src))

            denoised_motion = denoised_motion + motion_embeds[:tgt_size]



        denoised_motion[~motion_in_mask.T] = 0


        # [batch_size, seq_len, feat_dim] <= [seq_len, batch_size, feat_dim]
        denoised_motion = denoised_motion.permute(1, 0, 2)
        return denoised_motion

    def forward_with_guidance(self,
                              noised_motion,
                              timestep,
                              in_motion_mask,
                              text_embeds,
                              condition_mask,
                              guidance_motion,
                              guidance_text_n_motion,
                              motion_embeds=None,
                              lengths=None,
                              inpaint_dict=None,
                              max_steps=None,
                              prob_way='3way',
                              **kwargs):




        if max_steps is not None:
            curr_ts = timestep[0].item()

            g_m = max(1, guidance_motion*2*curr_ts/max_steps)
            guidance_motion = g_m
            g_t_tm = max(1, guidance_text_n_motion*2*curr_ts/max_steps)
            guidance_text_n_motion = g_t_tm


        if motion_embeds is None:

            half = noised_motion[: len(noised_motion) // 2]
            combined = torch.cat([half, half], dim=0)


            model_out = self.forward(combined, timestep,
                                    in_motion_mask=in_motion_mask,
                                    text_embeds=text_embeds,
                                    condition_mask=condition_mask,
                                    motion_embeds=motion_embeds,
                                    lengths=lengths)


            uncond_eps, cond_eps_text = torch.split(model_out, len(model_out) // 2,
                                                     dim=0)



            if inpaint_dict is not None:
                import torch.nn.functional as F
                source_mot = inpaint_dict['start_motion'].permute(1, 0, 2)


                if source_mot.shape[1] >= uncond_eps.shape[1]:
                    source_mot = source_mot[:, :uncond_eps.shape[1]]
                else:
                    pad = uncond_eps.shape[1] - source_mot.shape[1]
                    source_mot = F.pad(source_mot, (0, 0, 0, pad), 'constant', 0)

                mot_len = source_mot.shape[1]

                mask_src_parts = inpaint_dict['mask'].unsqueeze(1).repeat(1,
                                                                      mot_len,
                                                                      1)

                uncond_eps = uncond_eps*(mask_src_parts) + source_mot*(~mask_src_parts)
                cond_eps_text = cond_eps_text*(mask_src_parts) + source_mot*(~mask_src_parts)



            half_eps = uncond_eps + guidance_text_n_motion * (cond_eps_text - uncond_eps)


            eps = torch.cat([half_eps, half_eps], dim=0)


        else:

            third = noised_motion[: len(noised_motion) // 3]
            combined = torch.cat([third, third, third], dim=0)


            model_out = self.forward(combined, timestep,
                                     in_motion_mask=in_motion_mask,
                                     text_embeds=text_embeds,
                                     condition_mask=condition_mask,
                                     motion_embeds=motion_embeds,
                                     lengths=lengths)


            uncond_eps, cond_eps_motion, cond_eps_text_n_motion = torch.split(model_out,
                                                                            len(model_out) // 3,
                                                                            dim=0)


            if inpaint_dict is not None:
                import torch.nn.functional as F
                source_mot = inpaint_dict['start_motion'].permute(1, 0, 2)


                if source_mot.shape[1] >= uncond_eps.shape[1]:
                    source_mot = source_mot[:, :uncond_eps.shape[1]]
                else:
                    pad = uncond_eps.shape[1] - source_mot.shape[1]
                    source_mot = F.pad(source_mot, (0, 0, 0, pad), 'constant', 0)

                mot_len = source_mot.shape[1]

                mask_src_parts = inpaint_dict['mask'].unsqueeze(1).repeat(1,
                                                                      mot_len,
                                                                      1)


                uncond_eps = uncond_eps*(~mask_src_parts) + source_mot*mask_src_parts
                cond_eps_text = cond_eps_text*(~mask_src_parts) + source_mot*mask_src_parts
                cond_eps_text_n_motion = cond_eps_text_n_motion*(~mask_src_parts) + source_mot*mask_src_parts


            if prob_way=='3way':

                # ε = ε_uncond + s_{M_S} * (ε_motion - ε_uncond) + s_L * (ε_text+mot - ε_motion)
                third_eps = uncond_eps + guidance_motion * (cond_eps_motion - uncond_eps) + \
                            guidance_text_n_motion * (cond_eps_text_n_motion - cond_eps_motion)
            if prob_way=='2way':

                # ε = ε_uncond + s * (ε_text+mot - ε_uncond)
                third_eps = uncond_eps + guidance_text_n_motion * (cond_eps_text_n_motion - uncond_eps)


            eps = torch.cat([third_eps, third_eps, third_eps], dim=0)

        return eps
