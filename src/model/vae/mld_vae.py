"""
Motion latent VAE (MLD-style).

Encoder–decoder over motion sequences with optional skip connections, actor vs MLD
positional encodings, and either all-encoder or encoder–decoder layout.
"""

from typing import List, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions.distribution import Distribution

from src.model.utils.transf_utils import (
    SkipTransformerEncoder,
    SkipTransformerDecoder,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)
from src.model.utils.all_positional_encodings import build_position_encoding
from src.model.utils.positional_encoding import PositionalEncoding
from src.data.tools.tensors import lengths_to_mask


class MldVae(nn.Module):
    """
    VAE mapping motion features to a low-dimensional latent sequence and back.

    Args:
        nfeats: Input feature size per frame.
        latent_dim: ``[latent_size, latent_dim]`` (e.g. ``[1, 256]`` or ``[8, 256]``).
        ff_size: Transformer FFN hidden size.
        num_layers: Number of transformer layers.
        num_heads: Attention heads.
        dropout: Dropout probability.
        arch: ``"all_encoder"`` or ``"encoder_decoder"``.
        normalize_before: Pre-norm vs post-norm in transformer blocks.
        activation: Activation name (e.g. ``"gelu"``).
        position_embedding: Passed to MLD-style position encoding when ``pe_type=="mld"``.
        mlp_dist: If True, predict Gaussian params via an MLP head; else split tokens for mu/logvar.
        pe_type: ``"mld"`` or ``"actor"`` positional encoding.
        skip_connect: Use skip-transformer stacks when True.
    """

    def __init__(
        self,
        nfeats: int,
        latent_dim: list = [1, 256],
        ff_size: int = 1024,
        num_layers: int = 9,
        num_heads: int = 4,
        dropout: float = 0.1,
        arch: str = "encoder_decoder",
        normalize_before: bool = False,
        activation: str = "gelu",
        position_embedding: str = "learned",
        mlp_dist: bool = False,
        pe_type: str = "mld",
        skip_connect: bool = True,
        **kwargs,
    ) -> None:

        super().__init__()

        self.latent_size = latent_dim[0]
        self.latent_dim = latent_dim[-1]
        input_feats = nfeats
        output_feats = nfeats
        self.arch = arch
        self.mlp_dist = mlp_dist
        self.pe_type = pe_type
        self.skip_connect = skip_connect

        if self.pe_type == "actor":
            self.query_pos_encoder = PositionalEncoding(self.latent_dim, dropout)
            self.query_pos_decoder = PositionalEncoding(self.latent_dim, dropout)
        elif self.pe_type == "mld":
            self.query_pos_encoder = build_position_encoding(
                self.latent_dim, position_embedding=position_embedding
            )
            self.query_pos_decoder = build_position_encoding(
                self.latent_dim, position_embedding=position_embedding
            )
        else:
            raise ValueError("Not Support PE type")

        encoder_layer = TransformerEncoderLayer(
            self.latent_dim,
            num_heads,
            ff_size,
            dropout,
            activation,
            normalize_before,
        )
        encoder_norm = nn.LayerNorm(self.latent_dim)

        if self.skip_connect:
            self.encoder = SkipTransformerEncoder(encoder_layer, num_layers, encoder_norm)
        else:
            self.encoder = TransformerEncoder(encoder_layer, num_layers, encoder_norm)

        if self.arch == "all_encoder":
            decoder_norm = nn.LayerNorm(self.latent_dim)
            if self.skip_connect:
                self.decoder = SkipTransformerEncoder(encoder_layer, num_layers, decoder_norm)
            else:
                self.decoder = TransformerEncoder(encoder_layer, num_layers, decoder_norm)
        elif self.arch == "encoder_decoder":
            decoder_layer = TransformerDecoderLayer(
                self.latent_dim,
                num_heads,
                ff_size,
                dropout,
                activation,
                normalize_before,
            )
            decoder_norm = nn.LayerNorm(self.latent_dim)
            if self.skip_connect:
                self.decoder = SkipTransformerDecoder(decoder_layer, num_layers, decoder_norm)
            else:
                self.decoder = TransformerDecoder(decoder_layer, num_layers, decoder_norm)
        else:
            raise ValueError("Not support architecture!")

        if self.mlp_dist:
            self.global_motion_token = nn.Parameter(torch.randn(self.latent_size, self.latent_dim))
            self.dist_layer = nn.Linear(self.latent_dim, 2 * self.latent_dim)
        else:
            self.global_motion_token = nn.Parameter(torch.randn(self.latent_size * 2, self.latent_dim))

        self.skel_embedding = nn.Linear(input_feats, self.latent_dim)
        self.final_layer = nn.Linear(self.latent_dim, output_feats)

    def forward(self, features: Tensor, lengths: Optional[List[int]] = None):
        z, dist = self.encode(features, lengths)
        feats_rst = self.decode(z, lengths)
        return feats_rst, z, dist

    def encode(
        self,
        features: Tensor,
        lengths: Optional[List[int]] = None,
    ) -> Union[Tensor, Distribution]:
        # MotionFix uses [T, B, C]; internal path expects [B, T, C].
        if features.dim() == 3:
            features = features.permute(1, 0, 2)

        if lengths is None:
            lengths = [features.shape[1] for _ in range(features.shape[0])]

        device = features.device
        bs, nframes, nfeats = features.shape

        mask = lengths_to_mask(lengths, device)
        x = features
        x = self.skel_embedding(x)
        x = x.permute(1, 0, 2)

        dist = torch.tile(self.global_motion_token[:, None, :], (1, bs, 1))
        dist_masks = torch.ones((bs, dist.shape[0]), dtype=bool, device=x.device)
        aug_mask = torch.cat((dist_masks, mask), 1)
        xseq = torch.cat((dist, x), 0)

        if self.pe_type == "actor":
            xseq = self.query_pos_encoder(xseq)
            dist = self.encoder(xseq, src_key_padding_mask=~aug_mask)[: dist.shape[0]]
        elif self.pe_type == "mld":
            xseq = self.query_pos_encoder(xseq)
            dist = self.encoder(xseq, src_key_padding_mask=~aug_mask)[: dist.shape[0]]

        if self.mlp_dist:
            tokens_dist = self.dist_layer(dist)
            mu = tokens_dist[:, :, : self.latent_dim]
            logvar = tokens_dist[:, :, self.latent_dim :]
        else:
            mu = dist[0 : self.latent_size, ...]
            logvar = dist[self.latent_size :, ...]

        std = logvar.exp().pow(0.5)
        dist = torch.distributions.Normal(mu, std)
        latent = dist.rsample()
        return latent, dist

    def decode(self, z: Tensor, lengths: List[int]):
        mask = lengths_to_mask(lengths, z.device)
        bs, nframes = mask.shape
        queries = torch.zeros(nframes, bs, self.latent_dim, device=z.device)

        if self.arch == "all_encoder":
            xseq = torch.cat((z, queries), axis=0)
            z_mask = torch.ones((bs, self.latent_size), dtype=bool, device=z.device)
            augmask = torch.cat((z_mask, mask), axis=1)
            if self.pe_type == "actor":
                xseq = self.query_pos_decoder(xseq)
                output = self.decoder(xseq, src_key_padding_mask=~augmask)[z.shape[0] :]
            elif self.pe_type == "mld":
                xseq = self.query_pos_decoder(xseq)
                output = self.decoder(xseq, src_key_padding_mask=~augmask)[z.shape[0] :]

        elif self.arch == "encoder_decoder":
            if self.pe_type == "actor":
                queries = self.query_pos_decoder(queries)
                output = self.decoder(
                    tgt=queries,
                    memory=z,
                    tgt_key_padding_mask=~mask,
                ).squeeze(0)
            elif self.pe_type == "mld":
                queries = self.query_pos_decoder(queries)
                output = self.decoder(
                    tgt=queries,
                    memory=z,
                    tgt_key_padding_mask=~mask,
                ).squeeze(0)

        output = self.final_layer(output)
        output[~mask.T] = 0
        feats = output
        return feats
