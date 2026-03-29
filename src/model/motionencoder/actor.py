

import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl

from typing import List, Optional, Union
from torch import nn, Tensor
from torch.distributions.distribution import Distribution

from src.model.utils import PositionalEncoding
from src.data.tools import lengths_to_mask


class ActorAgnosticEncoder(nn.Module):

    def __init__(self, nfeats: int,
                 latent_dim: int = 256, ff_size: int = 1024,
                 num_layers: int = 4, num_heads: int = 4,
                 dropout: float = 0.1,
                 activation: str = "gelu", **kwargs) -> None:
        super().__init__()

        input_feats = nfeats

        self.skel_embedding = nn.Linear(input_feats, latent_dim)


        self.emb_token = nn.Parameter(torch.randn(latent_dim))


        self.sequence_pos_encoding = PositionalEncoding(latent_dim, dropout)

        # Transformer Encoder
        seq_trans_encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim,
                                                             nhead=num_heads,
                                                             dim_feedforward=ff_size,
                                                             dropout=dropout,
                                                             activation=activation)

        self.seqTransEncoder = nn.TransformerEncoder(seq_trans_encoder_layer,
                                                     num_layers=num_layers)

    def forward(self, features: Tensor, mask: Tensor) -> Union[Tensor, Distribution]:

        in_mask = mask
        device = features.device

        nframes, bs, nfeats = features.shape

        x = features

        x = self.skel_embedding(x)


        emb_token = torch.tile(self.emb_token, (bs,)).reshape(bs, -1)


        xseq = torch.cat((emb_token[None], x), 0)


        token_mask = torch.ones((bs, 1), dtype=bool, device=x.device)
        aug_mask = torch.cat((token_mask, in_mask), 1)


        xseq = self.sequence_pos_encoding(xseq)

        final = self.seqTransEncoder(xseq, src_key_padding_mask=~aug_mask)


        return final[0]
