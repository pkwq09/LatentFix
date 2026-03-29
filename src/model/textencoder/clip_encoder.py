

import os
from typing import List, Union

import torch
from torch import Tensor, nn
from torch.distributions.distribution import Distribution
from src.utils.file_io import hack_path
import pytorch_lightning as pl

class ClipTextEncoder(pl.LightningModule):


    def __init__(
            self,
            modelpath: str,
            finetune: bool = False,
            last_hidden_state: bool = True,
            **kwargs
        ) -> None:

        super().__init__()
        self.save_hyperparameters(logger=False)
        from transformers import logging
        from transformers import AutoModel, AutoTokenizer
        logging.set_verbosity_error()


        os.environ["TOKENIZERS_PARALLELISM"] = "false"


        self.tokenizer = AutoTokenizer.from_pretrained(hack_path(modelpath))
        self.text_model = AutoModel.from_pretrained(hack_path(modelpath))


        if not finetune:
            self.text_model.training = False
            for p in self.text_model.parameters():
                p.requires_grad = False


        self.max_length = self.tokenizer.model_max_length
        if "clip" in modelpath:
            self.text_encoded_dim = self.text_model.config.text_config.hidden_size
            if last_hidden_state:
                self.variant = "clip_hidden"
            else:
                self.variant = "clip"
        elif "bert" in modelpath:
            self.variant = "bert"
            self.text_encoded_dim = self.text_model.config.hidden_size
        else:
            raise ValueError(f"Model {modelpath} not supported")

    def forward(self, texts: List[str]):


        if self.variant in ["clip", "clip_hidden"]:

            text_inputs = self.tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids.to(self.text_model.device)
            txt_att_mask = text_inputs.attention_mask.to(self.text_model.device)


            if text_input_ids.shape[-1] > self.tokenizer.model_max_length:
                text_input_ids = text_input_ids[:, :self.tokenizer.
                                                model_max_length]
        elif self.variant == "bert":

            text_inputs = self.tokenizer(texts,
                                         return_tensors="pt",
                                         padding=True)


        if self.variant == "clip":

            text_embeddings = self.text_model.get_text_features(
                text_input_ids.to(self.text_model.device))

            text_embeddings = text_embeddings.unsqueeze(1)

            txt_att_mask = torch.ones(
                (text_embeddings.shape[0], text_embeddings.shape[1]),
                dtype=torch.bool,
                device=text_embeddings.device,
            )
        elif self.variant == "clip_hidden":

            text_embeddings = self.text_model.text_model(text_input_ids,
                            # attention_mask=txt_att_mask
                            ).last_hidden_state
        elif self.variant == "bert":

            text_embeddings = self.text_model(
                **text_inputs.to(self.text_model.device)).last_hidden_state
        else:
            raise NotImplementedError(f"Model {self.name} not implemented")

        return text_embeddings, txt_att_mask.bool()
