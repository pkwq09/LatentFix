

from typing import List
import torch
import numpy as np
from torch import Tensor
from torchmetrics import Metric
from src.model.metrics.utils import (
    calculate_activation_statistics_np,
    calculate_frechet_distance_np,
    calculate_diversity_np
)


class VAEFIDMetrics(Metric):


    def __init__(self,
                 dist_sync_on_step: bool = True,
                 diversity_times: int = 300,
                 **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.name = "VAE FID and Diversity"
        self.diversity_times = diversity_times



        self.add_state("count", default=torch.tensor([0]), dist_reduce_fx="sum")
        self.add_state("count_seq", default=torch.tensor([0]), dist_reduce_fx="sum")
        self.add_state("FID", default=torch.tensor([0.0]), dist_reduce_fx="mean")
        self.add_state("Diversity", default=torch.tensor([0.0]), dist_reduce_fx="mean")
        self.add_state("gt_Diversity", default=torch.tensor([0.0]), dist_reduce_fx="mean")

        self.metrics = ["FID", "Diversity", "gt_Diversity"]


        self.add_state("recmotion_embeddings", default=[], dist_reduce_fx=None)
        self.add_state("gtmotion_embeddings", default=[], dist_reduce_fx=None)


        self._has_updated = False

    def has_updated(self) -> bool:

        return self._has_updated

    def update(
        self,
        gtmotion_embeddings: Tensor,
        lengths: List[int],
        recmotion_embeddings: Tensor = None,
    ):


        self._has_updated = True


        if torch.cuda.is_available():
            target_device = gtmotion_embeddings.device if gtmotion_embeddings.is_cuda else torch.device('cuda')

            if isinstance(self.count, torch.Tensor) and self.count.device.type != 'cuda':
                self.count = self.count.to(target_device)
            if isinstance(self.count_seq, torch.Tensor) and self.count_seq.device.type != 'cuda':
                self.count_seq = self.count_seq.to(target_device)
            if isinstance(self.FID, torch.Tensor) and self.FID.device.type != 'cuda':
                self.FID = self.FID.to(target_device)
            if isinstance(self.Diversity, torch.Tensor) and self.Diversity.device.type != 'cuda':
                self.Diversity = self.Diversity.to(target_device)
            if isinstance(self.gt_Diversity, torch.Tensor) and self.gt_Diversity.device.type != 'cuda':
                self.gt_Diversity = self.gt_Diversity.to(target_device)

            count_device = target_device
            count_seq_device = target_device
        else:
            count_device = self.count.device if isinstance(self.count, torch.Tensor) else torch.device('cpu')
            count_seq_device = self.count_seq.device if isinstance(self.count_seq, torch.Tensor) else torch.device('cpu')


        count_add = torch.tensor([sum(lengths)], device=count_device, dtype=torch.long)
        count_seq_add = torch.tensor([len(lengths)], device=count_seq_device, dtype=torch.long)
        self.count += count_add
        self.count_seq += count_seq_add



        if gtmotion_embeddings.dim() == 3:


            if gtmotion_embeddings.shape[0] < gtmotion_embeddings.shape[1]:
                gtmotion_embeddings = gtmotion_embeddings.permute(1, 0, 2)  # [B, latent_size, latent_dim]


        B = gtmotion_embeddings.shape[0]
        for i in range(B):

            sample_embedding = torch.flatten(gtmotion_embeddings[i], start_dim=0).detach().cpu()
            self.gtmotion_embeddings.append(sample_embedding)


        if recmotion_embeddings is not None:

            if recmotion_embeddings.dim() == 3:

                if recmotion_embeddings.shape[0] < recmotion_embeddings.shape[1]:
                    recmotion_embeddings = recmotion_embeddings.permute(1, 0, 2)  # [B, latent_size, latent_dim]


            B = recmotion_embeddings.shape[0]
            for i in range(B):

                sample_embedding = torch.flatten(recmotion_embeddings[i], start_dim=0).detach().cpu()
                self.recmotion_embeddings.append(sample_embedding)

    def compute(self, sanity_flag: bool = False):


        metrics = {metric: getattr(self, metric) for metric in self.metrics}


        if sanity_flag:
            return metrics


        if torch.cuda.is_available():
            target_device = torch.device('cuda')

            if isinstance(self.count, torch.Tensor) and self.count.device.type != 'cuda':
                self.count = self.count.to(target_device)
            if isinstance(self.count_seq, torch.Tensor) and self.count_seq.device.type != 'cuda':
                self.count_seq = self.count_seq.to(target_device)
            if isinstance(self.FID, torch.Tensor) and self.FID.device.type != 'cuda':
                self.FID = self.FID.to(target_device)
            if isinstance(self.Diversity, torch.Tensor) and self.Diversity.device.type != 'cuda':
                self.Diversity = self.Diversity.to(target_device)
            if isinstance(self.gt_Diversity, torch.Tensor) and self.gt_Diversity.device.type != 'cuda':
                self.gt_Diversity = self.gt_Diversity.to(target_device)

            device = target_device
        else:
            device = self.FID.device if isinstance(self.FID, torch.Tensor) else torch.device('cpu')


        if len(self.gtmotion_embeddings) == 0:
            return {
                "FID": torch.tensor([0.0], device=device),
                "Diversity": torch.tensor([0.0], device=device),
                "gt_Diversity": torch.tensor([0.0], device=device)
            }



        if len(self.gtmotion_embeddings) == 0:
            return {
                "FID": torch.tensor([0.0], device=device),
                "Diversity": torch.tensor([0.0], device=device),
                "gt_Diversity": torch.tensor([0.0], device=device)
            }


        all_gtmotions = torch.stack(self.gtmotion_embeddings, dim=0)  # [N, latent_size*latent_dim]


        all_gtmotions_np = all_gtmotions.numpy()



        gt_norms = np.linalg.norm(all_gtmotions_np, axis=1, keepdims=True)

        all_gtmotions_np = all_gtmotions_np / (gt_norms + 1e-8)


        gt_count_seq = all_gtmotions_np.shape[0]
        if gt_count_seq > self.diversity_times:
            gt_diversity_value = calculate_diversity_np(all_gtmotions_np, self.diversity_times)
            gt_diversity_tensor = torch.tensor([gt_diversity_value], device=device)
            metrics["gt_Diversity"] = gt_diversity_tensor
        else:
            metrics["gt_Diversity"] = torch.tensor([0.0], device=device)


        if len(self.recmotion_embeddings) == 0:
            return {
                "FID": torch.tensor([0.0], device=device),
                "Diversity": torch.tensor([0.0], device=device),
                **metrics
            }



        all_genmotions = torch.stack(self.recmotion_embeddings, dim=0)  # [N, latent_size*latent_dim]


        all_genmotions_np = all_genmotions.numpy()


        gen_norms = np.linalg.norm(all_genmotions_np, axis=1, keepdims=True)

        all_genmotions_np = all_genmotions_np / (gen_norms + 1e-8)


        mu, cov = calculate_activation_statistics_np(all_genmotions_np)
        gt_mu, gt_cov = calculate_activation_statistics_np(all_gtmotions_np)
        fid_value = calculate_frechet_distance_np(gt_mu, gt_cov, mu, cov)


        fid_tensor = torch.tensor([fid_value], device=device)
        metrics["FID"] = fid_tensor


        gen_count_seq = all_genmotions_np.shape[0]
        if gen_count_seq > self.diversity_times:
            diversity_value = calculate_diversity_np(all_genmotions_np, self.diversity_times)
            diversity_tensor = torch.tensor([diversity_value], device=device)
            metrics["Diversity"] = diversity_tensor
        else:
            metrics["Diversity"] = torch.tensor([0.0], device=device)

        return metrics

    def reset(self):

        super().reset()

        self._has_updated = False

