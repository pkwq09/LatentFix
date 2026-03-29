import numpy as np
import torch
import torch.nn as nn
from torchmetrics import Metric


class MLDLosses(Metric):
    """
    MLD Loss
    """

    def __init__(self, predict_epsilon, lmd_prior,
                 lmd_kl, lmd_gen, lmd_recons,
                 stage='diffusion',
                 lmd_joint=None,
                 **kwargs):
        super().__init__(dist_sync_on_step=True)


        self.predict_epsilon = predict_epsilon
        self.lmd_prior = lmd_prior
        self.lmd_kl = lmd_kl
        self.lmd_gen = lmd_gen
        self.lmd_recons = lmd_recons
        self.stage = stage
        self.lmd_joint = lmd_joint if lmd_joint is not None else lmd_recons
        losses = []


        losses.append("inst_loss")
        losses.append("x_loss")
        if lmd_prior != 0.0:
            losses.append("prior_loss")


        if self.stage in ['vae', 'vae_diffusion']:
            losses.append("recons_feature")
            losses.append("recons_joints")
            losses.append("kl_motion")

        losses.append("loss")


        for loss in losses:
            self.add_state(loss,
                           default=torch.tensor(0.0),
                           dist_reduce_fx="sum")
            # self.register_buffer(loss, torch.tensor(0.0))
        self.add_state("count", torch.tensor(0), dist_reduce_fx="sum")
        self.losses = losses

        self._losses_func = {}
        self._params = {}
        for loss in losses:
            if loss.split('_')[0] == 'inst':
                self._losses_func[loss] = nn.MSELoss(reduction='mean')
                self._params[loss] = 1
            elif loss.split('_')[0] == 'x':
                self._losses_func[loss] = nn.MSELoss(reduction='mean')
                self._params[loss] = 1
            elif loss.split('_')[0] == 'prior':
                self._losses_func[loss] = nn.MSELoss(reduction='mean')
                self._params[loss] = self.lmd_prior
            if loss.split('_')[0] == 'kl':
                if self.lmd_kl != 0.0:
                    self._losses_func[loss] = KLLoss()
                    self._params[loss] = self.lmd_kl
            elif loss.split('_')[0] == 'recons':
                self._losses_func[loss] = torch.nn.SmoothL1Loss(reduction='mean')

                if loss.split('_')[-1] == 'joints':

                    self._params[loss] = self.lmd_joint
                else:

                    self._params[loss] = self.lmd_recons
            elif loss.split('_')[0] == 'gen':
                self._losses_func[loss] = torch.nn.SmoothL1Loss(reduction='mean')
                self._params[loss] = self.lmd_gen
            else:
                ValueError("This loss is not recognized.")

    def update(self, rs_set):
        total_loss: float = 0.0

        if self.stage in ["vae", "vae_diffusion"]:


            total_loss += self._update_loss("recons_feature",
                                           rs_set['m_rst'],
                                           rs_set['m_ref'])



            if 'lengths' in rs_set and 'joints_rst' in rs_set and 'joints_ref' in rs_set:
                from src.data.tools.tensors import lengths_to_mask
                lengths = rs_set['lengths']
                mask = lengths_to_mask(lengths, device=rs_set['joints_rst'].device)


                B, T, njoints, _ = rs_set['joints_rst'].shape
                joints_rst_flat = rs_set['joints_rst'].reshape(B * T, njoints, 3)
                joints_ref_flat = rs_set['joints_ref'].reshape(B * T, njoints, 3)
                joints_rst_masked = joints_rst_flat[mask]  # [N_valid, 22, 3]
                joints_ref_masked = joints_ref_flat[mask]  # [N_valid, 22, 3]
                total_loss += self._update_loss("recons_joints",
                                               joints_rst_masked,
                                               joints_ref_masked)
            elif 'joints_rst' in rs_set and 'joints_ref' in rs_set:

                total_loss += self._update_loss("recons_joints",
                                               rs_set['joints_rst'],
                                               rs_set['joints_ref'])

            total_loss += self._update_loss("kl_motion",
                                           rs_set['dist_m'],
                                           rs_set['dist_ref'])
        else:


            if self.predict_epsilon:
                total_loss += self._update_loss("inst_loss", rs_set['noise_pred'],
                                            rs_set['noise'])
            else:
                total_loss += self._update_loss("x_loss", rs_set['pred'],
                                            rs_set['diff_in'])

            if self.lmd_prior != 0.0:
                total_loss += self._update_loss("prior_loss", rs_set['noise_prior'],
                                            rs_set['dist_m1'])

        self.loss += total_loss
        self.count += 1

        return total_loss

    def compute(self):
        count = getattr(self, "count")

        return {loss: getattr(self, loss) / count for loss in self.losses}

    def _update_loss(self, loss: str, outputs, inputs):

        val = self._losses_func[loss](outputs, inputs)
        getattr(self, loss).__iadd__(val)
        weighted_loss = self._params[loss] * val
        return weighted_loss

    def loss2logname(self, loss: str, split: str):


        if loss == "loss":
            log_name = f"total/{split}"
        else:
            loss_type, name = loss.split("_", 1)
            log_name = f"{loss_type}/{name}/{split}"
        return log_name

class KLLoss:

    def __init__(self):
        pass

    def __call__(self, q, p):
        div = torch.distributions.kl_divergence(q, p)
        return div.mean()

    def __repr__(self):
        return "KLLoss()"


class KLLossMulti:

    def __init__(self):
        self.klloss = KLLoss()

    def __call__(self, qlist, plist):

        return sum([self.klloss(q, p) for q, p in zip(qlist, plist)])

    def __repr__(self):
        return "KLLossMulti()"
