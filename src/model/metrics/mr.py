

from typing import List
import torch
from torch import Tensor
from torchmetrics import Metric
from src.model.metrics.utils import calc_mpjpe, calc_pampjpe, calc_accel


class MRMetrics(Metric):


    def __init__(self,
                 njoints: int = 22,
                 jointstype: str = "smplnh",
                 force_in_meter: bool = True,
                 align_root: bool = True,
                 dist_sync_on_step: bool = True,
                 **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)


        if jointstype not in ["mmm", "humanml3d", "smplnh"]:
            raise NotImplementedError(f"This jointstype ({jointstype}) is not implemented. Supported: mmm, humanml3d, smplnh")

        self.name = 'Motion Reconstructions'
        self.jointstype = jointstype
        self.align_root = align_root
        self.force_in_meter = force_in_meter





        self.add_state("count", default=torch.tensor([0]), dist_reduce_fx="sum")
        self.add_state("count_seq", default=torch.tensor([0]), dist_reduce_fx="sum")
        self.add_state("MPJPE", default=torch.tensor([0.0]), dist_reduce_fx="sum")
        self.add_state("PAMPJPE", default=torch.tensor([0.0]), dist_reduce_fx="sum")
        self.add_state("ACCEL", default=torch.tensor([0.0]), dist_reduce_fx="sum")

        self.MR_metrics = ["MPJPE", "PAMPJPE", "ACCL"]
        self.metrics = self.MR_metrics



        self._has_updated = False

    def has_updated(self) -> bool:

        return self._has_updated

    def update(self, joints_rst: Tensor, joints_ref: Tensor, lengths: List[int]):

        assert joints_rst.shape == joints_ref.shape
        assert joints_rst.dim() == 4  # (B, T, njoints, 3)


        self._has_updated = True




        if torch.cuda.is_available():

            target_device = joints_rst.device if joints_rst.is_cuda else torch.device('cuda')


            if isinstance(self.count, torch.Tensor) and self.count.device.type != 'cuda':
                self.count = self.count.to(target_device)
            if isinstance(self.count_seq, torch.Tensor) and self.count_seq.device.type != 'cuda':
                self.count_seq = self.count_seq.to(target_device)
            if isinstance(self.MPJPE, torch.Tensor) and self.MPJPE.device.type != 'cuda':
                self.MPJPE = self.MPJPE.to(target_device)
            if isinstance(self.PAMPJPE, torch.Tensor) and self.PAMPJPE.device.type != 'cuda':
                self.PAMPJPE = self.PAMPJPE.to(target_device)
            if isinstance(self.ACCEL, torch.Tensor) and self.ACCEL.device.type != 'cuda':
                self.ACCEL = self.ACCEL.to(target_device)


            count_device = target_device
            count_seq_device = target_device
        else:

            count_device = self.count.device if isinstance(self.count, torch.Tensor) else torch.device('cpu')
            count_seq_device = self.count_seq.device if isinstance(self.count_seq, torch.Tensor) else torch.device('cpu')


        count_add = torch.tensor([sum(lengths)], device=count_device, dtype=torch.long)
        count_seq_add = torch.tensor([len(lengths)], device=count_seq_device, dtype=torch.long)
        self.count += count_add
        self.count_seq += count_seq_add


        rst = joints_rst.detach().cpu()
        ref = joints_ref.detach().cpu()



        if self.align_root and self.jointstype in ['mmm', 'humanml3d', 'smplnh']:
            align_inds = [0]
        else:
            align_inds = None





        if torch.cuda.is_available():
            target_device = self.MPJPE.device if isinstance(self.MPJPE, torch.Tensor) else torch.device('cuda')
        else:
            target_device = self.MPJPE.device if isinstance(self.MPJPE, torch.Tensor) else torch.device('cpu')

        for i in range(len(lengths)):

            seq_len = lengths[i]
            rst_seq = rst[i, :seq_len]  # [T, njoints, 3]
            ref_seq = ref[i, :seq_len]  # [T, njoints, 3]




            mpjpe_val = torch.sum(calc_mpjpe(rst_seq, ref_seq, align_inds=align_inds))
            if mpjpe_val.dim() == 0:
                mpjpe_val = mpjpe_val.unsqueeze(0)

            mpjpe_val = mpjpe_val.to(target_device)
            self.MPJPE += mpjpe_val


            pampjpe_val = torch.sum(calc_pampjpe(rst_seq, ref_seq))
            if pampjpe_val.dim() == 0:
                pampjpe_val = pampjpe_val.unsqueeze(0)

            pampjpe_val = pampjpe_val.to(target_device)
            self.PAMPJPE += pampjpe_val


            accel_val = torch.sum(calc_accel(rst_seq, ref_seq))
            if accel_val.dim() == 0:
                accel_val = accel_val.unsqueeze(0)

            accel_val = accel_val.to(target_device)
            self.ACCEL += accel_val

    def compute(self, sanity_flag: bool = False):



        if torch.cuda.is_available():

            target_device = torch.device('cuda')


            if isinstance(self.count, torch.Tensor) and self.count.device.type != 'cuda':
                self.count = self.count.to(target_device)
            if isinstance(self.count_seq, torch.Tensor) and self.count_seq.device.type != 'cuda':
                self.count_seq = self.count_seq.to(target_device)
            if isinstance(self.MPJPE, torch.Tensor) and self.MPJPE.device.type != 'cuda':
                self.MPJPE = self.MPJPE.to(target_device)
            if isinstance(self.PAMPJPE, torch.Tensor) and self.PAMPJPE.device.type != 'cuda':
                self.PAMPJPE = self.PAMPJPE.to(target_device)
            if isinstance(self.ACCEL, torch.Tensor) and self.ACCEL.device.type != 'cuda':
                self.ACCEL = self.ACCEL.to(target_device)

            device = target_device
        else:
            device = self.MPJPE.device if isinstance(self.MPJPE, torch.Tensor) else torch.device('cpu')

        if self.force_in_meter:

            factor = 1000.0
        else:
            factor = 1.0

        count = self.count
        count_seq = self.count_seq



        if isinstance(count, torch.Tensor):
            count_val = count.item() if count.numel() == 1 else count[0].item()
        else:
            count_val = count
        if isinstance(count_seq, torch.Tensor):
            count_seq_val = count_seq.item() if count_seq.numel() == 1 else count_seq[0].item()
        else:
            count_seq_val = count_seq


        if count_val == 0:

            return {
                "MPJPE": torch.tensor([0.0], device=device),
                "PAMPJPE": torch.tensor([0.0], device=device),
                "ACCEL": torch.tensor([0.0], device=device)
            }

        mr_metrics = {}




        mpjpe = self.MPJPE / count_val * factor
        pampjpe = self.PAMPJPE / count_val * factor


        if mpjpe.dim() == 0:
            mpjpe = mpjpe.unsqueeze(0)
        if pampjpe.dim() == 0:
            pampjpe = pampjpe.unsqueeze(0)


        accel_denominator = count_val - 2 * count_seq_val
        if accel_denominator <= 0:

            accel = torch.tensor([0.0], device=device)
        else:
            accel = self.ACCEL / accel_denominator * factor

            if accel.dim() == 0:
                accel = accel.unsqueeze(0)



        if torch.cuda.is_available():
            if mpjpe.device.type != 'cuda':
                mpjpe = mpjpe.cuda()
            if pampjpe.device.type != 'cuda':
                pampjpe = pampjpe.cuda()
            if accel.device.type != 'cuda':
                accel = accel.cuda()

        mr_metrics["MPJPE"] = mpjpe
        mr_metrics["PAMPJPE"] = pampjpe
        mr_metrics["ACCEL"] = accel

        return mr_metrics

    def reset(self):

        super().reset()

        self._has_updated = False




