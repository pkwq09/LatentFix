

import torch
import scipy.linalg
import numpy as np


def batch_compute_similarity_transform_torch(S1, S2):

    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.permute(0, 2, 1)
        S2 = S2.permute(0, 2, 1)
        transposed = True
    assert S2.shape[1] == S1.shape[1]


    mu1 = S1.mean(axis=-1, keepdims=True)
    mu2 = S2.mean(axis=-1, keepdims=True)

    X1 = S1 - mu1
    X2 = S2 - mu2


    var1 = torch.sum(X1**2, dim=1).sum(dim=1)


    K = X1.bmm(X2.permute(0, 2, 1))


    U, s, V = torch.svd(K)


    Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
    Z = Z.repeat(U.shape[0], 1, 1)
    Z[:, -1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0, 2, 1))))


    R = V.bmm(Z.bmm(U.permute(0, 2, 1)))


    scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1


    t = mu2 - (scale.unsqueeze(-1).unsqueeze(-1) * (R.bmm(mu1)))


    S1_hat = scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(S1) + t

    if transposed:
        S1_hat = S1_hat.permute(0, 2, 1)

    return S1_hat, (scale, R, t)


def compute_mpjpe(preds, target, valid_mask=None, pck_joints=None, sample_wise=True):

    assert preds.shape == target.shape, f"Shape mismatch: {preds.shape} vs {target.shape}"
    mpjpe = torch.norm(preds - target, p=2, dim=-1)  # [B, J]

    if pck_joints is None:
        if sample_wise:
            mpjpe_seq = ((mpjpe * valid_mask.float()).sum(-1) /
                         valid_mask.float().sum(-1)
                         if valid_mask is not None else mpjpe.mean(-1))
        else:
            mpjpe_seq = mpjpe[valid_mask] if valid_mask is not None else mpjpe
        return mpjpe_seq
    else:
        mpjpe_pck_seq = mpjpe[:, pck_joints]
        return mpjpe_pck_seq


def align_by_parts(joints, align_inds=None):

    if align_inds is None:
        return joints
    pelvis = joints[:, align_inds].mean(1)  # [B, 3]
    return joints - torch.unsqueeze(pelvis, dim=1)  # [B, J, 3]


def calc_mpjpe(preds, target, align_inds=[0], sample_wise=True, trans=None):


    valid_mask = target[:, :, 0] != -2.0
    if align_inds is not None:
        preds_aligned = align_by_parts(preds, align_inds=align_inds)
        if trans is not None:
            preds_aligned += trans
        target_aligned = align_by_parts(target, align_inds=align_inds)
    else:
        preds_aligned, target_aligned = preds, target
    mpjpe_each = compute_mpjpe(preds_aligned,
                               target_aligned,
                               valid_mask=valid_mask,
                               sample_wise=sample_wise)
    return mpjpe_each


def calc_accel(preds, target):

    assert preds.shape == target.shape, f"Shape mismatch: {preds.shape} vs {target.shape}"
    assert preds.dim() == 3, f"Expected 3D tensor [T, J, 3], got {preds.shape}"


    accel_gt = target[:-2] - 2 * target[1:-1] + target[2:]
    accel_pred = preds[:-2] - 2 * preds[1:-1] + preds[2:]
    normed = torch.linalg.norm(accel_pred - accel_gt, dim=-1)  # [T-2, J]
    accel_seq = normed.mean(1)  # [T-2]
    return accel_seq


def calc_pampjpe(preds, target, sample_wise=True, return_transform_mat=False):


    target, preds = target.float(), preds.float()

    preds_tranformed, PA_transform = batch_compute_similarity_transform_torch(
        preds, target)
    pa_mpjpe_each = compute_mpjpe(preds_tranformed,
                                  target,
                                  sample_wise=sample_wise)

    if return_transform_mat:
        return pa_mpjpe_each, PA_transform
    else:
        return pa_mpjpe_each


def calculate_activation_statistics_np(activations):

    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov


def calculate_frechet_distance_np(mu1, sigma1, mu2, sigma2, eps=1e-6):

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (mu1.shape == mu2.shape
            ), "Training and test mean vectors have different lengths"
    assert (sigma1.shape == sigma2.shape
            ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ("fid calculation produces singular product; "
               "adding %s to diagonal of cov estimates") % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))


    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real
    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(
        sigma2) - 2 * tr_covmean


def calculate_diversity_np(activation, diversity_times):

    assert len(activation.shape) == 2, f"Expected 2D array [num_samples, dim_feat], got shape {activation.shape}"
    assert activation.shape[0] > diversity_times, \
        f"Number of samples ({activation.shape[0]}) must be greater than diversity_times ({diversity_times})"

    num_samples = activation.shape[0]


    first_indices = np.random.choice(num_samples,
                                     diversity_times,
                                     replace=False)
    second_indices = np.random.choice(num_samples,
                                      diversity_times,
                                      replace=False)


    dist = scipy.linalg.norm(activation[first_indices] -
                             activation[second_indices],
                             axis=1)


    return dist.mean()










