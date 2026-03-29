

import os
from omegaconf import DictConfig
import logging
import hydra
import yaml
from tqdm import tqdm
from pathlib import Path
import numpy as np
import torch
from typing import List, Dict
from torch import Tensor

from src.model.metrics.utils import (
    calculate_activation_statistics_np,
    calculate_frechet_distance_np,
    calculate_diversity_np,
)

from src.utils.file_io import write_json, read_json

logger = logging.getLogger(__name__)


mat2name = {
            'sim_matrix_s_t': 'source_target',
            'sim_matrix_t_t': 'target_generated'
            }

import os
import json
from omegaconf import DictConfig, OmegaConf


def save_config(cfg: DictConfig) -> str:

    path = os.path.join(cfg.run_dir, "config.json")
    config = OmegaConf.to_container(cfg, resolve=True)
    with open(path, "w") as f:
        string = json.dumps(config, indent=4)
        f.write(string)
    return path


def read_config(run_dir: str, return_json=False) -> DictConfig:

    path = os.path.join(run_dir, "config.json")
    with open(path, "r") as f:
        config = json.load(f)
    if return_json:
        return config
    cfg = OmegaConf.create(config)
    cfg.run_dir = run_dir
    return cfg


def length_to_mask(length, device: torch.device = None) -> Tensor:

    if device is None:
        device = "cpu"

    if isinstance(length, list):
        length = torch.tensor(length, device=device)

    max_len = max(length)
    mask = torch.arange(max_len, device=device).expand(
        len(length), max_len
    ) < length.unsqueeze(1)
    return mask

def l2_norm(x1, x2, dim):
    return torch.linalg.vector_norm(x1 - x2, ord=2, dim=dim)


def save_metric(path, metrics):
    strings = yaml.dump(metrics, indent=4, sort_keys=False)
    with open(path, "w") as f:
        f.write(strings)

def line2dict(line):

    names_of_metrics = ["R@1_s2t", "R@2_s2t", "R@3_s2t", "R@5_s2t", "R@10_s2t", "MedR_s2t", "AvgR_s2t",
                        "R@1", "R@2", "R@3", "R@5", "R@10", "MedR", "AvgR"]
    metrics_nos = line.replace('\\', '').split('&')
    metrics_nos = [x.strip() for x in metrics_nos if x]
    return dict(zip(names_of_metrics, metrics_nos))

def lengths_to_mask_njoints(lengths: List[int], njoints: int, device: torch.device) -> Tensor:


    joints_lengths = [njoints*l for l in lengths]
    joints_mask = lengths_to_mask(joints_lengths, device)
    return joints_mask


def lengths_to_mask(lengths: List[int], device: torch.device) -> Tensor:

    lengths = torch.tensor(lengths, device=device)
    max_len = max(lengths)
    mask = torch.arange(max_len,
                        device=device).expand(len(lengths),
                                              max_len) < lengths.unsqueeze(1)
    return mask

def collect_gen_samples(gener_motions, normalizer, device):

    cur_samples = {}
    cur_samples_raw = {}
    from src.data.features import _get_body_transl_delta_pelv_infer

    if isinstance(gener_motions, str):
        # you have a path and not the motions themselves
        import glob
        sample_files = glob.glob(f'{gener_motions}/*.npy')
        for fname in tqdm(sample_files):
            keyid = str(Path(fname).name).replace('.npy', '')
            gen_motion_b = np.load(fname,
                                allow_pickle=True).item()['pose']
            gen_motion_b = torch.from_numpy(gen_motion_b)
            trans = gen_motion_b[..., :3]
            global_orient_6d = gen_motion_b[..., 3:9]
            body_pose_6d = gen_motion_b[..., 9:]
            trans_delta = _get_body_transl_delta_pelv_infer(global_orient_6d,
                                                    trans)
            gen_motion_b_fixed = torch.cat([trans_delta, body_pose_6d,
                                            global_orient_6d], dim=-1)
            gen_motion_b_fixed = normalizer(gen_motion_b_fixed)
            cur_samples[keyid] = gen_motion_b_fixed.to(device)
            cur_samples_raw[keyid] = torch.cat([trans, global_orient_6d,
                                                body_pose_6d], dim=-1).to(device)
    elif isinstance(gener_motions, dict):

        for keyid, motion_feats in gener_motions.items():
            trans = motion_feats[..., :3]
            global_orient_6d = motion_feats[..., 3:9]
            body_pose_6d = motion_feats[..., 9:]
            trans_delta = _get_body_transl_delta_pelv_infer(global_orient_6d,
                                                    trans)
            gen_motion_b_fixed = torch.cat([trans_delta, body_pose_6d,
                                            global_orient_6d], dim=-1)
            gen_motion_b_fixed = normalizer(gen_motion_b_fixed)
            cur_samples[keyid] = gen_motion_b_fixed.to(device)
            cur_samples_raw[keyid] = torch.cat([trans, global_orient_6d,
                                                body_pose_6d], dim=-1).to(device)
    else:

        raise TypeError(
            f"collect_gen_samples expected dict {{keyid: motion_tensor}} or a path string, "
            f"but got type: {type(gener_motions)}. "
            f"This may be due to VAE-stage validation_step_outputs format differing from diffusion. "
            f"Ensure samples_gen is a dict before calling retrieval."
        )

    return cur_samples, cur_samples_raw


def _collect_motion_embeddings_for_metrics(
    model,
    dataset,
    keyids,
    gen_samples,
    batch_size: int = 256,
    progress: bool = False,
):

    if not isinstance(gen_samples, dict) or len(gen_samples) == 0:
        return None, None, []

    from src.data.tools.collate import collate_tensor_with_padding

    device = model.device
    if batch_size > len(dataset):
        batch_size = len(dataset)
    nsplit = int(np.ceil(len(dataset) / max(batch_size, 1)))

    embeddings_gt = []
    embeddings_gen = []
    lengths_all = []

    with torch.no_grad():
        all_data = [dataset.load_keyid(keyid) for keyid in keyids]
        if nsplit > len(all_data):
            nsplit = len(all_data) if len(all_data) > 0 else 1
        all_data_splitted = np.array_split(all_data, nsplit) if all_data else []

        data_iter = tqdm(all_data_splitted, leave=False) if progress else all_data_splitted
        for data in data_iter:
            if len(data) == 0:
                continue
            target_motions = []
            gen_motions = []
            lengths_batch = []
            for sample in data:
                keyid = sample['keyid']
                if keyid not in gen_samples:
                    continue
                gen_motion = gen_samples[keyid]
                tgt_motion = sample['motion_target']
                clip_len = min(len(tgt_motion), gen_motion.shape[0])
                if clip_len == 0:
                    continue
                target_motions.append(tgt_motion[:clip_len])
                gen_motions.append(gen_motion[:clip_len])
                lengths_batch.append(clip_len)

            if len(target_motions) == 0:
                continue

            motion_gt = collate_tensor_with_padding(target_motions).to(device)
            motion_gen = collate_tensor_with_padding(gen_motions).to(device)

            masks = length_to_mask(lengths_batch, device=motion_gt.device)
            motion_gt_dict = {'length': lengths_batch, 'mask': masks, 'x': motion_gt}
            motion_gen_dict = {'length': lengths_batch, 'mask': masks, 'x': motion_gen}

            latent_gt = model.encode(motion_gt_dict, sample_mean=True)
            latent_gen = model.encode(motion_gen_dict, sample_mean=True)

            embeddings_gt.append(latent_gt.detach().cpu())
            embeddings_gen.append(latent_gen.detach().cpu())
            lengths_all.extend(lengths_batch)

    if len(embeddings_gt) == 0 or len(embeddings_gen) == 0:
        return None, None, []

    gt_embeddings = torch.cat(embeddings_gt, dim=0)
    gen_embeddings = torch.cat(embeddings_gen, dim=0)
    return gt_embeddings, gen_embeddings, lengths_all


def compute_fid_diversity_metrics(
    model,
    dataset,
    keyids,
    gen_samples,
    batch_size: int = 256,
    diversity_times: int = 300,
):

    gt_embeddings, gen_embeddings, lengths = _collect_motion_embeddings_for_metrics(
        model,
        dataset,
        keyids,
        gen_samples,
        batch_size=batch_size,
    )

    if gt_embeddings is None or gen_embeddings is None or gt_embeddings.shape[0] == 0:
        return {}




    if gt_embeddings.ndim == 3:

        print(f"⚠️  Detected 3D embeddings, using mean pooling: {gt_embeddings.shape}")
        gt_np = gt_embeddings.mean(dim=1).numpy()  # [N, T, D] -> [N, D]
        gen_np = gen_embeddings.mean(dim=1).numpy()
    elif gt_embeddings.ndim == 2:

        gt_np = gt_embeddings.numpy()  # [N, D]
        gen_np = gen_embeddings.numpy()
    else:
        raise ValueError(
            f"Unexpected embedding dimension: {gt_embeddings.ndim}D. "
            f"Expected 2D [N, D] for TMR or 3D [N, T, D] for VAE latents. "
            f"Got shape: {gt_embeddings.shape}"
        )



    gt_norms = np.linalg.norm(gt_np, axis=1, keepdims=True)
    gen_norms = np.linalg.norm(gen_np, axis=1, keepdims=True)

    gt_np = gt_np / (gt_norms + 1e-8)
    gen_np = gen_np / (gen_norms + 1e-8)

    fid_metrics = {}


    if not hasattr(compute_fid_diversity_metrics, '_debug_printed'):
        print(f"\n🔍 [FID Computation Info]")
        print(f"   Original embeddings shape: {gt_embeddings.shape}")
        print(f"   Final numpy array shape: {gt_np.shape}")
        print(f"   Feature dimension: {gt_np.shape[1]}")
        print(f"   Number of samples: {gt_np.shape[0]}")
        print(f"   Aligned with official MotionFix: ✅ No flatten operation")
        print(f"   L2 Normalization: ✅ Applied (mean norm before: {gt_norms.mean():.4f})")
        print(f"   L2 Normalization: ✅ Applied (mean norm after: {np.linalg.norm(gt_np, axis=1).mean():.4f})\n")
        compute_fid_diversity_metrics._debug_printed = True


    mu_gen, cov_gen = calculate_activation_statistics_np(gen_np)
    mu_gt, cov_gt = calculate_activation_statistics_np(gt_np)
    fid_metrics["FID"] = float(
        calculate_frechet_distance_np(mu_gt, cov_gt, mu_gen, cov_gen)
    )


    if gen_np.shape[0] > diversity_times:
        fid_metrics["Diversity"] = float(
            calculate_diversity_np(gen_np, diversity_times)
        )
    else:
        fid_metrics["Diversity"] = 0.0

    if gt_np.shape[0] > diversity_times:
        fid_metrics["gt_Diversity"] = float(
            calculate_diversity_np(gt_np, diversity_times)
        )
    else:
        fid_metrics["gt_Diversity"] = 0.0

    fid_metrics["num_samples"] = len(lengths)
    return fid_metrics

def run_smpl_fwd(body_transl, body_orient, body_pose, body_model):

    from src.tools.transforms3d import transform_body_pose

    if len(body_transl.shape) > 2:
        bs, seqlen = body_transl.shape[:2]
        body_transl = body_transl.flatten(0, 1)
        body_orient = body_orient.flatten(0, 1)
        body_pose = body_pose.flatten(0, 1)
    else:
        bs = 1
        seqlen = body_transl.shape[0]

    batch_size = body_transl.shape[0]
    body_model.batch_size = batch_size


    if hasattr(body_model, 'smpl_forward_fast'):
        output = body_model.smpl_forward_fast(
            transl=body_transl,
            body_pose=transform_body_pose(body_pose, '6d->rot'),
            global_orient=transform_body_pose(body_orient, '6d->rot'),
            return_verts=False
        )
    else:
        output = body_model(
            transl=body_transl,
            body_pose=transform_body_pose(body_pose, '6d->rot'),
            global_orient=transform_body_pose(body_orient, '6d->rot')
        )

    jts = output.joints[:, :22]
    return jts.reshape(bs, seqlen, -1, 3)


def get_motion_distances(model, dataset, keyids, gen_samples_raw, batch_size=256, body_model=None):

    import torch
    import numpy as np
    from src.data.tools.collate import collate_tensor_with_padding

    device = model.device
    if batch_size > len(dataset):
        batch_size = len(dataset)
    nsplit = int(np.ceil(len(dataset) / batch_size))

    if body_model is None:
        import smplx
        from src.model.utils.smpl_fast import smpl_forward_fast
        body_model = smplx.SMPLHLayer('data/body_models/smplh',
                                      model_type='smplh',
                                      gender='neutral',
                                      ext='npz').to(device).eval()

        setattr(smplx.SMPLHLayer, 'smpl_forward_fast', smpl_forward_fast)

    with torch.no_grad():
        all_data = [dataset.load_keyid_raw(keyid) for keyid in keyids]
        if nsplit > len(all_data):
            nsplit = len(all_data)
        all_data_splitted = np.array_split(all_data, nsplit)

        motions_a = []
        motions_b = []
        tot_lens_a = []
        tot_lens_b = []

        for data in tqdm(all_data_splitted, leave=False, desc="Computing L2 distance"):
            keyids_of_cursplit = [x['keyid'] for x in data]


            motion_a = collate_tensor_with_padding(
                [x['motion_target'] for x in data]).to(device)
            lengths_a = [len(x['motion_target']) for x in data]


            cur_samples = [gen_samples_raw[kd][:lengths_a[ix]]
                          for ix, kd in enumerate(keyids_of_cursplit)]
            lengths_b = [len(x) for x in cur_samples]
            motion_b = collate_tensor_with_padding(cur_samples).to(device)


            def split_into_chunks(N, k):
                chunked = [k*i for i in range(1, N//k+1)] + ([N] if N%k else [])
                return [0] + chunked


            ids_for_smpl = split_into_chunks(motion_a.shape[0], 64)

            for i in range(len(ids_for_smpl) - 1):
                s, e = ids_for_smpl[i], ids_for_smpl[i+1]

                jts_a = run_smpl_fwd(motion_a[s:e, :, :3],
                                    motion_a[s:e, :, 3:9],
                                    motion_a[s:e, :, 9:],
                                    body_model)
                jts_b = run_smpl_fwd(motion_b[s:e, :, :3],
                                    motion_b[s:e, :, 3:9],
                                    motion_b[s:e, :, 9:],
                                    body_model)
                motions_a.append(jts_a.detach().cpu())
                motions_b.append(jts_b.detach().cpu())
            tot_lens_a.extend(lengths_a)
            tot_lens_b.extend(lengths_b)

        mask_a = length_to_mask(tot_lens_a, device='cpu')


        max_len = max(max(x.shape[1] for x in motions_a),
                     max(x.shape[1] for x in motions_b))

        motions_a_proc = []
        for x in motions_a:
            if x.shape[1] != max_len:
                zeros_to_add = torch.zeros(x.size(0), max_len - x.shape[1], 22, 3)
                motions_a_proc.append(torch.cat((x, zeros_to_add), dim=1))
            else:
                motions_a_proc.append(x)

        motions_b_proc = []
        for x in motions_b:
            if x.shape[1] != max_len:
                zeros_to_add = torch.zeros(x.size(0), max_len - x.shape[1], 22, 3)
                motions_b_proc.append(torch.cat((x, zeros_to_add), dim=1))
            else:
                motions_b_proc.append(x)

        motions_a = torch.cat(motions_a_proc)
        motions_b = torch.cat(motions_b_proc)


        l2_distance = torch.sqrt(torch.sum((motions_a - motions_b) ** 2, dim=-1))
        mask_expanded = mask_a.unsqueeze(-1)
        masked_l2_distance = l2_distance * mask_expanded
        total_distance = masked_l2_distance.sum()
        valid_elements = mask_a.sum() * l2_distance.shape[-1]
        mean_distance = total_distance / valid_elements

    return mean_distance.item()


def compute_gt_sim_matrix(model, dataset, keyids, batch_size=256, progress=True):

    import torch
    import numpy as np
    from src.data.tools.collate import collate_tensor_with_padding
    from src.tmr.tmr import get_sim_matrix

    device = model.device
    if batch_size > len(dataset):
        batch_size = len(dataset)
    nsplit = int(np.ceil(len(dataset) / batch_size))
    returned = {}
    keyids_ordered = {}

    with torch.no_grad():
        all_data = [dataset.load_keyid(keyid) for keyid in keyids]
        if nsplit > len(all_data):
            nsplit = len(all_data)
        all_data_splitted = np.array_split(all_data, nsplit)


        for sett in ['s_t', 't_t']:
            latent_motions_A = []
            latent_motions_B = []
            keys_ordered_for_run = []

            if progress:
                data_iter = tqdm(all_data_splitted, leave=False, desc=f"Computing GT {sett}")
            else:
                data_iter = all_data_splitted

            for data in data_iter:
                cur_batch_keys = [x['keyid'] for x in data]
                keys_ordered_for_run.extend(cur_batch_keys)

                if sett == 's_t':

                    motion_a = collate_tensor_with_padding(
                        [x['motion_source'] for x in data]).to(model.device)
                    lengths_a = [len(x['motion_source']) for x in data]
                    motion_b = collate_tensor_with_padding(
                        [x['motion_target'] for x in data]).to(model.device)
                    lengths_b = [len(x['motion_target']) for x in data]
                elif sett == 't_t':

                    motion_a = collate_tensor_with_padding(
                        [x['motion_target'] for x in data]).to(model.device)
                    lengths_a = [len(x['motion_target']) for x in data]
                    motion_b = collate_tensor_with_padding(
                        [x['motion_target'] for x in data]).to(model.device)
                    lengths_b = [len(x['motion_target']) for x in data]

                masks_a = length_to_mask(lengths_a, device=motion_a.device)
                masks_b = length_to_mask(lengths_b, device=motion_b.device)
                motion_a_dict = {'length': lengths_a, 'mask': masks_a, 'x': motion_a}
                motion_b_dict = {'length': lengths_b, 'mask': masks_b, 'x': motion_b}

                # Encode both motions
                latent_motion_A = model.encode(motion_a_dict, sample_mean=True)
                latent_motion_B = model.encode(motion_b_dict, sample_mean=True)
                latent_motions_A.append(latent_motion_A)
                latent_motions_B.append(latent_motion_B)

            latent_motions_A = torch.cat(latent_motions_A)
            latent_motions_B = torch.cat(latent_motions_B)
            sim_matrix = get_sim_matrix(latent_motions_A, latent_motions_B)
            returned[f'sim_matrix_{sett}'] = sim_matrix.cpu().numpy()
            keyids_ordered[sett] = keys_ordered_for_run

    return returned, keyids_ordered


def compute_sim_matrix(model, dataset, keyids, gen_samples,
                       batch_size=256, progress=True):

    import torch
    import numpy as np
    from src.data.tools.collate import collate_text_motion
    from src.tmr.tmr import get_sim_matrix
    import numpy as np
    device = model.device
    if batch_size > len(dataset):
        batch_size = len(dataset)
    nsplit = int(np.ceil(len(dataset) / batch_size))
    returned = {}
    keyids_ordered = {}
    with torch.no_grad():

        all_data = [dataset.load_keyid(keyid) for keyid in keyids]
        if nsplit > len(all_data):
            nsplit = len(all_data)
        all_data_splitted = np.array_split(all_data, nsplit)
        # by batch (can be too costly on cuda device otherwise)
        for sett in ['s_t', 't_t']:
            cur_samples = []
            latent_motions_A = []
            latent_motions_B = []
            keys_ordered_for_run = []

            if progress:
                data_iter = tqdm(all_data_splitted, leave=False)
            else:
                data_iter = all_data_splitted
            for data in data_iter:
                # batch = collate_text_motion(data, device=device)
                from src.data.tools.collate import collate_tensor_with_padding
                cur_batch_keys = [x['keyid'] for x in data]
                keys_ordered_for_run.extend(cur_batch_keys)
                # TODO load the motions for the generations
                # Text is already encoded
                if sett == 's_t':
                    motion_a = collate_tensor_with_padding(
                        [x['motion_source'] for x in data]).to(model.device)
                    lengths_a = [len(x['motion_source']) for x in data]
                    lengths_tgt = [len(x['motion_target']) for x in data]
                    if gen_samples:
                        cur_samples = [gen_samples[key_in_batch][:lengths_tgt[ix]] for ix, key_in_batch in enumerate(cur_batch_keys)]
                        lengths_b = [len(x) for x in cur_samples]
                        motion_b = collate_tensor_with_padding(
                            cur_samples).to(model.device)
                    else:
                        motion_b = collate_tensor_with_padding(
                           [x['motion_target'] for x in data]).to(model.device)
                        lengths_b = [len(x['motion_target']) for x in data]

                    masks_a = length_to_mask(lengths_a, device=motion_a.device)
                    masks_b = length_to_mask(lengths_b, device=motion_b.device)
                    motion_a_dict = {'length': lengths_a, 'mask': masks_a,
                                    'x': motion_a}
                    motion_b_dict = {'length': lengths_b, 'mask': masks_b,
                                    'x': motion_b}
                elif sett == 't_t':
                    motion_a = collate_tensor_with_padding(
                        [x['motion_target'] for x in data]).to(model.device)
                    lengths_a = [len(x['motion_target']) for x in data]
                    lengths_tgt = [len(x['motion_target']) for x in data]

                    if gen_samples:
                        cur_samples = [gen_samples[key_in_batch][:lengths_tgt[ix]] for ix, key_in_batch in enumerate(cur_batch_keys)]
                        lengths_b = [len(x) for x in cur_samples]
                        motion_b = collate_tensor_with_padding(cur_samples
                                                               ).to(model.device)
                    else:
                        motion_b = collate_tensor_with_padding([
                            x['motion_target'] for x in data]).to(
                                model.device)
                        lengths_b = [len(x['motion_target']) for x in data]

                    masks_a = length_to_mask(lengths_a, device=motion_a.device)
                    masks_b = length_to_mask(lengths_b, device=motion_b.device)
                    motion_a_dict = {'length': lengths_a, 'mask': masks_a,
                                    'x': motion_a}
                    motion_b_dict = {'length': lengths_b, 'mask': masks_b,
                                    'x': motion_b}

                # Encode both motion and text
                latent_motion_A = model.encode(motion_a_dict,
                                            sample_mean=True)
                latent_motion_B = model.encode(motion_b_dict,
                                            sample_mean=True)
                latent_motions_A.append(latent_motion_A)
                latent_motions_B.append(latent_motion_B)

            latent_motions_A = torch.cat(latent_motions_A)
            latent_motions_B = torch.cat(latent_motions_B)
            sim_matrix = get_sim_matrix(latent_motions_A, latent_motions_B)
            returned[f'sim_matrix_{sett}'] = sim_matrix.cpu().numpy()
            keyids_ordered[sett] = keys_ordered_for_run
    return returned, keyids_ordered

def shorten_metric_line(line_to_shorten):
    # Split the string into a list of numbers
    numbers = line_to_shorten.split('&')

    # Remove the elements at the 4th, 5th, 6th, 11th, 12th, and 13th indices
    indices_to_remove = [4, 5, 6, 11, 12, 13]
    for index in sorted(indices_to_remove, reverse=True):
        del numbers[index]

    # Join the list back into a string
    return '&'.join(numbers)

def retrieval(samples_to_eval, evaluate_gt=False, compute_distance=True):

    protocol = ['normal', 'batches']
    device = 'cuda'
    run_dir = 'eval-deps'
    ckpt_name = 'last'
    batch_size = 256

    protocols = protocol
    dataset = 'motionfix' # motionfix
    sets = 'test' # val all
    # save_dir = os.path.join(run_dir, "motionfix/contrastive_metrics")
    # os.makedirs(save_dir, exist_ok=True)

    # Load last config
    curdir = Path(hydra.utils.get_original_cwd())

    cfg = read_config(curdir / run_dir)

    import pytorch_lightning as pl
    import numpy as np
    from hydra.utils import instantiate
    from src.tmr.load_model import load_model_from_cfg
    from src.tmr.metrics import all_contrastive_metrics_mot2mot, print_latex_metrics_m2m

    pl.seed_everything(cfg.seed)

    logger.info("Loading the evaluation TMR model")
    model = load_model_from_cfg(cfg, ckpt_name, eval_mode=True, device=device)

    datasets = {}
    results = {}
    keyids_ord = {}
    bs_m2m = 32 # for the batch size metric
    # calculate splits
    from src.tmr.data.motionfix_loader import Normalizer
    normalizer = Normalizer(curdir/run_dir/'stats/humanml3d/amass_feats')
    gen_samples, gen_samples_raw = collect_gen_samples(samples_to_eval,
                                        normalizer,
                                        model.device)
    exist_gen_keys = list(gen_samples.keys())
    if sets == 'all':
        sets_to_load = ['val', 'test']
        extra_str = '_val_test'
    elif sets == 'val':
        sets_to_load = ['val']
        extra_str = '_val'
    else:
        sets_to_load = ['test']
        extra_str = ''

    for protocol in protocols:
        # logger.info(f"|------Protocol {protocol.upper()}-----|")
        # Load the dataset if not already
        if protocol not in datasets:
            from src.tmr.data.motionfix_loader import MotionFixLoader
            dataset = MotionFixLoader(sets=sets_to_load,
                                      keys_to_load=exist_gen_keys)

            datasets.update(
                {key: dataset for key in ["normal", "batches"]}
            )
        gen_samples = {k:v for k, v in gen_samples.items() if k in dataset.motions.keys()}
        dataset = datasets[protocol]

        # Compute sim_matrix for each protocol
        if protocol not in results:
            if protocol=="normal":
                res, keyids_ord_for_all = compute_sim_matrix(
                    model, dataset, dataset.keyids,
                    gen_samples=gen_samples,
                    batch_size=batch_size,
                )
                keyids_ord['all'] = keyids_ord_for_all
                results.update({key: res for key in ["normal"]})
                # dists = get_motion_distances(
                #     model, dataset, dataset.keyids,
                #     gen_samples=gen_samples_raw,
                #     batch_size=batch_size,
                # )

            elif protocol == "batches":
                keyids = sorted(dataset.keyids)
                N = len(keyids)

                # make batches of 32
                idx = np.arange(N)
                np.random.seed(0)
                np.random.shuffle(idx)
                idx_batches = [
                    idx[bs_m2m * i : bs_m2m * (i + 1)] for i in range(len(keyids) // bs_m2m)
                ]

                # split into batches of 32
                # batched_keyids = [ [32], [32], [...]]
                results["batches"] = []
                keyids_ord["batches"] = []
                for idx_batch in tqdm(idx_batches):
                    res_matrs, res_keys = compute_sim_matrix(model, dataset,
                                                             np.array(keyids)[idx_batch],
                                                             gen_samples=gen_samples,
                                                             batch_size=batch_size,
                                                             progress=False)
                    results["batches"].append(res_matrs)
                    keyids_ord["batches"].append(res_keys)
                # results_v2v["guo"] = [
                #     get_motion_distances(
                #         model,
                #         dataset,
                #         np.array(keyids)[idx_batch],
                #         gen_samples=gen_samples_raw,
                #         batch_size=batch_size,
                #     )
                #     for idx_batch in idx_batches
                # ]

        result = results[protocol]

        # Compute the metrics
        if protocol == "batches":
            protocol_name = protocol
            def compute_batches_metrics(sim_matrix_lst):
                all_metrics = []
                all_cols = []
                for sim_matrix in sim_matrix_lst:
                    metrics, cols_for_metr = all_contrastive_metrics_mot2mot(sim_matrix,
                                                      rounding=None,  return_cols=True)
                    all_metrics.append(metrics)
                    all_cols.append(cols_for_metr)

                avg_metrics = {}
                for key in all_metrics[0].keys():
                    avg_metrics[key] = round(
                        float(np.mean([metrics[key] for metrics in all_metrics])), 2
                    )
                return avg_metrics, all_cols
            metrics_dico = {}
            result_packed_to_d = {key: [d[key] for d in result]
                                  for key in result[0]
                                  }
            keyids_ord['batches'] = {key: [d[key] for d in keyids_ord["batches"]]
                        for key in keyids_ord["batches"][0]
                        }
            str_for_tab = ''

            for var, lst_of_sim_matrs in result_packed_to_d.items():
                metr_name = mat2name[var]
                if var == 'sim_matrix_s_t':
                    keyids_for_sel = keyids_ord['batches']['s_t']
                else:
                    keyids_for_sel = keyids_ord['batches']['t_t']

                metrics_dico[metr_name], cols_for_metr_temp = compute_batches_metrics(lst_of_sim_matrs)

                idxs_good = [np.where(el < 2)[0] for el in cols_for_metr_temp]
                cols_for_metr_unmerged = [list(np.array(for_sel_cur
                                                         )[idxs]) for idxs, for_sel_cur in zip(idxs_good, keyids_for_sel)]
                cols_for_metr[metr_name] = [item for sublist in cols_for_metr_unmerged for item in sublist]
                str_for_tab += print_latex_metrics_m2m(metrics_dico[metr_name])
                metric_name = f"{protocol_name}_{metr_name}.yaml"
            cand_keyids_batches = cols_for_metr['target_generated']
            # if motion_gen_path is not None:
            #     write_json(cand_keyids_guo, Path(motion_gen_path) / f'guo_candkeyids{extra_str}.json')
            line_for_batches = str_for_tab.replace("\\\&", "&")

        else:
            protocol_name = protocol
            emb, threshold = None, None
            metrics = {}
            cols_for_metr = {}
            str_for_tab = ''
            for var, sim_matrix in result.items():
                metr_name = mat2name[var]
                if var == 'sim_matrix_s_t':
                    keyids_for_sel = keyids_ord['all']['s_t']
                else:
                    keyids_for_sel = keyids_ord['all']['t_t']

                metrics[metr_name], cols_for_metr_temp = all_contrastive_metrics_mot2mot(sim_matrix,
                                                    emb, threshold=threshold, return_cols=True)
                idxs_good = np.where(cols_for_metr_temp < 5)[0]
                cols_for_metr[metr_name] = list(np.array(keyids_for_sel
                                                         )[idxs_good])
                str_for_tab += print_latex_metrics_m2m(metrics[metr_name])
            cand_keyids_all = cols_for_metr['target_generated']
            line_for_all = str_for_tab.replace("\\\&", "&")
            # TODO do this at some point!
            # run = wandb.init()
            # my_table = wandb.Table(columns=["a", "b"],
            #                        data=[["1a", "1b"], ["2a", "2b"]])
            # run.log({"table_key": my_table})

    dict_batches = line2dict(line_for_batches)
    dict_full = line2dict(line_for_all)
    names_to_keep = ["R@1_s2t", "R@2_s2t", "R@3_s2t", "AvgR_s2t",
                    "R@1", "R@2", "R@3", "AvgR"]
    metrs_full = {key: dict_full[key] for key in names_to_keep if key in dict_full}
    metrs_batches = {key: dict_batches[key] for key in names_to_keep if key in dict_batches}

    fid_dataset = datasets.get("normal", dataset)
    fid_metrics = {}
    if fid_dataset is not None:
        fid_metrics = compute_fid_diversity_metrics(
            model,
            fid_dataset,
            fid_dataset.keyids,
            gen_samples,
            batch_size=batch_size,
        )

    # ========== L2 Distance Metric Computation ==========
    distance_metric = None
    if compute_distance and fid_dataset is not None and gen_samples_raw:
        logger.info("Computing geometric L2 distance metric...")
        try:
            distance_metric = get_motion_distances(
                model,
                fid_dataset,
                fid_dataset.keyids,
                gen_samples_raw,
                batch_size=batch_size
            )
            logger.info(f"L2 distance computed: {distance_metric:.4f}")
        except Exception as e:
            logger.warning(f"Failed to compute L2 distance: {e}")
            distance_metric = None
    elif not compute_distance:
        logger.info("Skipping geometric L2 distance computation (compute_distance=False)")

    # ========== GT Metrics Computation ==========
    gt_metrs_batches = {}
    gt_metrs_full = {}
    gt_fid_metrics = {}
    gt_distance_metric = None

    if evaluate_gt:
        logger.info("Computing Ground Truth (GT) metrics as upper bound reference...")
        gt_results = {}
        gt_keyids_ord = {}

        # Compute GT similarity matrices
        for protocol in protocols:
            gt_dataset = datasets[protocol]

            if protocol not in gt_results:
                if protocol == "normal":
                    gt_res, gt_keyids_ord_for_all = compute_gt_sim_matrix(
                        model, gt_dataset, gt_dataset.keyids,
                        batch_size=batch_size,
                    )
                    gt_keyids_ord['all'] = gt_keyids_ord_for_all
                    gt_results.update({key: gt_res for key in ["normal"]})

                elif protocol == "batches":
                    keyids = sorted(gt_dataset.keyids)
                    N = len(keyids)
                    idx = np.arange(N)
                    np.random.seed(0)
                    np.random.shuffle(idx)
                    idx_batches = [
                        idx[bs_m2m * i : bs_m2m * (i + 1)] for i in range(len(keyids) // bs_m2m)
                    ]

                    gt_results["batches"] = []
                    gt_keyids_ord["batches"] = []
                    for idx_batch in tqdm(idx_batches, desc="GT batches"):
                        gt_res_matrs, gt_res_keys = compute_gt_sim_matrix(
                            model, gt_dataset,
                            np.array(keyids)[idx_batch],
                            batch_size=batch_size,
                            progress=False
                        )
                        gt_results["batches"].append(gt_res_matrs)
                        gt_keyids_ord["batches"].append(gt_res_keys)

            gt_result = gt_results[protocol]

            # Compute GT metrics
            if protocol == "batches":
                def compute_batches_metrics_gt(sim_matrix_lst):
                    all_metrics = []
                    for sim_matrix in sim_matrix_lst:
                        metrics, _ = all_contrastive_metrics_mot2mot(sim_matrix,
                                                          rounding=None, return_cols=True)
                        all_metrics.append(metrics)

                    avg_metrics = {}
                    for key in all_metrics[0].keys():
                        avg_metrics[key] = round(
                            float(np.mean([metrics[key] for metrics in all_metrics])), 2
                        )
                    return avg_metrics

                gt_metrics_dico = {}
                gt_result_packed_to_d = {key: [d[key] for d in gt_result]
                                      for key in gt_result[0]}
                gt_str_for_tab = ''

                for var, lst_of_sim_matrs in gt_result_packed_to_d.items():
                    metr_name = mat2name[var]
                    gt_metrics_dico[metr_name] = compute_batches_metrics_gt(lst_of_sim_matrs)
                    gt_str_for_tab += print_latex_metrics_m2m(gt_metrics_dico[metr_name])

                gt_line_for_batches = gt_str_for_tab.replace("\\\&", "&")
            else:
                gt_metrics = {}
                gt_str_for_tab = ''
                for var, sim_matrix in gt_result.items():
                    metr_name = mat2name[var]
                    gt_metrics[metr_name], _ = all_contrastive_metrics_mot2mot(
                        sim_matrix, None, threshold=None, return_cols=True
                    )
                    gt_str_for_tab += print_latex_metrics_m2m(gt_metrics[metr_name])
                gt_line_for_all = gt_str_for_tab.replace("\\\&", "&")

        # Parse GT metrics
        gt_dict_batches = line2dict(gt_line_for_batches)
        gt_dict_full = line2dict(gt_line_for_all)
        names_to_keep = ["R@1_s2t", "R@2_s2t", "R@3_s2t", "AvgR_s2t",
                        "R@1", "R@2", "R@3", "AvgR"]
        gt_metrs_full = {f"gt_{key}": gt_dict_full[key] for key in names_to_keep if key in gt_dict_full}
        gt_metrs_batches = {f"gt_{key}": gt_dict_batches[key] for key in names_to_keep if key in gt_dict_batches}

        # Compute GT FID (should be very small for target-to-target)
        gt_fid_dataset = datasets.get("normal", dataset)
        if gt_fid_dataset is not None:
            # Create a fake gen_samples dict where each generated motion is actually the target
            gt_gen_samples = {}
            for keyid in gt_fid_dataset.keyids:
                sample = gt_fid_dataset.load_keyid(keyid)
                gt_gen_samples[keyid] = sample['motion_target']

            gt_fid_metrics_raw = compute_fid_diversity_metrics(
                model,
                gt_fid_dataset,
                gt_fid_dataset.keyids,
                gt_gen_samples,
                batch_size=batch_size,
            )
            # Add gt_ prefix to FID metrics
            gt_fid_metrics = {f"gt_{key}": val for key, val in gt_fid_metrics_raw.items()}


        gt_distance_metric = None

        logger.info("GT metrics computation completed.")

    return metrs_batches, metrs_full, fid_metrics, distance_metric, gt_metrs_batches, gt_metrs_full, gt_fid_metrics, gt_distance_metric

if __name__ == "__main__":
    retrieval()
