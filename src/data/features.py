from src.tools.transforms3d import change_for, transform_body_pose
from src.utils.genutils import to_tensor
import torch

def _get_body_pose(data):

    # default is axis-angle representation: Frames x (Jx3) (J=21)
    if not torch.is_tensor(data):
        pose = to_tensor(data['rots'][..., 3:3 + 21*3])  # drop pelvis orientation
    else:
        pose = to_tensor(data[..., 3:3 + 21*3])

    pose = transform_body_pose(pose, f"aa->6d")
    return pose

def _get_body_transl(data):

    if not torch.is_tensor(data):
        tran = data['trans']
    else:
        tran = data
    return to_tensor(tran)

def _get_body_orient(data):

    # default is axis-angle representation
    if not torch.is_tensor(data):
        pelvis_orient = to_tensor(data['rots'][..., :3])
    else:
        pelvis_orient = data
    # axis-angle to rotation matrix & drop last row
    pelvis_orient = transform_body_pose(pelvis_orient, "aa->6d")
    return pelvis_orient

def _get_body_transl_delta_pelv(data):

    trans = to_tensor(data['trans'])
    trans_vel = trans - trans.roll(1, 0)  # shift one right and subtract
    pelvis_orient = transform_body_pose(to_tensor(data['rots'][..., :3]), "aa->rot")
    trans_vel_pelv = change_for(trans_vel, pelvis_orient.roll(1, 0))
    trans_vel_pelv[0] = 0  # zero out velocity of first frame
    return trans_vel_pelv

def _get_body_transl_delta_pelv_infer(pelvis_orient, trans):

    trans = to_tensor(trans)
    trans_vel = trans - trans.roll(1, 0)  # shift one right and subtract
    pelvis_orient = transform_body_pose(to_tensor(pelvis_orient), "6d->rot")
    trans_vel_pelv = change_for(trans_vel, pelvis_orient.roll(1, 0))
    trans_vel_pelv[0] = 0  # zero out velocity of first frame
    return trans_vel_pelv
