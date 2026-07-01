#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
from kornia import create_meshgrid
import torch
from utils.depth_utils import estimate_depth

import sys

from torchcubicspline import (natural_cubic_spline_coeffs, NaturalCubicSpline)
import splines.quaternion
import torch.nn.functional as F

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    # from pytorch3d
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    # from pytorch3d
    """
    Convert rotations given as rotation matrices to quaternions.
    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    # from pytorch3d
    """
    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def pix2ndc(v, S):
    return (v * 2.0 + 1.0) / S - 1.0

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    depth = estimate_depth(gt_image.cuda()).cpu().numpy() ### midas
    # depth = depth_anything(gt_image.cuda(), 'vits', model = model).cpu().numpy()
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device, depth_image=depth)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    # encoder = 'vits'
    # DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    # DepthAnything_model = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(encoder)).to(DEVICE).eval()

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry

# def loadCam(args, id, cam_info, resolution_scale):
#     orig_w, orig_h = cam_info.image.size

#     if args.resolution in [1, 2, 4, 8]:
#         resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
#     else:  # should be a type that converts to float
#         if args.resolution == -1:
#             if orig_w > 1600:
#                 global WARNED
#                 if not WARNED:
#                     print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
#                         "If this is not desired, please explicitly specify '--resolution/-r' as 1")
#                     WARNED = True
#                 global_down = orig_w / 1600
#             else:
#                 global_down = 1
#         else:
#             global_down = orig_w / args.resolution

#         scale = float(global_down) * float(resolution_scale)
#         resolution = (int(orig_w / scale), int(orig_h / scale))

#     resized_image_rgb = PILtoTorch(cam_info.image, resolution)

#     gt_image = resized_image_rgb[:3, ...]
#     loaded_mask = None

#     if resized_image_rgb.shape[1] == 4:
#         loaded_mask = resized_image_rgb[3:4, ...]

#     return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
#                   FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
#                   image=gt_image, gt_alpha_mask=loaded_mask,
#                   image_name=cam_info.image_name, uid=id, data_device=args.data_device, norm_K=cam_info.norm_K)

# def cameraList_from_camInfos(cam_infos, resolution_scale, args):
#     camera_list = []

#     for id, c in enumerate(cam_infos):
#         camera_list.append(loadCam(args, id, c, resolution_scale))

#     return camera_list

# def camera_to_JSON(id, camera : Camera):
#     Rt = np.zeros((4, 4))
#     Rt[:3, :3] = camera.R.transpose()
#     Rt[:3, 3] = camera.T
#     Rt[3, 3] = 1.0

#     W2C = np.linalg.inv(Rt)
#     pos = W2C[:3, 3]
#     rot = W2C[:3, :3]
#     serializable_array_2d = [x.tolist() for x in rot]
#     camera_entry = {
#         'id' : id,
#         'img_name' : camera.image_name,
#         'width' : camera.width,
#         'height' : camera.height,
#         'position': pos.tolist(),
#         'rotation': serializable_array_2d,
#         'fy' : fov2focal(camera.FovY, camera.height),
#         'fx' : fov2focal(camera.FovX, camera.width)
#     }
#     return camera_entry


def set_rays_od(cams):
    for id, cam in enumerate(cams):
        rayd=1
        if rayd is not None:
            projectinverse = cam.projection_matrix.T.inverse()
            camera2wold = cam.world_view_transform.T.inverse()
            pixgrid = create_meshgrid(cam.image_height, cam.image_width, normalized_coordinates=False, device="cpu")[0]
            pixgrid = pixgrid.cuda()  # H,W,
            xindx = pixgrid[:,:,0] # x
            yindx = pixgrid[:,:,1] # y
            ndcy, ndcx = pix2ndc(yindx, cam.image_height), pix2ndc(xindx, cam.image_width)
            ndcx = ndcx.unsqueeze(-1)
            ndcy = ndcy.unsqueeze(-1)# * (-1.0)
            ndccamera = torch.cat((ndcx, ndcy,   torch.ones_like(ndcy) * (1.0) , torch.ones_like(ndcy)), 2) # N,4
            projected = ndccamera @ projectinverse.T
            diretioninlocal = projected / projected[:,:,3:] #v
            direction = diretioninlocal[:,:,:3] @ camera2wold[:3,:3].T
            # rays_d = torch.nn.functional.normalize(direction, p=2.0, dim=-1)
            rays_d = direction
            rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
            cam.rayo = cam.camera_center.expand(rays_d.shape).permute(2, 0, 1).unsqueeze(0).cpu()
            cam.rayd = rays_d.permute(2, 0, 1).unsqueeze(0).cpu()
        else :
            cam.rayo = None
            cam.rayd = None

def set_rays(scene,resolution_scales=[1.0]):
    set_rays_od(scene.getTrainCameras())
    for resolution_scale in resolution_scales:
        for cam in scene.train_cameras[resolution_scale]:
            if cam.rayo is not None:
                cam.rays = torch.cat([cam.rayo, cam.rayd], dim=1)

def render_time_interp(all_poses,wobble=False):
    pos_spline_idxs=torch.linspace(0,all_poses.size(0)-1,15 if 1 else 40)
    rot_spline_idxs=torch.linspace(0,all_poses.size(0)-1,15 if 1 else 40)
    #pos_spline_idxs=torch.arange(all_poses.size(0)).float()
    #rot_spline_idxs=torch.arange(all_poses.size(0)).float()

    all_pos_splines=[]
    all_quat_splines=[]

    all_pos_spline=[]
    all_quat_spline=[]

    all_pos_spline.append(NaturalCubicSpline(natural_cubic_spline_coeffs(pos_spline_idxs, all_poses[pos_spline_idxs.long(),:3,-1].cpu())))
    quats = matrix_to_quaternion(all_poses[:,:3,:3])
    all_quat_spline.append(splines.quaternion.PiecewiseSlerp([splines.quaternion.UnitQuaternion.from_unit_xyzw(quat_) 
                                    for quat_ in quats[rot_spline_idxs.long()].detach().cpu().numpy()],grid=rot_spline_idxs.detach().tolist()))
    all_pos_splines.append(all_pos_spline)
    all_quat_splines.append(all_quat_spline)

    n=300
    thetas=np.linspace(0,np.pi*10*len(all_poses)/60,n)

    query_poses=[]
    for t_i,t in enumerate(torch.linspace(0,all_poses.size(0)-1,n)):
        print(t)

        pos_splines=all_pos_splines[0]
        quat_splines_=all_quat_splines[0]
        pos_spline=pos_splines[0]
        quat_spline_=quat_splines_[0]
        custom_pose=torch.eye(4).cuda()
        custom_pose[:3,-1]=pos_spline.evaluate(t)
        scale = .015 #.3 * model.far/30
        if wobble:
            custom_pose[0,-1]+=np.cos(thetas[t_i]) * scale
            custom_pose[1,-1]+=np.sin(thetas[t_i]) * scale
        quat_eval=quat_spline_.evaluate(t.item())
        curr_quats = torch.tensor(list(quat_eval.vector)+[quat_eval.scalar])
        custom_pose[:3,:3] = quaternion_to_matrix(curr_quats)
        query_poses.append(custom_pose)
    return torch.stack(query_poses)


def render_time_interp(all_poses,wobble=False):
    pos_spline_idxs=torch.linspace(0,all_poses.size(0)-1,15 if 1 else 40)
    rot_spline_idxs=torch.linspace(0,all_poses.size(0)-1,15 if 1 else 40)
    #pos_spline_idxs=torch.arange(all_poses.size(0)).float()
    #rot_spline_idxs=torch.arange(all_poses.size(0)).float()

    all_pos_splines=[]
    all_quat_splines=[]

    all_pos_spline=[]
    all_quat_spline=[]

    all_pos_spline.append(NaturalCubicSpline(natural_cubic_spline_coeffs(pos_spline_idxs, all_poses[pos_spline_idxs.long(),:3,-1].cpu())))
    quats = matrix_to_quaternion(all_poses[:,:3,:3])
    all_quat_spline.append(splines.quaternion.PiecewiseSlerp([splines.quaternion.UnitQuaternion.from_unit_xyzw(quat_) 
                                    for quat_ in quats[rot_spline_idxs.long()].detach().cpu().numpy()],grid=rot_spline_idxs.detach().tolist()))
    all_pos_splines.append(all_pos_spline)
    all_quat_splines.append(all_quat_spline)

    n=300
    thetas=np.linspace(0,np.pi*10*len(all_poses)/60,n)

    query_poses=[]
    for t_i,t in enumerate(torch.linspace(0,all_poses.size(0)-1,n)):
        print(t)

        pos_splines=all_pos_splines[0]
        quat_splines_=all_quat_splines[0]
        pos_spline=pos_splines[0]
        quat_spline_=quat_splines_[0]
        custom_pose=torch.eye(4).cuda()
        custom_pose[:3,-1]=pos_spline.evaluate(t)
        scale = .015 #.3 * model.far/30
        if wobble:
            custom_pose[0,-1]+=np.cos(thetas[t_i]) * scale
            custom_pose[1,-1]+=np.sin(thetas[t_i]) * scale
        quat_eval=quat_spline_.evaluate(t.item())
        curr_quats = torch.tensor(list(quat_eval.vector)+[quat_eval.scalar])
        custom_pose[:3,:3] = quaternion_to_matrix(curr_quats)
        query_poses.append(custom_pose)
    return torch.stack(query_poses)


###
import torch
import numpy as np
from trimesh.creation import icosphere as IcoSphere

from dataclasses import dataclass


@dataclass
class Rays:
    o: torch.Tensor  # [..., 3]
    d: torch.Tensor  # [..., 3]

    def __len__(self):
        return len(self.o)
    def __getitem__(self, indices):
        return Rays(self.o[indices], self.d[indices])

    def collapse(self):
        return self.o, self.d

@dataclass
class BoundedRays:
    o: torch.Tensor     # [..., 3]
    d: torch.Tensor     # [..., 3]
    near: torch.Tensor  # [..., 1]
    far: torch.Tensor   # [..., 1]

    def __len__(self):
        return len(self.o)
    def __getitem__(self, indices):
        return BoundedRays(self.o[indices], self.d[indices], self.near[indices], self.far[indices])

    def collapse(self):
        return self.o, self.d, self.near, self.far


def cat_rays(rays):
    rays_o = torch.cat([_.o for _ in rays], dim=0)
    rays_d = torch.cat([_.d for _ in rays], dim=0)
    return Rays(rays_o, rays_d)


def apply_rot(pts, rot_mat):
    assert rot_mat.shape == (3, 3)
    return torch.matmul(rot_mat, pts[..., None])[..., 0]


def apply_rot_trans(pts, rot_mat, pos):
    assert rot_mat.shape == (3, 3)
    assert pos.shape == (3,)
    return torch.matmul(rot_mat, pts[..., None])[..., 0] + pos


def apply_transform(pts, pose):
    return apply_rot_trans(pts, pose[:3, :3], pose[:3, 3])


# Camera rays, OpenCV style
def cam_rays_cam_space(height: int, width=-1, fovy=np.deg2rad(90.), aspect_ratio=1.):
    '''
    OpenCV style!
    :param height:
    :param width:
    :param fovy:
    :param aspect_ratio:
    :return: Tensor with shape [height, width, 3]
    '''
    if width < 0:
        width = int(np.round(height * aspect_ratio))
    else:
        aspect_ratio = width / height

    span_y = np.tan(fovy * .5)
    span_x = span_y * aspect_ratio
    y = torch.linspace(-span_y, span_y, height)
    x = torch.linspace(-span_x, span_x, width)
    y, x = torch.meshgrid(y, x, indexing='ij')
    xyz = torch.stack([x, y, torch.ones_like(x)], -1)
    return xyz / torch.linalg.norm(xyz, 2, -1, True)


def look_at(to_vec, up_vec=None):
    '''
    :param to_vec: [n, 3]
    :param up_vec: [n, 3]
    :return: rotation matrices [n, 3, 3]
    '''
    n = to_vec.shape[0]
    if up_vec is None:
        up_vec = torch.cat([torch.zeros([n, 2]), torch.ones([n, 1])], -1)
    down_vec = -up_vec
    to_vec = to_vec / torch.linalg.norm(to_vec, 2, -1, True)
    ri_vec = torch.linalg.cross(down_vec, to_vec)
    ri_vec = ri_vec / torch.linalg.norm(ri_vec, 2, -1, True)
    down_vec = torch.linalg.cross(to_vec, ri_vec)
    c2w = torch.stack([ri_vec, down_vec, to_vec], 2)
    return c2w

def ang2vec(angles):
    '''
    :param angles: [n, 2]
    :return: [n, 3]
    '''
    ang_x, ang_y = angles[..., 0], angles[..., 1]
    vecs = torch.stack([torch.cos(ang_x) * torch.cos(ang_y),
                        torch.sin(ang_x) * torch.cos(ang_y),
                        torch.sin(ang_y)], dim=-1)

    return vecs


def img_coord_from_hw(h, w):
    i = torch.linspace(.5 / h, 1. - .5 / h, h)
    j = torch.linspace(.5 / w, 1. - .5 / w, w)
    ii, jj = torch.meshgrid(i, j, indexing='ij')
    return torch.stack([ii, jj], -1)


def img_to_pano_coord(coords):
    '''
    :param coords: [n, 2] range of [0, 1]. (row coord, col coord)
    :return: pano coords
    '''
    y, x = coords[..., 0], coords[..., 1]
    return torch.stack([-(y - .5) * np.pi, -(x - .5) * 2. * np.pi], -1)


def pano_to_img_coord(coords):
    y, x = coords[..., 0], coords[..., 1]
    return torch.stack([-y / np.pi + .5, -(x / (2. * np.pi)) + .5], -1)


def direction_to_pano_coord(dirs):
    dirs = dirs / torch.linalg.norm(dirs, 2, -1, True)
    beta = torch.arcsin(dirs[..., 2])
    xy = dirs[..., :2] / torch.cos(beta)[..., None]
    alpha = torch.view_as_complex(xy).angle()   # [-np.pi., np.pi]
    return torch.stack([beta, alpha], -1)


def pano_coord_to_direction(coords):
    beta, alpha = coords[..., 0], coords[..., 1]
    dirs = torch.stack([torch.cos(alpha) * torch.cos(beta),
                        torch.sin(alpha) * torch.cos(beta),
                        torch.sin(beta)], dim=-1)
    return dirs


def direction_to_img_coord(dirs):
    return pano_to_img_coord(direction_to_pano_coord(dirs))


def img_coord_to_pano_direction(coords):
    return pano_coord_to_direction(img_to_pano_coord(coords))

@torch.no_grad()
def direction_to_pers_img_coord(dirs, to_vec, down_vec, right_vec):
    eps = 1e-5
    dirs = dirs / torch.linalg.norm(dirs, 2, -1, True)
    to_vec_len = torch.linalg.norm(to_vec, 2, -1).item()
    to_vec = to_vec / to_vec_len
    down_vec = down_vec / to_vec_len
    right_vec = right_vec / to_vec_len
    down_vec_len = torch.linalg.norm(down_vec, 2, -1).item()
    right_vec_len = torch.linalg.norm(right_vec, 2, -1).item()

    project_len = (dirs * to_vec).sum(-1, True)
    mask = project_len > eps
    project_len = project_len.clip(eps, None)
    dirs = dirs / project_len

    i = ((dirs - to_vec) * down_vec).sum(-1, True) / down_vec_len**2
    j = ((dirs - to_vec) * right_vec).sum(-1, True) / right_vec_len**2
    mask = (mask & (i.abs() <= 1.) & (j.abs() <= 1.)).float()
    ij = (torch.cat([i, j], dim=-1) + 1.) * .5
    return ij, mask


def img_coord_to_sample_coord(coords):
    return torch.stack([coords[..., 1], coords[..., 0]], -1) * 2. - 1.


def get_rand_horizontal_points(batch_size, dim=3):
    rs = torch.sqrt(torch.rand(batch_size))
    theta = (torch.rand(batch_size) * 2. - 1.) * np.pi
    pos = [rs * torch.cos(theta), rs * torch.sin(theta)]
    if dim == 3:
        pos += [ torch.zeros([batch_size]) ]

    return torch.stack(pos, -1)

def get_panorama_sphere_points(h, w):
    img_coords = img_coord_from_hw(h, w)
    pts = img_coord_to_pano_direction(img_coords)
    pts = pts / torch.linalg.norm(pts, 2, -1, True)
    return pts

def pers_depth_to_normal(depth, down_len, right_len):
    assert depth.min().item() > 1e-5
    if len(depth.shape) == 2:
        depth = depth[..., None]
    h, w, _ = depth.shape
    ii, jj = torch.meshgrid(
        torch.linspace(.5 / h, 1. - .5 / h, h),
        torch.linspace(.5 / w, 1. - .5 / w, w),
        indexing='ij'
    )
    z = torch.ones_like(ii)
    x = (jj * 2. - 1.) * right_len
    y = (ii * 2. - 1.) * down_len
    pts = torch.stack([x, y, z], dim=-1)
    pts = pts * depth
    right_vec = pts[:-1, 1:] - pts[:-1, :-1]
    down_vec  = pts[1:, :-1] - pts[:-1, :-1]
    # right_vec_len = torch.linalg.norm(right_vec, 2, -1, True)
    # down_vec_len = torch.linalg.norm(down_vec, 2, -1, True)
    right_vec = right_vec / torch.linalg.norm(right_vec, 2, -1, True).detach()
    down_vec = down_vec / torch.linalg.norm(down_vec, 2, -1, True).detach()
    to_vec = torch.cross(right_vec, down_vec)
    # to_vec_len = torch.linalg.norm(to_vec, 2, -1, True)
    to_vec = to_vec / torch.linalg.norm(to_vec, 2, -1, True).detach()
    assert not torch.any(torch.isnan(to_vec))
    return -to_vec


# -----------

def gen_pano_rays(pose, height=512, width=1024):
    img_coord = img_coord_from_hw(height, width)
    rays_d = img_coord_to_pano_direction(img_coord)
    rays_d = apply_rot(rays_d, pose[:3, :3])
    rays_o = pose[None, None, :3, 3].repeat(height, width, 1)
    return Rays(rays_o, rays_d)


def gen_pers_rays(pose, fov, res):
    rays_d = cam_rays_cam_space(height=res, width=res, fovy=fov)
    rays_o = torch.zeros_like(rays_d) + pose[:3, 3][None, None, :]
    rays_d = apply_rot(rays_d, pose[:3, :3])
    return Rays(rays_o, rays_d)

