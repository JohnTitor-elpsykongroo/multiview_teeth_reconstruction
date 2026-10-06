"""Frozen DMM decoding and implicit surface/image derivative utilities.

The fitter uses no source-case geometry, depth, or active ground-truth latent.
Contour correspondences and visibility are refreshed between local solves.
"""

from __future__ import annotations
import contextlib
import io
import json
import sys
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes
from run_forward_check import rasterize, sha256
from run_pose_only import boundary_pixels, sample_rows, mean_iou


def load_model(config, manifest):
    exp = Path(config['experiment'])
    model_path = exp/'ModelParameters'/f"dmm_{config['checkpoint']}.pth"
    specs_path = exp/'specs.json'
    if sha256(model_path) != manifest['model_sha256'] or sha256(specs_path) != manifest['specs_sha256']:
        raise ValueError('model or specs changed since preparation')
    dmm = Path(config['dmm_root'])
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(dmm), str(dmm/'third_party')]
    from networks.dmm_net import DMM
    with contextlib.redirect_stdout(io.StringIO()):
        net = DMM(json.loads(specs_path.read_text()))
    state = torch.load(model_path, map_location='cpu', weights_only=True)
    if state['epoch'] != manifest['checkpoint_epoch']:
        raise ValueError('checkpoint epoch mismatch')
    net.load_state_dict(state['model_state_dict'], strict=True)
    net.eval().double()
    for param in net.parameters():
        param.requires_grad_(False)
    torch.set_num_threads(int(config['cpu_threads']))
    return net


def decode(net, label, code, box, query_batch=8192):
    n = int(box['grid_n'])
    origin, top = np.array(box['grid_origin']), np.array(box['grid_top'])
    axes = [np.linspace(origin[i], top[i], n) for i in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
    sdf = []
    with torch.no_grad():
        c = torch.tensor(code[None], dtype=torch.float64)
        for start in range(0, len(grid), query_batch):
            sdf.append(net.inference({label: c}, torch.tensor(grid[start:start+query_batch])).flatten().numpy())
    cube = np.concatenate(sdf).reshape((n,)*3)
    if cube.min() >= 0 or cube.max() <= 0:
        raise ValueError(f'no surface for tooth {label}')
    boundary = np.concatenate([cube[0].ravel(), cube[-1].ravel(), cube[:,0].ravel(),
                               cube[:,-1].ravel(), cube[:,:,0].ravel(), cube[:,:,-1].ravel()])
    if np.count_nonzero(boundary <= 0):
        raise ValueError(f'tooth {label} surface crosses sampling box')
    xyz, faces, _, _ = marching_cubes(cube, 0, spacing=tuple((top-origin)/(n-1)))
    return (xyz+origin).astype(np.float32), faces.astype(np.int32)


def world_meshes(meshes, pose):
    pose = np.asarray(pose)
    return {k: (xyz @ pose[:3,:3].T + pose[:3,3], faces) for k,(xyz,faces) in meshes.items()}


def fixed_render_cache(meshes, cameras, pose):
    return {cam['name']: rasterize(world_meshes(meshes, pose), cam, include_gum=False, shade=False)
            for cam in cameras}


def render_scene(active_meshes, cache, cameras, pose, with_points=False):
    masks, maps = {}, {}
    moved = world_meshes(active_meshes, pose)
    for camera in cameras:
        name = camera['name']
        output = rasterize(moved, camera, include_gum=False, shade=False, return_world_points=with_points)
        background = cache[name]
        visible = output[1] < background[1]
        labels = background[0].copy()
        labels[visible] = output[0][visible]
        masks[name] = labels
        if with_points:
            points = output[4]
            points[~visible] = np.nan
            maps[name] = points
    return masks, maps


def make_pairs(masks, point_maps, targets, cameras, labels, pose, maximum):
    rotation, translation = np.asarray(pose)[:3,:3], np.asarray(pose)[:3,3]
    pairs = {}
    for label in labels:
        points, pixels, indices = [], [], []
        for ci, camera in enumerate(cameras):
            name = camera['name']
            pred = boundary_pixels(masks[name], label)
            target = boundary_pixels(targets[name], label)
            if len(pred) < 10 or len(target) < 10:
                continue
            for queries, reference, reverse in [(sample_rows(pred,maximum),target,False),
                                                 (sample_rows(target,maximum),pred,True)]:
                distances, nearest = cKDTree(reference).query(queries)
                for query, ni, distance in zip(queries,nearest,distances):
                    if distance > 30:
                        continue
                    predicted = reference[ni] if reverse else query
                    observed = query if reverse else reference[ni]
                    x,y = predicted
                    world = point_maps[name][y,x]
                    if np.isfinite(world).all():
                        points.append((world-translation) @ rotation)
                        pixels.append(observed.astype(float)+0.5)
                        indices.append(ci)
        if len(points) < 100:
            raise ValueError(f'insufficient contour points for tooth={label}')
        pairs[label] = (np.array(points), np.array(pixels), np.array(indices))
    return pairs


def sdf_and_gradient(net, label, code, xyz):
    x = torch.tensor(xyz, dtype=torch.float64, requires_grad=True)
    c = torch.tensor(code[None], dtype=torch.float64)
    phi = net.inference({label:c}, x).flatten()
    grad = torch.autograd.grad(phi.sum(),x)[0]
    return phi.detach().numpy(), grad.detach().numpy()


def project_and_jacobian(points, cameras, indices, pose):
    pixels = np.empty((len(points),2))
    jac = np.empty((len(points),2,3))
    pose = np.asarray(pose)
    for ci,camera in enumerate(cameras):
        selected = indices == ci
        R = np.asarray(camera['R_world_to_camera']) @ pose[:3,:3]
        t = np.asarray(camera['R_world_to_camera']) @ pose[:3,3] + np.asarray(camera['t_world_to_camera'])
        K = np.asarray(camera['K'])
        cam = points[selected] @ R.T + t
        x,y,z = cam.T
        if np.any(z<=0):
            raise ValueError('surface behind camera')
        pixels[selected] = np.column_stack((K[0,0]*x/z+K[0,2],K[1,1]*y/z+K[1,2]))
        J = np.zeros((len(z),2,3))
        J[:,0,0] = K[0,0]/z
        J[:,1,1] = K[1,1]/z
        J[:,0,2] = -K[0,0]*x/z**2
        J[:,1,2] = -K[1,1]*y/z**2
        jac[selected] = J @ R
    return pixels,jac


class SurfaceImageObjective:
    """Local exact isosurface roots along frozen normals, projected to image.

    For phi(x,z)=0 and x=base+s*n, ds/dq=-phi_q/(grad(phi).n).
    Forward root solves and autograd phi_q form a consistent local derivative.
    Visibility/nearest boundary assignments remain fixed within one solve.
    """
    def __init__(self, net, label, code0, scale, q0, pairs, cameras, pose, damping):
        self.net,self.label,self.code0,self.scale = net,label,code0,scale
        self.points,self.target,self.indices = pairs
        self.cameras,self.pose,self.damping = cameras,pose,damping
        code = code0 + q0*scale
        points = self.points.copy()
        for _ in range(4):
            phi,grad = sdf_and_gradient(net,label,code,points)
            points -= phi[:,None]*grad/np.maximum(np.sum(grad**2,1)[:,None],1e-10)
        _,grad = sdf_and_gradient(net,label,code,points)
        self.base = points
        self.normal = grad/np.maximum(np.linalg.norm(grad,axis=1)[:,None],1e-10)
        self.cached_q = None

    def forward(self,q):
        if self.cached_q is not None and np.array_equal(q,self.cached_q):
            return self.cached
        code = self.code0 + q*self.scale
        roots = self.base.copy()
        for _ in range(6):
            phi,grad = sdf_and_gradient(self.net,self.label,code,roots)
            denominator = np.sum(grad*self.normal,1)
            if np.any(np.abs(denominator)<1e-6):
                raise ValueError('singular implicit surface derivative')
            step = phi/denominator
            roots -= np.clip(step,-0.03,0.03)[:,None]*self.normal
            if np.max(np.abs(phi)) < 1e-10:
                break
        phi,grad = sdf_and_gradient(self.net,self.label,code,roots)
        if np.max(np.abs(phi)) > 1e-6:
            raise ValueError('local surface root failed to converge')
        pixels,pixel_jac = project_and_jacobian(roots,self.cameras,self.indices,self.pose)
        residual = np.r_[(pixels-self.target).ravel(), np.sqrt(self.damping)*q]
        self.cached_q = q.copy()
        self.cached = residual,roots,grad,pixel_jac
        return self.cached

    def residual(self,q):
        return self.forward(q)[0]

    def jacobian(self,q):
        _,roots,grad,pixel_jac = self.forward(q)
        code0 = torch.tensor(self.code0,dtype=torch.float64)
        scale = torch.tensor(self.scale,dtype=torch.float64)
        def field(q_tensor):
            return self.net.inference({self.label:(code0+q_tensor*scale)[None]},
                                       torch.tensor(roots,dtype=torch.float64)).flatten()
        phi_q = torch.autograd.functional.jacobian(field, torch.tensor(q,dtype=torch.float64),
                                                    vectorize=True,strategy='forward-mode').detach().numpy()
        denominator = np.sum(grad*self.normal,1)
        pixel_normal = np.einsum('nij,nj->ni',pixel_jac,self.normal)
        J = -pixel_normal[:,:,None]*phi_q[:,None,:]/denominator[:,None,None]
        return np.vstack([J.reshape(-1,len(q)), np.sqrt(self.damping)*np.eye(len(q))])


def gradient_check(objectives, seed=517):
    rng = np.random.default_rng(seed)
    records = []
    for label,objective in objectives.items():
        q = np.zeros_like(objective.code0)
        residual = objective.residual(q)
        jac = objective.jacobian(q)
        gradient = 2*jac.T@residual/len(residual)
        if not np.isfinite(jac).all() or np.linalg.norm(gradient)<1e-10:
            raise ValueError('image-to-latent derivative is nonfinite or zero')
        for direction_index in range(4):
            direction = gradient/np.linalg.norm(gradient) if direction_index==0 else rng.normal(size=len(q))
            direction /= np.linalg.norm(direction)
            analytic = float(gradient@direction)
            for epsilon in [1e-3,1e-4]:
                plus = objective.residual(q+epsilon*direction)
                minus = objective.residual(q-epsilon*direction)
                finite = float((np.mean(plus**2)-np.mean(minus**2))/(2*epsilon))
                records.append({'label':label,'direction':direction_index,'epsilon_training_std':epsilon,
                                'analytic_image_loss_derivative':analytic,'finite_difference_derivative':finite,
                                'relative_error':abs(analytic-finite)/max(abs(analytic),abs(finite),1e-8)})
    return {'objective':'mean squared projected contour distance through implicit zero-surface roots',
            'visibility_and_correspondences':'fixed inside each local derivative check; refreshed between outer iterations',
            'hard_raster_iou_is_differentiated':False,
            'max_relative_error':max(v['relative_error'] for v in records),'directions':records}
