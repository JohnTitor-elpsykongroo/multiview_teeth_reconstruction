"""Coupled image residual and derivatives of latent and shared arch pose."""
from __future__ import annotations
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from shape_only_common import SurfaceImageObjective, project_and_jacobian


def pose_matrix(pose):
    result=np.eye(4)
    result[:3,:3]=Rotation.from_rotvec(pose[:3]).as_matrix()
    result[:3,3]=pose[3:]
    return result


def skew(points):
    points=np.asarray(points)
    result=np.zeros(points.shape[:-1]+(3,3))
    result[...,0,1]=-points[...,2]; result[...,0,2]=points[...,1]
    result[...,1,0]=points[...,2]; result[...,1,2]=-points[...,0]
    result[...,2,0]=-points[...,1]; result[...,2,1]=points[...,0]
    return result


def right_jacobian(rotvec):
    theta=np.linalg.norm(rotvec)
    A=skew(rotvec)
    if theta<1e-5:
        return np.eye(3)-.5*A+(1/6)*(A@A)
    return np.eye(3)-(1-np.cos(theta))/theta**2*A+(theta-np.sin(theta))/theta**3*(A@A)


class JointImageObjective:
    def __init__(self,net,labels,initial,scales,pairs,cameras,initial_pose,pose_scale,x0,damping):
        self.net,self.labels,self.initial,self.scales=net,labels,initial,scales
        self.cameras,self.initial_pose,self.pose_scale=cameras,initial_pose,pose_scale
        self.damping=damping
        self.slices={}
        self.surfaces={}
        start=6
        pose=pose_matrix(self.pose(x0))
        for label in labels:
            key=f'label_{label}'
            self.slices[label]=slice(start,start+len(initial[key]))
            start+=len(initial[key])
            q=x0[self.slices[label]]
            self.surfaces[label]=SurfaceImageObjective(net,label,initial[key],scales[key],q,pairs[label],cameras,pose,damping)
        self.dimensions=start
        self.image_rows=sum(2*len(v.target) for v in self.surfaces.values())
        self.cached_x=None

    def pose(self,x):
        return self.initial_pose+x[:6]*self.pose_scale

    def forward(self,x):
        if self.cached_x is not None and np.array_equal(x,self.cached_x):
            return self.cached
        matrix=pose_matrix(self.pose(x))
        blocks=[]
        geometry={}
        for label in self.labels:
            obj=self.surfaces[label]
            q=x[self.slices[label]]
            _,roots,grad,_=obj.forward(q)
            pixels,J=project_and_jacobian(roots,self.cameras,obj.indices,matrix)
            blocks.append((pixels-obj.target).ravel())
            geometry[label]=(roots,grad,J)
        residual=np.r_[np.concatenate(blocks),np.sqrt(self.damping)*x[6:]]
        self.cached_x=x.copy()
        self.cached=(residual,geometry,matrix)
        return self.cached

    def residual(self,x):
        return self.forward(x)[0]

    def jacobian(self,x):
        _,geometry,matrix=self.forward(x)
        Jr=right_jacobian(self.pose(x)[:3])
        J=np.zeros((self.image_rows+self.dimensions-6,self.dimensions))
        row=0
        for label in self.labels:
            obj=self.surfaces[label]
            roots,grad,pixel_jac=geometry[label]
            end=row+2*len(roots)
            rotation_jac=-np.einsum('nij,njk->nik',pixel_jac,skew(roots))@Jr
            translation_jac=pixel_jac@matrix[:3,:3].T
            J[row:end,:6]=(np.concatenate([rotation_jac,translation_jac],axis=2)*self.pose_scale).reshape(-1,6)
            q=x[self.slices[label]]
            code0=torch.tensor(obj.code0,dtype=torch.float64)
            scale=torch.tensor(obj.scale,dtype=torch.float64)
            def field(q_tensor):
                return self.net.inference({label:(code0+q_tensor*scale)[None]},torch.tensor(roots,dtype=torch.float64)).flatten()
            phi_q=torch.autograd.functional.jacobian(field,torch.tensor(q,dtype=torch.float64),
                vectorize=True,strategy='forward-mode').detach().numpy()
            denominator=np.sum(grad*obj.normal,axis=1)
            pixel_normal=np.einsum('nij,nj->ni',pixel_jac,obj.normal)
            code_jac=-pixel_normal[:,:,None]*phi_q[:,None,:]/denominator[:,None,None]
            J[row:end,self.slices[label]]=code_jac.reshape(-1,len(q))
            row=end
        J[self.image_rows:,6:]=np.sqrt(self.damping)*np.eye(self.dimensions-6)
        return J


def gradient_check(objective,x):
    rng=np.random.default_rng(917)
    J=objective.jacobian(x)
    residual=objective.residual(x)
    if not np.isfinite(J).all():
        raise ValueError('nonfinite Joint derivative')
    directions=[]
    for index in range(6):
        direction=np.zeros(len(x)); direction[index]=1
        directions.append((f'pose_axis_{index}',direction))
    for label in objective.labels:
        for index in range(3):
            direction=np.zeros(len(x))
            direction[objective.slices[label]]=rng.normal(size=objective.slices[label].stop-objective.slices[label].start)
            direction/=np.linalg.norm(direction)
            directions.append((f'latent_{label}_{index}',direction))
    for index in range(3):
        direction=rng.normal(size=len(x)); direction/=np.linalg.norm(direction)
        directions.append((f'mixed_{index}',direction))
    records=[]
    for name,direction in directions:
        analytic=J@direction
        analytic_loss=float(2*residual@analytic/len(residual))
        for epsilon in [1e-3,1e-4]:
            plus=objective.residual(x+epsilon*direction)
            minus=objective.residual(x-epsilon*direction)
            finite=(plus-minus)/(2*epsilon)
            relative=float(np.linalg.norm(analytic-finite)/max(np.linalg.norm(analytic),np.linalg.norm(finite),1e-10))
            records.append({'direction':name,'epsilon_normalized':epsilon,'relative_error':relative,
                'analytic_loss_derivative':analytic_loss,'finite_difference_loss_derivative':float((np.mean(plus**2)-np.mean(minus**2))/(2*epsilon))})
    return {'max_relative_error':max(r['relative_error'] for r in records),'directions':records,
        'objective':'joint projected contour residual through implicit surface roots and shared SO3 rotation/translation',
        'checked_variables':['six pose axes','per-tooth latent directions','mixed pose and latent directions'],
        'visibility_and_correspondences':'fixed within derivative check; refreshed in outer iterations',
        'hard_raster_iou_is_differentiated':False}
