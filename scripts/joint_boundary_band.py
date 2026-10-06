"""C1 radial boundary tolerance and exact derivatives; fixed input uncertainty."""
from __future__ import annotations
import numpy as np
from joint_common import JointImageObjective


def boundary_band(vectors,radius,transition):
    if radius==0:
        return vectors.copy(),np.broadcast_to(np.eye(2),(len(vectors),2,2)).copy()
    if radius<0 or transition<=0:
        raise ValueError('invalid boundary band')
    length=np.linalg.norm(vectors,axis=1)
    safe=np.maximum(length,1e-12)
    offset=length-radius
    value=np.where(offset<=0,0,np.where(offset<transition,offset**2/(2*transition),offset-transition/2))
    derivative=np.where(offset<=0,0,np.where(offset<transition,offset/transition,1))
    factor=value/safe
    J=factor[:,None,None]*np.eye(2)+(derivative/safe**2-value/safe**3)[:,None,None]*(vectors[:,:,None]*vectors[:,None,:])
    return vectors*factor[:,None],J


class BandJointObjective(JointImageObjective):
    def __init__(self,*args,band_radius,band_transition,**kwargs):
        super().__init__(*args,**kwargs)
        self.band_radius,self.band_transition=band_radius,band_transition

    def residual(self,x):
        residual=super().residual(x).copy()
        residual[:self.image_rows]=boundary_band(residual[:self.image_rows].reshape(-1,2),self.band_radius,self.band_transition)[0].ravel()
        return residual

    def jacobian(self,x):
        raw=super().residual(x)
        _,multiplier=boundary_band(raw[:self.image_rows].reshape(-1,2),self.band_radius,self.band_transition)
        J=super().jacobian(x).copy()
        J[:self.image_rows]=np.einsum('nij,njk->nik',multiplier,J[:self.image_rows].reshape(-1,2,self.dimensions)).reshape(self.image_rows,self.dimensions)
        return J


def fixed_band_residual(objective,u,radius,transition):
    return boundary_band(objective.residual(u).reshape(-1,2),radius,transition)[0].ravel()


def fixed_band_jacobian(objective,u,radius,transition):
    _,M=boundary_band(objective.residual(u).reshape(-1,2),radius,transition)
    return np.einsum('nij,njk->nik',M,objective.jacobian(u).reshape(-1,2,6)).reshape(-1,6)
