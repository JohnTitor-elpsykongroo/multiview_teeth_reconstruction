"""Population anchored latent prior; input contains aggregates, no target code."""
import numpy as np
from joint_boundary_band import BandJointObjective

class PopulationJointObjective(BandJointObjective):
    def __init__(self,*args,population,**kwargs):
        super().__init__(*args,**kwargs)
        self.prior_J=np.zeros((self.dimensions-6,self.dimensions))
        self.prior_offset=np.zeros(self.dimensions-6)
        for label in self.labels:
            s=self.slices[label]; key=f'label_{label}'
            W=population[key+'_whitener']
            self.prior_J[s.start-6:s.stop-6,s]=W*self.scales[key][None,:]
            self.prior_offset[s.start-6:s.stop-6]=W@(self.initial[key]-population[key+'_mean'])
    def residual(self,x):
        r=super().residual(x)
        r[self.image_rows:]=self.prior_offset+self.prior_J@x
        return r
    def jacobian(self,x):
        J=super().jacobian(x)
        J[self.image_rows:]=self.prior_J
        return J
