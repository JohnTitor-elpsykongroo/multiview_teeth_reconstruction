"""Standalone derivative check at interior, transition, exterior and seams."""
import json
import sys
from pathlib import Path
import numpy as np
from joint_boundary_band import boundary_band
radius=2**.5; transition=.25; h=1e-6
vectors=np.array([[0,0],[.2,.4],[radius,0],[radius+.1,0],[radius+transition,0],[2,.3]],dtype=float)
_,J=boundary_band(vectors,radius,transition)
finite=np.stack([(boundary_band(vectors+np.eye(2)[k]*h,radius,transition)[0]-boundary_band(vectors-np.eye(2)[k]*h,radius,transition)[0])/(2*h) for k in range(2)],2)
identity,J0=boundary_band(vectors,0,transition)
result={'status':'PASS' if np.max(abs(J-finite))<3e-6 and np.array_equal(identity,vectors) else 'FAIL','max_absolute_jacobian_error':float(np.max(abs(J-finite))),'zero_band_identity':bool(np.array_equal(identity,vectors)),'radius':radius,'transition':transition,'finite_difference_step':h,'test_vectors':vectors.tolist(),'scope':'C1 radial band analytic derivative including seams; zero radius equals legacy residual'}
path=Path(sys.argv[1]); path.parent.mkdir(exist_ok=True,parents=True)
if path.exists(): raise FileExistsError(path)
path.write_text(json.dumps(result,indent=2)); print(json.dumps(result)); assert result['status']=='PASS'
