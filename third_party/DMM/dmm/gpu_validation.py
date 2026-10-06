"""Fail closed on a CUDA build whose antialias position gradients are broken."""
from types import SimpleNamespace
import numpy as np
import torch

from dmm.rendering import RenderConfig, render_mesh
from dmm.surface import DifferentiableMesh
from dmm.validation import require, sha256

_VALIDATED = {}


def validate_cuda_backend(device):
    import _nvdiffrast_c
    import nvdiffrast
    binary_hash = sha256(_nvdiffrast_c.__file__)
    key = (str(device), torch.__version__, binary_hash)
    if key in _VALIDATED: return _VALIDATED[key]
    obs = SimpleNamespace(view_id='cuda_preflight', labels=np.zeros((40, 40), np.uint8),
                          K=np.array([[65., 0, 19.5], [0, 65., 19.5], [0, 0, 1.]]), T_camera_from_world=np.eye(4))
    vertices = torch.tensor([[-10., -10., 80.], [10., -10., 80.], [0., 10., 80.]], device=device)
    faces = torch.tensor([[0, 1, 2]], device=device)
    semantics = torch.zeros(3, 29, device=device)
    semantics[:, 1] = 1
    direction = torch.zeros_like(vertices)
    direction[2, 1] = 1
    def area(value):
        mesh = DifferentiableMesh(vertices + value * direction, faces, semantics, torch.zeros(1, device=device, dtype=torch.long), {})
        return render_mesh(mesh, obs, RenderConfig()).semantics[..., 1].sum()
    with torch.enable_grad():
        delta = torch.tensor(0., device=device, requires_grad=True)
        gradient = float(torch.autograd.grad(area(delta), delta)[0])
        with torch.no_grad(): finite = float((area(delta + .002) - area(delta - .002)) / .004)
    require(np.isfinite(gradient) and np.isfinite(finite) and abs(gradient) > 1e-6
            and abs(gradient-finite) <= .001 + .05*abs(finite),
            f'CUDA antialias gradient preflight failed: {gradient} vs {finite}; validate/rebuild backend before fitting')
    report = dict(torch=torch.__version__, cuda=torch.version.cuda, device=torch.cuda.get_device_name(device),
                  capability=list(torch.cuda.get_device_capability(device)), nvdiffrast=nvdiffrast.__version__,
                  binary_sha256=binary_hash, gradient=gradient, finite_difference=finite)
    _VALIDATED[key] = report
    return report
