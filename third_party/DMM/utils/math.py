import torch
import numpy as np


def compute_jacobian(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y[..., 0])
    jac = [torch.autograd.grad(y[..., t], x, grad_outputs=grad_outputs, create_graph=True)[0] for t in range(3)]
    jac = torch.stack(jac, dim=-2)
    return jac


def compute_gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


def compose_affine_trans(s, R, T):
    affine_trans = np.eye(4)
    affine_trans[:3, :3] = s * R
    affine_trans[:3, 3] = T
    return affine_trans


def apply_affine(trans, v):
    return (np.pad(v, [(0, 0), (0, 1)], mode='constant', constant_values=1) @ trans.T)[:, :3]


def screw_axis_to_rt(screw_axis):
    """SE(3) exponential stable at zero, including first/second derivatives."""
    omega, velocity = screw_axis[..., :3], screw_axis[..., 3:]
    skew = hat(omega)
    theta2 = (omega * omega).sum(-1)
    safe2 = theta2.clamp_min(1e-8)
    theta = safe2.sqrt()
    small = theta2 < 1e-6
    a = torch.where(small, 1 - theta2 / 6 + theta2**2 / 120, theta.sin() / theta)
    b = torch.where(small, .5 - theta2 / 24 + theta2**2 / 720, (1 - theta.cos()) / safe2)
    c = torch.where(small, 1 / 6 - theta2 / 120 + theta2**2 / 5040, (theta - theta.sin()) / (safe2 * theta))
    eye = torch.eye(3, dtype=screw_axis.dtype, device=screw_axis.device).expand(len(screw_axis), 3, 3)
    square = skew @ skew
    rotation = eye + a[:, None, None] * skew + b[:, None, None] * square
    jacobian = eye + b[:, None, None] * skew + c[:, None, None] * square
    return rotation, jacobian @ velocity[..., None]


def transform_screw(points, screw):
    rotation, translation = screw_axis_to_rt(screw)
    return (rotation @ points[..., None] + translation).squeeze(-1)


def hat(v: torch.Tensor) -> torch.Tensor:
    """
    From Pytorch3d:

    Compute the Hat operator [1] of a batch of 3D vectors.

    Args:
        v: Batch of vectors of shape `(minibatch , 3)`.

    Returns:
        Batch of skew-symmetric matrices of shape
        `(minibatch, 3 , 3)` where each matrix is of the form:
            `[    0  -v_z   v_y ]
             [  v_z     0  -v_x ]
             [ -v_y   v_x     0 ]`

    Raises:
        ValueError if `v` is of incorrect shape.

    [1] https://en.wikipedia.org/wiki/Hat_operator
    """

    N, dim = v.shape
    if dim != 3:
        raise ValueError("Input vectors have to be 3-dimensional.")

    h = torch.zeros((N, 3, 3), dtype=v.dtype, device=v.device)

    x, y, z = v.unbind(1)

    h[:, 0, 1] = -z
    h[:, 0, 2] = y
    h[:, 1, 0] = z
    h[:, 1, 2] = -x
    h[:, 2, 0] = -y
    h[:, 2, 1] = x

    return h


def get_submesh_by_vertex_labels(mesh, labels, target_id, mask=None):
    v_teeth = mesh.vertices[labels[:] == target_id, :]
    f_teeth = mesh.vertex_faces[labels[:] == target_id]

    f_teeth = set(f_teeth.flatten().tolist())
    if mask is not None:
        f_teeth = f_teeth.intersection(mask)
    f_teeth.discard(-1)

    if len(v_teeth) == 0 or len(f_teeth) == 0:
        print("No submesh labelled as {}".format(target_id))
        return None

    f_teeth = np.array(list(f_teeth))
    submesh = mesh.submesh([f_teeth])[0]

    return submesh
