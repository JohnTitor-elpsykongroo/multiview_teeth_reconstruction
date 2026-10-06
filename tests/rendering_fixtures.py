"""Deterministic *untrained* original DMM networks for gradient verification."""
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "DMM"))
from dmm import TEETH
from dmm.bundle import ModelBundle
from dmm.scene import ArchState, DualArchScene, Observation
from networks.dmm_net import DMM


def observation(view_id="front", size=40, offset=0.):
    K = np.array([[65., 0., (size - 1) / 2], [0., 65., (size - 1) / 2], [0., 0., 1.]])
    transform = np.eye(4)
    transform[0, 3] = offset
    return Observation(view_id, K, transform, np.zeros((size, size), np.uint8), np.ones((size, size), bool), {})


def arch_state(arch, dtype=torch.float64):
    gum = dict(latent_dim=2, model_type="sine", hyper_hidden_layers=0, hyper_hidden_features=4,
               mlp_input_dim=3, mlp_output_dim=5, mlp_num_hidden_layers=0, mlp_hidden_features=4)
    specs = dict(labels=[0, *TEETH[arch]], NetworkArchRef="mlp", GumDeformNetworkSpecs=gum,
                 TeethDeformNetworkSpecs=dict(gum, mlp_output_dim=8),
                 NetworkSpecsRef=dict(init_dims=[6], output_dims=1, activation="relu"))
    model = DMM(specs, arch=arch)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        for label in model.labels:
            # Phi = |x|+|y|+|z|-0.3, so ||gradient|| != 1. No training.
            ref = model.ref_nets_dict[str(label)].module
            ref.lin_0.weight.copy_(torch.cat((torch.eye(3), -torch.eye(3))))
            ref.lin_1.weight.fill_(1.)
            ref.lin_1.bias.fill_(-.3)
            hyper = model.deform_nets_dict[str(label)].module.hyper_net
            net = hyper.nets[-1]  # generated final deformation-output bias
            assert hyper.names[-1].endswith("bias")
            net.net[-1][0].bias[-1] = -6. if label == 0 else 6.
            if label:
                # Actual hypernetwork maps q[0] into SDF correction -0.1*q[0].
                net.net[0][0].weight[0, 0] = 1.
                net.net[0][0].bias[0] = 1.
                net.net[-1][0].weight[-2, 0] = -.1
                net.net[-1][0].bias[-2] = .1
                # q[1] changes soft semantic blending through the same network.
                net.net[0][0].weight[1, 1] = 1.
                net.net[0][0].bias[1] = 1.
                net.net[-1][0].weight[-1, 1] = 2.
                net.net[-1][0].bias[-1] -= 2.
    model.eval().requires_grad_(False)
    means = {k: torch.zeros(2) for k in model.labels}
    factors = {k: torch.eye(2) for k in model.labels}
    meta = dict(arch=arch, model_id=f"untrained-octahedron-{arch}", sampling_domain_model=[[-.6]*3, [.6]*3])
    bundle = ModelBundle(model, meta, Path("untrained-fixture-no-bundle-file"), means, factors)
    pose = np.eye(4)
    pose[:3, 3] = [-7. if arch == "upper" else 7., -6. if arch == "upper" else 6., 85. if arch == "upper" else 95.]
    tooth = TEETH[arch][0]
    state = ArchState(bundle, {k: k == tooth for k in TEETH[arch]}, dict(T_world_from_arch=pose.tolist(), q={str(tooth): [0., 0.]}))
    return state.to(dtype=dtype)


def fixture_scene(dtype=torch.float64):
    return DualArchScene({"scene_id": "untrained-render-bridge-fixture"},
                         [observation(), observation("side", offset=5.)],
                         {arch: arch_state(arch, dtype) for arch in TEETH})
