

import torch
from torch import nn
import contextlib
import io
from dmm import TEETH
from dmm.validation import require
from utils.math import transform_screw
from utils.math import compute_gradient, compute_jacobian, screw_axis_to_rt
from networks.loss import component_sdf_normal_loss, overall_sdf_normal_loss
from networks.deform_net import DeformNet
from torchmeta.modules import DataParallel


class _SingleDeviceModule(nn.Module):
    """Keep checkpoint '.module' names without DataParallel's eager GPU move."""
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class DMM(nn.Module):
    def __init__(self, specs, avg_centers=None, arch=None, **kwargs):
        super().__init__()
        # Define DMM-Net
        self.label_list = specs["labels"]
        self.labels = tuple(self.label_list)
        self.arch = arch
        self.specs = specs
        self.dimensions = {k: specs["GumDeformNetworkSpecs" if k == 0 else "TeethDeformNetworkSpecs"]["latent_dim"] for k in self.labels}
        if arch is not None:
            require(arch in TEETH and self.label_list == [0, *TEETH[arch]], "spec labels must match the full ordered arch")
            require(specs["NetworkArchRef"] == "mlp", "v1 supports official mlp reference")
            require(specs["GumDeformNetworkSpecs"]["mlp_output_dim"] == 5 and specs["TeethDeformNetworkSpecs"]["mlp_output_dim"] == 8,
                    "v1 requires gum displacement and tooth screw deformation")
            require(specs["NetworkSpecsRef"].get("output_dims", 1) == 1, "reference must output one SDF")
            require(all(type(d) is int and d > 0 for d in self.dimensions.values()), "invalid latent dimensions")

        self.deform_nets_dict = torch.nn.ModuleDict()
        self.ref_nets_dict = torch.nn.ModuleDict()

        arch_ref = __import__("networks." + specs["NetworkArchRef"], fromlist=["MLPNet"])

        for label_id in self.label_list:
            with contextlib.redirect_stdout(io.StringIO()) if arch is not None else contextlib.nullcontext():
                if label_id == 0:
                    self.deform_nets_dict[str(label_id)] = DeformNet(**specs["GumDeformNetworkSpecs"])
                else:
                    self.deform_nets_dict[str(label_id)] = DeformNet(**specs["TeethDeformNetworkSpecs"])
                self.ref_nets_dict[str(label_id)] = arch_ref.MLPNet(**specs["NetworkSpecsRef"])

            # Native independent-arch callers choose the device with .to(device).
            # DataParallel otherwise eagerly moves a single-GPU host to cuda:0.
            deform_wrapper = _SingleDeviceModule if arch is not None else DataParallel
            reference_wrapper = _SingleDeviceModule if arch is not None else torch.nn.DataParallel
            self.deform_nets_dict[str(label_id)] = deform_wrapper(self.deform_nets_dict[str(label_id)])
            self.ref_nets_dict[str(label_id)] = reference_wrapper(self.ref_nets_dict[str(label_id)])

        self.bce = torch.nn.BCEWithLogitsLoss()
        self.l1loss = torch.nn.L1Loss(reduction='sum')
        self.avg_centers = avg_centers

        if arch is None:
            print(self)

    def component(self, points, code, label):
        """One component on Nx3 model coordinates, preserving latent gradients."""
        require(points.ndim == 2 and points.shape[1] == 3, "query expects Nx3 model coordinates")
        require(code.shape == (self.dimensions[label],), f"latent dimension mismatch for {label}")
        output = self.deform_nets_dict[str(label)].module(points[None], code[None])[0]
        warped = points + output[:, :3] if label == 0 else transform_screw(points, output[:, :6])
        template = self.ref_nets_dict[str(label)].module(warped).squeeze(-1)
        return dict(sdf=template + output[:, -2], template=template, warped=warped,
                    deformation=warped - points, correction=output[:, -2], logit=output[:, -1])

    def query(self, points, codes):
        """Blend only gum and explicitly present tooth codes, within this arch."""
        require(0 in codes and set(codes).issubset(self.labels), "codes require gum and only this arch's present teeth")
        labels = tuple(k for k in self.labels if k in codes)
        components = {k: self.component(points, codes[k], k) for k in labels}
        logits = torch.stack([components[k]["logit"] for k in labels], -1)
        weights = torch.softmax(torch.nn.functional.logsigmoid(logits), dim=-1)
        fields = torch.stack([components[k]["sdf"] for k in labels], -1)
        return dict(sdf=(fields * weights).sum(-1), weights=weights, labels=labels, components=components)


    # for training
    def forward(self, latent_vecs, sdf_data, is_on_surf, normal, centers_tensor, indices, **kwargs):
        # Get coordinates
        coords = sdf_data[..., :3].requires_grad_(True)

        # Get labels
        label_ids = sdf_data[..., 4]

        # Collect all component indices that are involved in this batch
        batch_label_list = torch.unique(label_ids).int().tolist()
        # -1 means off-the-surface
        if -1 in batch_label_list:
            batch_label_list.remove(-1)

        blend_loss = 0.
        embedding_reg_loss = 0.
        sep_loss = 0.
        center_loss = 0.

        blend_weight_list = list()
        sdf_list = list()

        deform_model_output = dict()
        embeddings = dict()
        pred_sdf = dict()
        center = dict()
        deform_model_center_output = dict()
        new_coords = dict()
        deformation = dict()

        for label_id in batch_label_list:
            str_label_id = str(label_id)

            # Get the latent codes
            embeddings[label_id] = latent_vecs[str_label_id](indices)
            embedding_reg_loss += torch.mean(embeddings[label_id] ** 2) * 1e6

            # Deformed by the deform-net
            deform_model_output[label_id] = self.deform_nets_dict[str_label_id](coords, embeddings[label_id])

            if label_id > 0:
                # Deform centers
                center[label_id] = centers_tensor[..., label_id:label_id + 1, :]
                deform_model_center_output[label_id] = self.deform_nets_dict[str_label_id](center[label_id], embeddings[label_id])

                # Compute the centroid loss
                for batch_id in range(center[label_id].shape[0]):
                    # A meaningful center should be far from [0, 0, 0] in our coordinate system
                    if center[label_id][batch_id, 0, :].norm() > 1e-5:
                        center_screw_axis = deform_model_center_output[label_id][batch_id, ..., :6]

                        # Compute rotation and translation from the screw axis
                        R, T = screw_axis_to_rt(center_screw_axis)

                        # Transform the center from original position to a new position in the shared reference coordinate
                        center_new_coords = (torch.bmm(R, center[label_id][batch_id, 0, :][None, ..., None]) + T).squeeze()
                        center_loss += self.l1loss(center_new_coords, self.avg_centers[label_id, :3])

            # Compute deformed coordinates for all samples

            # If the output is 5 dims, it would be: [dx, dy, dz, \Delta s, \delta], where (dx, dy, dz) is the displacement,
            # \Delta s is the SDF correctioin, and \delta is the segmentation indicator
            # If the output is 8 dims, it would be: [r_x, r_y, r_z, v_x, v_y, v_z, \Delta s, \delta], where (r;v)) is the screw axis
            if deform_model_output[label_id].shape[-1] == 5:
                deformation[label_id] = deform_model_output[label_id][..., :3]
                new_coords[label_id] = coords + deformation[label_id]
            else:
                screw_axis = deform_model_output[label_id][..., :6]
                screw_axis = screw_axis.view(-1, screw_axis.shape[-1])
                R, T = screw_axis_to_rt(screw_axis)
                coords_flatten = coords.view(-1, coords.shape[-1])
                new_coords[label_id] = (torch.bmm(R, coords_flatten[..., None]) + T).squeeze().view(coords.shape)
                deformation[label_id] = new_coords[label_id] - coords

            # Forward the ref-net for SDF values
            pred_sdf[label_id] = self.ref_nets_dict[str(label_id)](new_coords[label_id])

            # Blend components for the overall SDF values
            blend_weights = deform_model_output[label_id][..., -1]
            correction = deform_model_output[label_id][..., -2:-1]

            # Supervise on surface only
            surface_point_filter = label_ids != -1

            # Flatten and compute the mask for BCE loss
            surface_point_weights = blend_weights[surface_point_filter]
            surface_point_id = label_ids[surface_point_filter]
            seg_mask = (surface_point_id == label_id).float()

            # Compute BCE loss
            blend_loss += self.bce(surface_point_weights, seg_mask) * 100

            # Create a mask for each case that indicates whether a specific tooth exists
            has_this_tooth = (label_ids == label_id).any(dim=-1).float()[..., None]

            # Use sigmoid to make sure the blend weights are positive
            blend_weights = torch.sigmoid(blend_weights)

            # For missing tooth, set it to zero
            blend_weight_list.append(blend_weights * has_this_tooth)

            # Compute SDF values for the component
            component_sdf = pred_sdf[label_id] + correction
            sdf_list.append(component_sdf)

            # Compute gradients for component-wise losses
            grad_template = compute_gradient(pred_sdf[label_id], new_coords[label_id])
            grad_deform = compute_jacobian(deformation[label_id], coords)

            losses = component_sdf_normal_loss(component_sdf, coords, is_on_surf, normal,
                                     correction, grad_deform, grad_template,
                                               label_ids[..., None], label_id)


            for loss_name, loss in losses.items():
                single_loss = loss.mean()
                sep_loss += single_loss

        # Collect segmentation indicators and normalize them
        predict_weights = torch.stack(blend_weight_list, dim=-1)
        predict_weights = torch.nn.functional.normalize(predict_weights, dim=-1, p=1)

        # Blend component-wise SDF values for a final SDF value
        sdf_final = torch.concat(sdf_list, dim=-1)
        sdf_final = torch.sum(sdf_final * predict_weights, dim=-1, keepdim=True)

        losses = overall_sdf_normal_loss(sdf_final, coords, is_on_surf, normal)

        train_loss = 0.
        for loss_name, loss in losses.items():
            single_loss = loss.mean()
            train_loss += single_loss * 0.1

        blend_loss = blend_loss / len(batch_label_list)
        embedding_reg_loss = embedding_reg_loss / len(batch_label_list)
        sep_loss = sep_loss / len(batch_label_list)
        center_loss = center_loss / len(batch_label_list)

        train_loss += blend_loss
        train_loss += embedding_reg_loss
        train_loss += sep_loss
        train_loss += center_loss

        losses_log = dict()
        losses_log['sep_loss'] = sep_loss.item()
        losses_log['embedding_reg_loss'] = embedding_reg_loss.item()
        losses_log['center_loss'] = center_loss.item()
        losses_log['blend_loss'] = blend_loss.item()

        return train_loss, losses_log


    def inference(self, latent_vec, sample_data, is_on_surface = None, normal_gt = None):
        coords = sample_data[..., :3].requires_grad_(True)

        blend_weight_list = list()
        sdf_list = list()

        embedding_loss = 0.

        for label_id in latent_vec.keys():
            str_label_id = str(int(label_id))
            embeddings = latent_vec[label_id]
            embedding_loss += torch.mean(embeddings ** 2) * 1e6

            deform_model_output = self.deform_nets_dict[str_label_id](coords, embeddings)
            if deform_model_output.shape[-1] == 5:
                deformation = deform_model_output[..., :3]
                new_coords = coords + deformation
            else:
                screw_axis = deform_model_output[..., :6]
                screw_axis = screw_axis.view(-1, screw_axis.shape[-1])
                R, T = screw_axis_to_rt(screw_axis)
                coords_flatten = coords.view(-1, coords.shape[-1])
                new_coords = (torch.bmm(R, coords_flatten[..., None]) + T).squeeze()
                new_coords = new_coords.view(coords.shape)

            correction = deform_model_output[..., -2:-1]
            blend_weights = deform_model_output[..., -1]

            pred_sdf = self.ref_nets_dict[str_label_id](new_coords)
            component_sdf = pred_sdf + correction

            blend_weight_list.append(blend_weights)
            sdf_list.append(component_sdf)

        predict_weights = torch.stack(blend_weight_list, dim=-1)
        predict_weights = torch.sigmoid(predict_weights)
        predict_weights = torch.nn.functional.normalize(predict_weights, dim=-1, p=1)

        sdf_final = torch.concat(sdf_list, dim=-1)
        sdf_final = torch.sum(sdf_final * predict_weights, dim=-1, keepdim=True)

        embedding_loss = embedding_loss / len(latent_vec)
        if is_on_surface is not None and normal_gt is not None:
            losses = overall_sdf_normal_loss(sdf_final, coords, is_on_surface, normal_gt)
            for loss_name, loss in losses.items():
                single_loss = loss.mean()
                embedding_loss += single_loss

            return sdf_final.squeeze(0), embedding_loss
        else:
            return sdf_final.squeeze(0)
