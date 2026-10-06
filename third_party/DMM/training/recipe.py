"""Versioned, resolved training recipes; numerical defaults are not calibrated."""
import math
from dmm.validation import require

WEIGHTS = dict(surface=3000., flip=3000., offsurface=500., normal=100.,
               eikonal=50., correction=500., smooth=5., template_normal=100.,
               blend=100., center=1., latent=1e6, global_surface=300.,
               global_offsurface=20., global_normal=10., global_eikonal=5.)


def resolve_recipe(config):
    supplied = config.get("recipe", {})
    require(isinstance(supplied, dict), "recipe must be a mapping")
    version = supplied.get("version", "effective_count_v1")
    require(version in ("effective_count_v1", "masked_mean_v1"), "unknown training recipe version")
    defaults = dict(version=version, reduction="all_points" if version == "masked_mean_v1" else "effective", weights=WEIGHTS,
                    latent_init_std=.001, flip_threshold_model=.03,
                    deformation_lr=config["learning_rate"], reference_lr=config["learning_rate"],
                    latent_lr=config["learning_rate"], lr_step_epochs=30, lr_gamma=1.,
                    gradient_clip_norm=0., center_reduction="all_components" if version == "masked_mean_v1" else "present_teeth")
    defaults["normal_epsilon"] = 1e-8
    require(isinstance(supplied, dict) and not set(supplied) - set(defaults), "unknown recipe field")
    result = dict(defaults, **supplied)
    require(result["center_reduction"] in ("all_components", "present_teeth"), "invalid center reduction")
    require(version != "masked_mean_v1" or (result["reduction"] == "all_points" and result["center_reduction"] == "all_components"), "masked_mean_v1 reduction mismatch")
    require(result["reduction"] in ("effective", "all_points"), "invalid loss reduction")
    require(isinstance(result["weights"], dict) and set(result["weights"]) == set(WEIGHTS), "recipe must resolve all loss weights")
    result["weights"] = dict(result["weights"])
    for key, value in {**result["weights"], **{k: result[k] for k in (
            "latent_init_std", "flip_threshold_model", "deformation_lr", "reference_lr", "latent_lr", "lr_gamma", "gradient_clip_norm", "normal_epsilon")}}.items():
        require(type(value) in (int, float) and math.isfinite(value) and value >= 0, f"invalid recipe value {key}")
    require(all(result[k] > 0 for k in ("deformation_lr", "reference_lr", "latent_lr", "latent_init_std", "lr_gamma"))
            and result["lr_gamma"] <= 1, "invalid learning rate/initialization")
    require(0 < result["normal_epsilon"] <= .1, "invalid normal_epsilon")
    require(type(result["lr_step_epochs"]) is int and result["lr_step_epochs"] > 0, "invalid lr_step_epochs")
    return result


def resolve_validation(config):
    defaults = dict(enabled=False, every_epochs=1, fit_steps=100, learning_rate=.001,
                    points_per_component=128, offsurface_points=1024, evaluation_points_per_component=512,
                    seed=20261006, mesh_resolution=48, mesh_samples_per_component=1024,
                    case_ids=None)
    supplied = config.get("validation", {})
    require(isinstance(supplied, dict) and not set(supplied) - set(defaults), "unknown validation field")
    result = dict(defaults, **supplied)
    require(type(result["enabled"]) is bool, "invalid validation enabled")
    for key in ("every_epochs", "fit_steps", "points_per_component", "offsurface_points", "evaluation_points_per_component", "mesh_resolution", "mesh_samples_per_component"):
        require(type(result[key]) is int and result[key] > 0, f"invalid validation {key}")
    require(8 <= result["mesh_resolution"] <= 256, "invalid validation mesh resolution")
    require(type(result["seed"]) is int and result["seed"] >= 0, "invalid validation seed")
    require(type(result["learning_rate"]) in (int, float) and math.isfinite(result["learning_rate"]) and result["learning_rate"] > 0, "invalid validation learning rate")
    ids = result["case_ids"]
    require(ids is None or (isinstance(ids, list) and bool(ids) and all(isinstance(x, str) for x in ids) and len(set(ids)) == len(ids)), "invalid validation case_ids")
    return result
