import argparse
import json
from pathlib import Path

from dmm.validation import reference, require, stamped, write_json


def main():
    parser = argparse.ArgumentParser(description="Dual-arch DMM interfaces (v1)")
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("make-config", help="create a hashed, independent single-arch training config")
    create.add_argument("--arch", choices=["upper", "lower"], required=True)
    create.add_argument("--manifest", required=True)
    create.add_argument("--canonical-reference", required=True)
    create.add_argument("--specs", required=True)
    create.add_argument("--output", required=True)
    create.add_argument("--epochs", type=int, default=100)
    create.add_argument("--recipe-json", help="explicit loss weights, reduction and optimizer recipe")
    create.add_argument("--validation-json", help="held-out latent fitting and evaluation settings")
    evaluate = commands.add_parser("evaluate-checkpoint", help="freeze decoder and fit temporary val latents")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--device", default="cpu")
    prepare = commands.add_parser("prepare-scene", help="validate stage-3 candidate and copy a fresh fit package")
    prepare.add_argument("candidate")
    prepare.add_argument("--output", required=True)
    validate = commands.add_parser("validate-training")
    validate.add_argument("--config", required=True)
    validate.add_argument("--other-config", help="optional cross-arch patient split check")
    train = commands.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--output", required=True, help="new output directory, never overwritten")
    train.add_argument("--device", default="cpu")
    train.add_argument("--resume")
    dual_create = commands.add_parser('make-dual-config', help='optional independent upper/lower workflow')
    dual_create.add_argument('--upper-config', required=True)
    dual_create.add_argument('--lower-config', required=True)
    dual_create.add_argument('--output', required=True)
    dual_validate = commands.add_parser('validate-dual-training')
    dual_validate.add_argument('--config', required=True)
    dual_train = commands.add_parser('train-dual', help='explicit opt-in: run independent upper then lower training')
    dual_train.add_argument('--config', required=True)
    dual_train.add_argument('--output', required=True)
    dual_train.add_argument('--device', default='cpu')
    dual_train.add_argument('--resume', help='unfinished workflow.json; use a fresh output directory')
    dual_train.add_argument('--upper-resume', help='unfinished upper arch checkpoint')
    dual_train.add_argument('--lower-resume', help='unfinished lower arch checkpoint')
    export = commands.add_parser("export-bundle")
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--model-id", required=True)
    scene = commands.add_parser("validate-scene")
    scene.add_argument("manifest")
    scene.add_argument("--device", default="cpu")
    scene.add_argument("--mesh-output", help="optional NPZ evaluation mesh, not differentiable")
    scene.add_argument("--resolution", type=int, default=64)
    render = commands.add_parser("render-scene", help="render all known views; optional image-loss backward diagnostic, no optimizer")
    render.add_argument("manifest")
    render.add_argument("--device", default="cuda")
    render.add_argument("--surface-config", required=True)
    render.add_argument("--render-config", required=True)
    render.add_argument("--matching-config", help="optional SemanticXY diagnostics")
    render.add_argument("--backward-check", action="store_true")
    render.add_argument("--output", required=True, help="fresh diagnostic directory")
    fit = commands.add_parser("fit-scene", help="staged fitting with frozen decoders and fixed cameras")
    fit.add_argument("manifest")
    fit.add_argument("--device", default="cuda")
    fit.add_argument("--fit-config", required=True)
    fit.add_argument("--surface-config", required=True)
    fit.add_argument("--render-config", required=True)
    fit.add_argument("--matching-config", required=True)
    fit.add_argument("--output", required=True)
    fit.add_argument("--resume", help="unfinished checkpoint; output must still be fresh")
    fit.add_argument('--collision-config', help='optional penetration proxy; disabled when omitted')
    fit.add_argument('--initialize-mean', action='store_true', help='rank independent mean-model pose proposals; incompatible with resume')
    args = parser.parse_args()
    if args.command == "make-config":
        path = Path(args.output).resolve()
        require(not path.exists(), "config already exists")
        config = stamped("arch_training_config", arch=args.arch, manifest=reference(args.manifest, path),
                         canonical_reference=reference(args.canonical_reference, path), specs=reference(args.specs, path),
                         epochs=args.epochs, points_per_component=256, offsurface_points=2048,
                         learning_rate=.0001, seed=42, checkpoint_every=10)
        from training.arch_training import load_training_config
        from dmm.validation import read_json
        if args.recipe_json: config["recipe"] = read_json(args.recipe_json)
        if args.validation_json: config["validation"] = read_json(args.validation_json)
        from networks.dmm_net import DMM
        config, _, _, specs = load_training_config(path, config=config)
        DMM(specs, arch=args.arch)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, config)
        result = dict(status="CONFIG_CREATED_AND_DATA_VALIDATED", config=str(path))
    elif args.command == "prepare-scene":
        from dmm.handoff import prepare_candidate
        result = prepare_candidate(args.candidate, args.output)
    elif args.command == "evaluate-checkpoint":
        from training.arch_training import evaluate_checkpoint
        result = evaluate_checkpoint(args.config, args.checkpoint, args.output, args.device)
    elif args.command == "validate-training":
        from training.arch_training import load_training_config
        from data.arch_dataset import validate_patient_splits
        from networks.dmm_net import DMM
        config, dataset, _, specs = load_training_config(args.config)
        model = DMM(specs, arch=config["arch"])
        if args.other_config:
            _, other, _, _ = load_training_config(args.other_config)
            validate_patient_splits(dataset, other)
        result = dict(status="TRAINING_INPUT_VALIDATED", arch=dataset.arch, training_cases=len(dataset),
                      component_dimensions=model.dimensions, cross_manifest_split_checked=bool(args.other_config))
    elif args.command == 'make-dual-config':
        from training.dual_training import make_dual_config
        result = make_dual_config(args.upper_config, args.lower_config, args.output)
    elif args.command == 'validate-dual-training':
        from training.dual_training import load_dual_config
        _, _, result = load_dual_config(args.config)
    elif args.command == 'train-dual':
        from training.dual_training import train_dual
        resumes = {a: p for a, p in [('upper', args.upper_resume), ('lower', args.lower_resume)] if p}
        result = train_dual(args.config, args.output, args.device, args.resume, resumes)
    elif args.command == "train":
        from training.arch_training import train_arch
        result = train_arch(args.config, args.output, args.device, args.resume)
    elif args.command == "export-bundle":
        from training.arch_training import export_bundle
        result = export_bundle(args.checkpoint, args.output, args.model_id)
    elif args.command == "fit-scene":
        from dmm.scene import load_scene
        from dmm.surface import SurfaceConfig
        from dmm.rendering import RenderConfig
        from dmm.semanticxy import SemanticXYConfig
        from dmm.fitting import FitConfig, fit_scene
        from dmm.collision import CollisionConfig
        loaded = load_scene(args.manifest, args.device)
        surface_config, render_config = SurfaceConfig.load(args.surface_config), RenderConfig.load(args.render_config)
        matching_config = SemanticXYConfig.load(args.matching_config)
        require(not args.initialize_mean or not args.resume, "mean initialization cannot be combined with resume")
        # Keep signature independent of a new initialization on resume; the fitted
        # checkpoint records its starting parameters and restores optimized state.
        initialization_report = None
        if args.initialize_mean:
            from dmm.initialization import initialize_mean_scene
            require(not Path(args.output).exists(), "fit output already exists")
            initialization_report = initialize_mean_scene(loaded, surface_config, render_config, matching_config)
        try:
            result = fit_scene(loaded, args.output, FitConfig.load(args.fit_config),
                               surface_config, render_config, matching_config, args.resume,
                               CollisionConfig.load(args.collision_config) if args.collision_config else None)
        finally:
            if initialization_report is not None and Path(args.output).is_dir():
                write_json(Path(args.output)/"mean_initialization.json", initialization_report)
        result = dict(status=result['status'], output=str(Path(args.output).resolve()))
    elif args.command == "render-scene":
        import torch
        from dmm.scene import load_scene
        from dmm.surface import SurfaceConfig
        from dmm.rendering import RenderConfig
        from dmm.render_io import write_render_artifacts
        from dmm.semanticxy import SemanticXYConfig, SemanticXYMatcher
        require(not Path(args.output).exists(), "render output already exists")
        require(not args.backward_check or args.matching_config, "backward check requires matching config")
        loaded = load_scene(args.manifest, args.device)
        rendered, mesh, _ = loaded.render_views(SurfaceConfig.load(args.surface_config), RenderConfig.load(args.render_config))
        matching, gradients = None, None
        if args.matching_config:
            match = SemanticXYMatcher(SemanticXYConfig.load(args.matching_config)).match_scene(rendered, loaded.observations)
            matching = dict(loss=float(match["loss"].detach()), views={k: v.diagnostics for k, v in match["views"].items()},
                            needs_visibility_recovery=match["needs_visibility_recovery"], resolved_config=match["resolved_config"])
            if args.backward_check:
                match["loss"].backward()
                gradients = {}
                for name, param in loaded.named_parameters():
                    if param.requires_grad:
                        require(param.grad is None or bool(torch.isfinite(param.grad).all()), f"nonfinite gradient: {name}")
                        gradients[name] = None if param.grad is None else float(param.grad.norm())
        result = write_render_artifacts(args.output, loaded, rendered, mesh, matching, gradients)
        result = dict(status=result["status"], output=str(Path(args.output).resolve()))
    else:
        from dmm.scene import load_scene
        loaded = load_scene(args.manifest, args.device)
        result = dict(status="SCENE_LOADED_NOT_RECONSTRUCTED", **loaded.summary())
        if args.mesh_output:
            import numpy as np
            path = Path(args.mesh_output)
            require(not path.exists(), "mesh output already exists")
            mesh = loaded.extract_mesh(args.resolution)
            with path.open("xb") as stream:
                np.savez(stream, vertices_world_mm=mesh.vertices_world_mm, faces=mesh.faces,
                         semantics=mesh.semantics, face_arch=mesh.face_arch)
            result["mesh"] = str(path.resolve())
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
