#!/usr/bin/env python3
"""Evaluate PhySIC renders with the same CLIP metric as module 06."""

from __future__ import annotations

import argparse
from pathlib import Path

from transformers import CLIPModel, CLIPProcessor

from physic_eval_utils import (
    DEFAULT_OUTPUT_MODE,
    PROJECT_DIR,
    discover_physic_interactions,
    ensure_dir,
    load_python_module,
    physic_eval_root,
    save_csv_rows,
    save_json,
)


BASE = load_python_module(
    "module06_semantics",
    PROJECT_DIR / "06_Evaluate_Interaction" / "03_evaluate_semantics.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate PhySIC render semantic consistency with CLIP."
    )
    parser.add_argument("--interaction_name", default="interaction_01")
    parser.add_argument("--output_mode", default=DEFAULT_OUTPUT_MODE)
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--clip_model", type=str, default=BASE.DEFAULT_CLIP_MODEL)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def eval_one(
    interaction_name: str,
    args: argparse.Namespace,
    model: CLIPModel,
    processor: CLIPProcessor,
    device,
) -> dict:
    render_root = physic_eval_root(args.output_mode) / interaction_name / "semantics"
    output_base = (
        Path(args.output_root).resolve()
        if args.output_root
        else physic_eval_root(args.output_mode)
    )
    output_root = output_base / interaction_name / "semantics"
    return BASE.evaluate_interaction_semantics(
        interaction_name=interaction_name,
        input_scene_json_path=PROJECT_DIR
        / "01_Generate_SIG"
        / "input_prompts"
        / interaction_name
        / "input_scene.json",
        render_root=render_root,
        output_root=output_root,
        model=model,
        processor=processor,
        device=device,
    )


def main() -> None:
    args = parse_args()
    all_mode = args.interaction_name == "all"
    if all_mode:
        interaction_names = discover_physic_interactions(args.output_mode)
    else:
        interaction_names = [args.interaction_name]

    device = BASE.parse_device(args.device)
    model = CLIPModel.from_pretrained(args.clip_model, use_safetensors=True).to(device)
    processor = CLIPProcessor.from_pretrained(args.clip_model)
    model.eval()
    rows = [
        eval_one(name, args, model, processor, device) for name in interaction_names
    ]

    if all_mode:
        combined_root = ensure_dir(
            Path(args.output_root).resolve()
            if args.output_root
            else physic_eval_root(args.output_mode)
        )
        mean_score = sum(float(row["clip_score"]) for row in rows) / len(rows)
        combined_rows = rows + [
            {
                "interaction_name": "__mean__",
                "clip_score": float(mean_score),
                "num_renders": sum(int(row["num_renders"]) for row in rows),
            }
        ]
        save_csv_rows(
            combined_root / "semantics.csv",
            combined_rows,
            ["interaction_name", "clip_score", "num_renders"],
        )
        save_json(
            combined_root / "semantics.json",
            {
                "interactions": rows,
                "aggregate": {
                    "num_interactions": len(rows),
                    "mean_clip_score": float(mean_score),
                },
            },
        )


if __name__ == "__main__":
    main()
