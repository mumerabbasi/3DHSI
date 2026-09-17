#!/usr/bin/env python3
"""Evaluate PhySIC renders with the same VLM rubric as module 06."""

from __future__ import annotations

import argparse
from pathlib import Path

from physic_eval_utils import (
    DEFAULT_OUTPUT_MODE,
    PROJECT_DIR,
    discover_physic_interactions,
    ensure_dir,
    load_python_module,
    physic_eval_root,
    save_csv_rows,
)


BASE = load_python_module(
    "module06_vlm",
    PROJECT_DIR / "06_Evaluate_Interaction" / "04_evaluate_vlm.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate PhySIC renders with the module-06 VLM verifier."
    )
    parser.add_argument(
        "--interaction_name", default="interaction_01", help="Interaction ID, or all."
    )
    parser.add_argument("--output_mode", default=DEFAULT_OUTPUT_MODE)
    parser.add_argument(
        "--aggregate_evals",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--vlm_provider",
        choices=BASE.VLM_PROVIDERS,
        default="gemini",
    )
    parser.add_argument("--qwen_model", type=str, default=BASE.DEFAULT_QWEN_MODEL)
    parser.add_argument("--gemini_model", type=str, default=BASE.DEFAULT_GEMINI_MODEL)
    parser.add_argument("--ollama_host", type=str, default="http://localhost:11434")
    parser.add_argument(
        "--gemini_api_key_file",
        type=str,
        default=str(PROJECT_DIR / ".secrets" / "gemini_api_key"),
    )
    parser.add_argument("--prompt_template", type=str, default=None)
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--max_image_side", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=BASE.DEFAULT_TEMPERATURE)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--timeout_s", type=int, default=600)
    parser.add_argument("--gemini_max_output_tokens", type=int, default=4096)
    parser.add_argument("--gemini_retries", type=int, default=3)
    parser.add_argument("--gemini_retry_sleep_s", type=float, default=10.0)
    args = parser.parse_args()
    return args


def aggregate(output_root: Path) -> None:
    metrics_csv_paths = sorted(output_root.glob("interaction_*/vlm/metrics.csv"))
    if not metrics_csv_paths:
        raise FileNotFoundError(
            f"No VLM metrics.csv files found under {output_root}/interaction_*/vlm."
        )
    rows = []
    for metrics_csv_path in metrics_csv_paths:
        row = {"interaction_name": metrics_csv_path.parent.parent.name}
        row.update(BASE.load_vlm_metrics_row(metrics_csv_path))
        rows.append(row)
    mean_row = {"interaction_name": "__mean__"}
    for fieldname in BASE.CSV_FIELDNAMES:
        values = [
            float(row[fieldname])
            for row in rows
            if isinstance(row.get(fieldname), int | float)
        ]
        mean_row[fieldname] = float(sum(values) / len(values)) if values else None
    ensure_dir(output_root)
    save_csv_rows(
        output_root / "vlm.csv",
        rows + [mean_row],
        BASE.AGGREGATE_CSV_FIELDNAMES,
    )
    print(f"Saved {output_root / 'vlm.csv'} with {len(rows)} interactions.")


def main() -> None:
    args = parse_args()
    output_base = (
        Path(args.output_root).resolve()
        if args.output_root
        else physic_eval_root(args.output_mode)
    )
    if args.aggregate_evals:
        aggregate(output_base)
        return

    prompt_template_override = args.prompt_template
    prompt_template_path = BASE.resolve_path(
        prompt_template_override,
        BASE.DEFAULT_PROMPT_TEMPLATE_PATH,
    )
    prompt_template = BASE.load_text(prompt_template_path)
    all_mode = args.interaction_name == "all"
    if all_mode:
        interaction_names = discover_physic_interactions(args.output_mode)
    else:
        interaction_names = [args.interaction_name]

    for interaction_name in interaction_names:
        BASE.evaluate_interaction_vlm(
            interaction_name=interaction_name,
            args=args,
            input_scene_json_path=PROJECT_DIR
            / "01_Generate_SIG"
            / "input_prompts"
            / interaction_name
            / "input_scene.json",
            render_root=physic_eval_root(args.output_mode)
            / interaction_name
            / "semantics",
            output_root=output_base / interaction_name / "vlm",
            prompt_template=prompt_template,
        )

    if all_mode:
        aggregate(output_base)


if __name__ == "__main__":
    main()
