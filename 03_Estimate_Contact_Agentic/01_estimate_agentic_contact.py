from __future__ import annotations

import argparse
import base64
from contextlib import ExitStack
import json
import shutil
import time
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

from common import (
    adjusted_intrinsics_for_crop,
    bbox_from_mask,
    build_contact_prompt,
    build_pinhole_intrinsics,
    build_sam3_processor,
    choose_contact_palette,
    classify_nearest_color,
    contact_palette_from_spec,
    crop_array,
    erode_binary_mask,
    fit_bbox_to_image_aspect,
    floor_contact_human_parts,
    get_default_sam3_device,
    keep_largest_components,
    load_binary_mask,
    load_json,
    load_rgb,
    normalize_label,
    normalize_scene_element,
    normalize_overlay_to_canvas,
    pad_bbox,
    resolve_scannet_root,
    resolve_transforms_path,
    run_sam3_text_prompt,
    save_binary_mask,
    save_contact_spec,
    save_contact_visualization,
    save_json,
    save_rgb,
    save_text,
    select_highest_confidence_mask,
    slugify,
    target_object_human_parts,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Agentic contact mask estimation with GPT Image 2 generation "
            "and Gemini evaluation."
        ),
    )
    parser.add_argument("--interaction_name", default="interaction_01")
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--sig-json", default=None)
    parser.add_argument("--scene-image", default=None)
    parser.add_argument("--inpainted-frame", default=None)
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--scannet-root", default=None)
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--vlm-prompt", default=None)
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument(
        "--api-key-file",
        default=str(PROJECT_DIR / ".secrets" / "gemini_api_key"),
    )
    parser.add_argument("--eval-model", default="gemini-3.7-flash")
    parser.add_argument(
        "--openai-api-key-file",
        default=str(PROJECT_DIR / ".secrets" / "openai_api_key"),
    )
    parser.add_argument("--image-model", default="gpt-image-2")
    parser.add_argument("--image-quality", default="medium")
    parser.add_argument("--image-size", default="auto")
    parser.add_argument("--image-retries", type=int, default=3)
    parser.add_argument("--image-retry-sleep-s", type=float, default=8.0)
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--gemini-retries", type=int, default=3)
    parser.add_argument("--gemini-retry-sleep-s", type=float, default=8.0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=72)
    parser.add_argument("--padding-frac", type=float, default=0.25)
    parser.add_argument(
        "--preserve-aspect-ratio-crop",
        action="store_true",
        help="Expand the human crop to the source image aspect ratio.",
    )
    parser.add_argument("--human-sam3-prompt", default="person")
    parser.add_argument("--floor-sam3-prompt", default="floor")
    parser.add_argument("--floor-mask-erode-pixels", type=int, default=3)
    parser.add_argument("--sam3-checkpoint", default=None)
    parser.add_argument("--sam3-bpe-path", default=None)
    parser.add_argument(
        "--sam3-device",
        default="auto",
        help="SAM3 device. Use 'auto' to choose cuda when torch reports it.",
    )
    parser.add_argument("--sam3-confidence-threshold", type=float, default=0.5)
    parser.add_argument("--no-sam3-hf-download", action="store_true")
    parser.add_argument("--color-max-distance", type=float, default=90.0)
    parser.add_argument("--min-component-area", type=int, default=0)
    parser.add_argument("--keep-components", type=int, default=1)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild prepared assets and regenerate contact overlays from round 1.",
    )
    return parser.parse_args(argv)


def resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    output_root = (
        Path(args.outdir).resolve()
        if args.outdir
        else SCRIPT_DIR / "output" / args.interaction_name
    )
    sig_json_path = (
        Path(args.sig_json).resolve()
        if args.sig_json
        else PROJECT_DIR
        / "01_Generate_SIG"
        / "output"
        / args.interaction_name
        / "sig.json"
    )
    scene_image_path = (
        Path(args.scene_image).resolve()
        if args.scene_image
        else sig_json_path.parent / "scene_image.png"
    )
    inpainted_frame_path = (
        Path(args.inpainted_frame).resolve()
        if args.inpainted_frame
        else PROJECT_DIR
        / "02_Generate_Human_Frame"
        / "output"
        / args.interaction_name
        / "inpainted_frame_resized.png"
    )
    input_dir = (
        Path(args.input_dir).resolve()
        if args.input_dir
        else PROJECT_DIR
        / "01_Generate_SIG"
        / "input_prompts"
        / args.interaction_name
    )
    system_prompt_path = (
        Path(args.system_prompt).resolve()
        if args.system_prompt
        else SCRIPT_DIR / "prompt_estimate_contact.md"
    )
    vlm_prompt_path = (
        Path(args.vlm_prompt).resolve()
        if args.vlm_prompt
        else SCRIPT_DIR / "prompt_evaluate_contact.md"
    )
    return {
        "output_root": output_root,
        "assets_dir": output_root / "assets",
        "sig_json": sig_json_path,
        "scene_image": scene_image_path,
        "inpainted_frame": inpainted_frame_path,
        "input_dir": input_dir,
        "system_prompt": system_prompt_path,
        "vlm_prompt": vlm_prompt_path,
    }


def target_object_label(sig_payload: dict[str, Any]) -> str:
    return ", ".join(obj["label"] for obj in sig_payload["target_objects"])


def color_mapping_text(palette: list[dict[str, Any]]) -> str:
    lines = []
    for item in palette:
        rgb = item["rgb"]
        lines.append(
            f"- {item['label']}: {item['hex']} / "
            f"RGB({rgb[0]}, {rgb[1]}, {rgb[2]}) ({item['color_name']})"
        )
    return "\n".join(lines)


def required_contact_facts_text(
    sig_payload: dict[str, Any],
    target_label: str,
    human_parts: list[str],
    palette: list[dict[str, Any]],
) -> str:
    notes_by_part: dict[str, list[str]] = {}
    for edge in sig_payload["interaction_edges"]:
        scene_element = normalize_scene_element(edge["scene_element"])
        if scene_element != "target_object":
            continue
        part = edge["human_part"]
        note = str(edge.get("notes", "")).strip()
        notes_by_part.setdefault(part, [])
        if note:
            notes_by_part[part].append(note)

    palette_by_part = {normalize_label(str(item["part"])): item for item in palette}
    lines: list[str] = []
    for part in human_parts:
        normalized = normalize_label(part)
        color_info = palette_by_part.get(normalized)
        if color_info is None:
            color_text = "the assigned body-part color"
            label = normalized
        else:
            rgb = color_info["rgb"]
            color_text = (
                f"{color_info['hex']} / RGB({rgb[0]}, {rgb[1]}, {rgb[2]}) "
                f"({color_info['color_name']})"
            )
            label = str(color_info["label"])
        notes = notes_by_part.get(normalized) or ["No SIG note provided."]
        lines.append(
            f"- {label}: required contact with target object "
            f"'{target_label}'. Use {color_text}. SIG note: " + " ".join(notes)
        )

    interaction = str(sig_payload.get("interaction", "")).strip()
    if interaction:
        lines.append(f"- Interaction description: {interaction}")
    return "\n".join(lines)


def add_required_contacts_to_prompt(
    prompt: str,
    required_contacts: str,
) -> str:
    section_header = "Required target-object contacts from SIG:"
    if section_header in prompt:
        return prompt.rstrip()
    return (
        prompt.rstrip()
        + "\n\n"
        + section_header
        + "\n"
        + required_contacts.strip()
    )


def generation_color_mapping_text(
    human_parts: list[str],
    palette: list[dict[str, Any]],
) -> str:
    palette_by_part = {
        normalize_label(str(item["part"])): item
        for item in palette
    }
    lines: list[str] = []
    for part in human_parts:
        part_label = normalize_label(part)
        color_info = palette_by_part[part_label]
        rgb = color_info["rgb"]
        lines.append(
            f"{part_label.title()}: The {part_label} is in contact with the "
            f"target object. Use solid {color_info['hex']} / "
            f"RGB({rgb[0]}, {rgb[1]}, {rgb[2]}) "
            f"({color_info['color_name']})."
        )
    return "\n".join(lines)


def render_generation_prompt(
    template: str,
    human_parts: list[str],
    palette: list[dict[str, Any]],
    required_contacts: str,
) -> str:
    prompt = template.strip()
    color_placeholder = "{color_mapping_for_contact_masks}"
    contacts_placeholder = "{required_contacts}"

    if color_placeholder in prompt:
        prompt = prompt.replace(
            color_placeholder,
            generation_color_mapping_text(human_parts, palette),
        )
    else:
        prompt = build_contact_prompt(prompt, human_parts, palette)

    if contacts_placeholder in prompt:
        prompt = prompt.replace(
            contacts_placeholder,
            required_contacts.strip(),
        )
    else:
        prompt = add_required_contacts_to_prompt(prompt, required_contacts)

    return prompt.rstrip()


def open_rgb_image(path: Path) -> Any:
    from PIL import Image

    return Image.open(path).convert("RGB")


def read_provider_api_key(path: Path, provider_name: str) -> str:
    if not path.exists():
        raise FileNotFoundError(
            f"{provider_name} API key file not found: {path}. "
            "Create it with a single API key line."
        )
    key = path.read_text(encoding="utf-8").strip()
    if not key:
        raise ValueError(f"{provider_name} API key file is empty: {path}")
    return key


def resolve_sam3_device(device: str) -> str:
    if device != "auto":
        return device
    return get_default_sam3_device()


def save_gemini_evaluation_artifact(
    artifact_path: Path,
    prompt: str,
    raw_response: str,
) -> None:
    text = (
        "PROMPT\n"
        "======\n"
        f"{prompt.rstrip()}\n\n"
        "RAW RESPONSE\n"
        "============\n"
        f"{raw_response.rstrip()}\n"
    )
    save_text(artifact_path, text)


def request_gemini(
    client: Any,
    *,
    model: str,
    contents: list[Any],
    config: Any,
    retries: int,
    retry_sleep_s: float,
) -> Any:
    from google.genai import errors
    from httpx import TransportError

    attempts = max(1, retries)
    for attempt in range(attempts):
        try:
            return client.models.generate_content(
                model=model, contents=contents, config=config
            )
        except (errors.APIError, TransportError) as exc:
            if isinstance(exc, errors.APIError) and exc.code != 429 and exc.code < 500:
                raise
            if attempt + 1 == attempts:
                raise
            print(f"Gemini request failed ({attempt + 1}/{attempts}): {exc}")
            time.sleep(max(0.0, retry_sleep_s))


def gemini_generate_json(
    api_key: str,
    model: str,
    user_prompt: str,
    image_paths: list[Path],
    temperature: float,
    seed: int,
    max_output_tokens: int,
    artifact_path: Path,
    retries: int,
    retry_sleep_s: float,
) -> str:
    from google import genai
    from google.genai import types

    config = types.GenerateContentConfig(
        temperature=temperature,
        seed=seed,
        max_output_tokens=max_output_tokens,
        response_mime_type="application/json",
        response_schema={
            "type": "object",
            "properties": {
                "done": {"type": "boolean"},
                "correction_instruction": {"type": "string"},
            },
            "required": ["done", "correction_instruction"],
        },
    )
    with genai.Client(api_key=api_key) as client:
        response = request_gemini(
            client,
            model=model,
            contents=[user_prompt] + [open_rgb_image(path) for path in image_paths],
            config=config,
            retries=retries,
            retry_sleep_s=retry_sleep_s,
        )
    content = response.text or ""
    save_gemini_evaluation_artifact(artifact_path, user_prompt, content)
    return content


def parse_json_response(raw_response: str) -> dict[str, Any]:
    parsed = json.loads(raw_response)
    if not isinstance(parsed, dict):
        raise ValueError("VLM response JSON must be an object.")
    return parsed


def render_vlm_prompt(
    template: str,
    target_label: str,
    palette: list[dict[str, Any]],
    required_contacts: str,
) -> str:
    return (
        template.replace("{target_object}", target_label)
        .replace("{color_mapping}", color_mapping_text(palette))
        .replace("{required_contacts}", required_contacts)
    )


def validate_vlm_evaluation(parsed: dict[str, Any]) -> dict[str, Any]:
    if type(parsed["done"]) is not bool:
        raise ValueError("VLM done must be a JSON boolean.")
    if not isinstance(parsed["correction_instruction"], str):
        raise ValueError("VLM correction_instruction must be a string.")
    return {
        "done": parsed["done"],
        "correction_instruction": parsed["correction_instruction"].strip(),
    }


def evaluate_round_gemini(
    args: argparse.Namespace,
    api_key: str,
    vlm_prompt_template: str,
    target_label: str,
    palette: list[dict[str, Any]],
    required_contacts: str,
    reference_path: Path,
    canvas_path: Path,
    composite_path: Path,
    artifact_path: Path,
) -> dict[str, Any]:
    image_paths = [reference_path, canvas_path, composite_path]
    prompt = render_vlm_prompt(
        vlm_prompt_template,
        target_label,
        palette,
        required_contacts,
    )
    raw_response = gemini_generate_json(
        api_key=api_key,
        model=args.eval_model,
        user_prompt=prompt,
        image_paths=image_paths,
        temperature=float(args.temperature),
        seed=int(args.seed),
        max_output_tokens=int(args.max_output_tokens),
        artifact_path=artifact_path,
        retries=args.gemini_retries,
        retry_sleep_s=args.gemini_retry_sleep_s,
    )
    parsed = parse_json_response(raw_response)
    return validate_vlm_evaluation(parsed)



def prepare_assets(
    args: argparse.Namespace,
    paths: dict[str, Path],
) -> dict[str, Any]:
    output_root = paths["output_root"]
    assets_dir = paths["assets_dir"]
    reference_crop_path = assets_dir / "reference_inpainted_crop.png"
    canvas_crop_path = assets_dir / "target_scene_crop.png"
    floor_mask_crop_path = output_root / "floor_mask_crop.png"
    contact_spec_path = output_root / "contact_spec.json"
    base_prompt_path = assets_dir / "base_prompt.md"
    sig_payload = load_json(paths["sig_json"])
    target_label = target_object_label(sig_payload)
    human_parts = target_object_human_parts(sig_payload)
    floor_parts = floor_contact_human_parts(sig_payload)

    if (
        not args.overwrite
        and reference_crop_path.exists()
        and canvas_crop_path.exists()
        and contact_spec_path.exists()
        and base_prompt_path.exists()
        and (not floor_parts or floor_mask_crop_path.exists())
    ):
        palette = contact_palette_from_spec(
            contact_spec_path,
            human_parts,
        )
        required_contacts = required_contact_facts_text(
            sig_payload=sig_payload,
            target_label=target_label,
            human_parts=human_parts,
            palette=palette,
        )
        base_prompt = base_prompt_path.read_text(encoding="utf-8")
        return {
            "sig_payload": sig_payload,
            "target_label": target_label,
            "human_parts": human_parts,
            "floor_parts": floor_parts,
            "reference_crop_path": reference_crop_path,
            "canvas_crop_path": canvas_crop_path,
            "floor_mask_crop_path": floor_mask_crop_path,
            "contact_spec_path": contact_spec_path,
            "base_prompt_path": base_prompt_path,
            "base_prompt": base_prompt,
            "vlm_prompt_path": paths["vlm_prompt"],
            "vlm_prompt_template": paths["vlm_prompt"].read_text(
                encoding="utf-8"
            ),
            "palette": palette,
            "required_contacts": required_contacts,
        }

    output_root.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)
    input_payload = load_json(paths["input_dir"] / "input_scene.json")
    system_prompt = paths["system_prompt"].read_text(encoding="utf-8")
    vlm_prompt_template = paths["vlm_prompt"].read_text(encoding="utf-8")

    scene_rgb = load_rgb(paths["scene_image"])
    inpainted_rgb = load_rgb(paths["inpainted_frame"])
    if scene_rgb.shape != inpainted_rgb.shape:
        raise ValueError(
            "Scene image and resized inpainted frame must have the same "
            "shape: "
            f"scene={scene_rgb.shape}, inpainted={inpainted_rgb.shape}"
        )
    image_h, image_w = scene_rgb.shape[:2]

    from PIL import Image

    sam3_processor = build_sam3_processor(
        checkpoint_path=(
            Path(args.sam3_checkpoint).resolve()
            if args.sam3_checkpoint
            else None
        ),
        bpe_path=(
            Path(args.sam3_bpe_path).resolve() if args.sam3_bpe_path else None
        ),
        device=resolve_sam3_device(str(args.sam3_device)),
        confidence_threshold=args.sam3_confidence_threshold,
        allow_hf_download=not args.no_sam3_hf_download,
    )
    human_predictions = run_sam3_text_prompt(
        processor=sam3_processor,
        image_rgb=Image.fromarray(inpainted_rgb),
        prompt=args.human_sam3_prompt,
    )
    selected_human = select_highest_confidence_mask(human_predictions)
    human_mask = selected_human["mask"]
    if human_mask.shape != (image_h, image_w):
        raise ValueError(
            "SAM3 human mask shape does not match image shape: "
            f"mask={human_mask.shape[::-1]}, image={image_w}x{image_h}"
        )
    human_bbox = bbox_from_mask(human_mask)
    if human_bbox is None:
        raise ValueError(
            f"SAM3 returned an empty human mask for prompt "
            f"'{args.human_sam3_prompt}'."
        )
    crop_xyxy = pad_bbox(
        human_bbox,
        image_width=image_w,
        image_height=image_h,
        padding_frac=args.padding_frac,
    )
    if args.preserve_aspect_ratio_crop:
        crop_xyxy = fit_bbox_to_image_aspect(
            crop_xyxy,
            image_width=image_w,
            image_height=image_h,
        )
    reference_crop = crop_array(inpainted_rgb, crop_xyxy)
    canvas_crop = crop_array(scene_rgb, crop_xyxy)

    save_rgb(reference_crop_path, reference_crop)
    save_rgb(canvas_crop_path, canvas_crop)
    if floor_parts:
        floor_predictions = run_sam3_text_prompt(
            processor=sam3_processor,
            image_rgb=Image.fromarray(scene_rgb),
            prompt=args.floor_sam3_prompt,
        )
        selected_floor = select_highest_confidence_mask(floor_predictions)
        floor_mask = selected_floor["mask"]
        if floor_mask.shape != (image_h, image_w):
            raise ValueError(
                "SAM3 floor mask shape does not match image shape: "
                f"mask={floor_mask.shape[::-1]}, image={image_w}x{image_h}"
            )
        floor_mask_crop = crop_array(floor_mask, crop_xyxy)
        if not floor_mask_crop.any():
            raise ValueError(
                f"SAM3 returned an empty floor mask inside the contact crop "
                f"for prompt '{args.floor_sam3_prompt}'."
            )
        floor_mask_crop = erode_binary_mask(
            floor_mask_crop,
            int(args.floor_mask_erode_pixels),
        )
        save_binary_mask(floor_mask_crop_path, floor_mask_crop)

    scannet_root = resolve_scannet_root(PROJECT_DIR, args.scannet_root)
    transforms_path = resolve_transforms_path(
        scannet_root, input_payload["scene_context"]
    )
    transforms_payload = load_json(transforms_path)
    source_intrinsics, metadata_w, metadata_h = build_pinhole_intrinsics(
        transforms_payload,
    )
    if (metadata_w, metadata_h) != (image_w, image_h):
        raise ValueError(
            "Scene image shape does not match ScanNet++ camera metadata: "
            f"image={image_w}x{image_h}, metadata={metadata_w}x{metadata_h}"
        )
    contact_intrinsics = adjusted_intrinsics_for_crop(
        source_intrinsics,
        crop_xyxy,
    )
    palette = choose_contact_palette(
        human_parts=human_parts,
        scene_rgb=canvas_crop,
        avoid_mask=None,
    )
    required_contacts = required_contact_facts_text(
        sig_payload=sig_payload,
        target_label=target_label,
        human_parts=human_parts,
        palette=palette,
    )
    save_contact_spec(contact_spec_path, contact_intrinsics, palette)
    base_prompt = render_generation_prompt(
        template=system_prompt,
        human_parts=human_parts,
        palette=palette,
        required_contacts=required_contacts,
    )
    save_text(base_prompt_path, base_prompt + "\n")

    return {
        "sig_payload": sig_payload,
        "target_label": target_label,
        "human_parts": human_parts,
        "floor_parts": floor_parts,
        "reference_crop_path": reference_crop_path,
        "canvas_crop_path": canvas_crop_path,
        "floor_mask_crop_path": floor_mask_crop_path,
        "contact_spec_path": contact_spec_path,
        "base_prompt_path": base_prompt_path,
        "base_prompt": base_prompt,
        "vlm_prompt_path": paths["vlm_prompt"],
        "vlm_prompt_template": vlm_prompt_template,
        "palette": palette,
        "required_contacts": required_contacts,
    }


def build_round_prompt(
    base_prompt: str,
    correction_instruction: str | None,
) -> str:
    if not correction_instruction:
        return base_prompt.rstrip() + "\n"
    return (
        base_prompt.rstrip()
        + "\n\n"
        + "Correction Instructions from an Evaluator:\n"
        + correction_instruction.strip()
        + "\n"
    )


def write_round_prompt_package(
    prompt_dir: Path,
    prompt: str,
    reference_path: Path,
    canvas_path: Path,
    previous_composite_path: Path | None,
) -> Path:
    prompt_dir.mkdir(parents=True, exist_ok=True)
    reference_copy = prompt_dir / "01_reference_image.png"
    canvas_copy = prompt_dir / "02_canvas_image.png"
    shutil.copy2(reference_path, reference_copy)
    shutil.copy2(canvas_path, canvas_copy)

    if previous_composite_path is not None:
        previous_copy = prompt_dir / "03_previous_composite.png"
        shutil.copy2(previous_composite_path, previous_copy)
    else:
        previous_copy = None

    prompt_path = prompt_dir / "prompt.md"
    save_text(
        prompt_path,
        prompt.rstrip() + "\n",
    )
    return prompt_path


def verify_generated_image(generated_path: Path) -> None:
    from PIL import Image

    if not generated_path.exists():
        raise FileNotFoundError(f"Generated image not found: {generated_path}")
    try:
        with Image.open(generated_path) as image:
            image.verify()
    except Exception as exc:
        raise RuntimeError(
            f"Generated image is not readable ({exc}): {generated_path}"
        ) from exc


def run_openai_image_edit_once(
    client: Any,
    model: str,
    quality: str,
    size: str,
    prompt: str,
    image_paths: list[Path],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        images = [stack.enter_context(path.open("rb")) for path in image_paths]
        response = client.images.edit(
            model=model,
            image=images,
            prompt=prompt,
            quality=quality,
            size=size,
            output_format="png",
        )
    image_b64 = response.data[0].b64_json
    if not image_b64:
        raise RuntimeError("OpenAI image edit returned no image data.")
    output_path.write_bytes(base64.b64decode(image_b64))
    verify_generated_image(output_path)


def run_openai_image_edit(
    args: argparse.Namespace,
    api_key: str,
    prompt: str,
    canvas_path: Path,
    reference_path: Path,
    previous_composite_path: Path | None,
    output_path: Path,
    prompt_artifact_path: Path,
) -> None:
    from openai import OpenAI, APIConnectionError, APIStatusError

    image_paths = [reference_path, canvas_path]
    if previous_composite_path is not None:
        image_paths.append(previous_composite_path)
    prompt = prompt.rstrip() + "\n"
    save_text(prompt_artifact_path, prompt)
    attempts = max(1, args.image_retries)
    with OpenAI(api_key=api_key, max_retries=0) as client:
        for attempt in range(attempts):
            try:
                run_openai_image_edit_once(
                    client=client,
                    model=args.image_model,
                    quality=args.image_quality,
                    size=args.image_size,
                    prompt=prompt,
                    image_paths=image_paths,
                    output_path=output_path,
                )
                return
            except (APIConnectionError, APIStatusError) as exc:
                if (
                    isinstance(exc, APIStatusError)
                    and exc.status_code not in {408, 409, 429}
                    and exc.status_code < 500
                ):
                    raise
                if attempt + 1 == attempts:
                    raise
                print(f"Image request failed ({attempt + 1}/{attempts}): {exc}")
                time.sleep(max(0.0, args.image_retry_sleep_s))


def extract_contact_masks_from_overlay(
    overlay_path: Path,
    resized_overlay_path: Path,
    canvas_path: Path,
    human_parts: list[str],
    palette: list[dict[str, Any]],
    color_max_distance: float,
    min_component_area: int,
    keep_components: int,
) -> list[tuple[str, Any]]:
    overlay_rgb, _canvas_hw = normalize_overlay_to_canvas(
        overlay_path=overlay_path,
        canvas_path=canvas_path,
        resized_overlay_path=resized_overlay_path,
    )
    palette_by_part = {
        normalize_label(str(item["part"])): item for item in palette
    }
    contact_colors = [
        tuple(
            int(value)
            for value in palette_by_part[normalize_label(part)]["rgb"]
        )
        for part in human_parts
    ]
    nearest_color, color_accept = classify_nearest_color(
        overlay_rgb=overlay_rgb,
        target_colors_rgb=contact_colors,
        color_max_distance=color_max_distance,
    )

    masks_by_part: list[tuple[str, Any]] = []
    for part_idx, part in enumerate(human_parts):
        mask = (nearest_color == part_idx) & color_accept
        mask = keep_largest_components(
            mask,
            min_area=min_component_area,
            keep_components=keep_components,
        )
        masks_by_part.append((part, mask))

    return masks_by_part


def save_final_contact_masks(
    contact_masks_dir: Path,
    masks_by_part: list[tuple[str, Any]],
    palette: list[dict[str, Any]],
    floor_parts: list[str],
    floor_mask_path: Path,
    color_max_distance: float,
    min_component_area: int,
    keep_components: int,
) -> None:
    if contact_masks_dir.exists():
        shutil.rmtree(contact_masks_dir)
    contact_masks_dir.mkdir(parents=True, exist_ok=True)
    for part, mask in masks_by_part:
        save_binary_mask(contact_masks_dir / f"{slugify(part)}.png", mask)
    floor_mask_written = False
    if floor_parts:
        expected_hw = masks_by_part[0][1].shape if masks_by_part else None
        floor_mask = load_binary_mask(floor_mask_path, expected_hw=expected_hw)
        for part in floor_parts:
            save_binary_mask(
                contact_masks_dir / f"{slugify(part)}.png",
                floor_mask,
            )
        floor_mask_written = True
    saved_floor_mask_path = (
        str(floor_mask_path) if floor_mask_written else None
    )
    save_json(
        contact_masks_dir / "metadata.json",
        {
            "palette": palette,
            "color_max_distance": float(color_max_distance),
            "min_component_area": int(min_component_area),
            "keep_components": int(keep_components),
            "floor_parts": floor_parts,
            "floor_mask_path": saved_floor_mask_path,
        },
    )


def recompose_masks(
    canvas_path: Path,
    masks_by_part: list[tuple[str, Any]],
    palette: list[dict[str, Any]],
    output_path: Path,
) -> None:
    import numpy as np

    canvas_rgb = load_rgb(canvas_path)
    composite = canvas_rgb.copy()
    palette_by_part = {
        normalize_label(str(item["part"])): tuple(
            int(value) for value in item["rgb"]
        )
        for item in palette
    }
    for part, mask in masks_by_part:
        normalized = normalize_label(part)
        if normalized not in palette_by_part:
            continue
        composite[mask.astype(bool)] = np.asarray(
            palette_by_part[normalized], dtype=np.uint8
        )
    save_rgb(output_path, composite)


def publish_round_outputs(
    output_root: Path,
    composite_path: Path,
    canvas_path: Path,
    masks_by_part: list[tuple[str, Any]],
    palette: list[dict[str, Any]],
    floor_parts: list[str],
    floor_mask_path: Path,
    color_max_distance: float,
    min_component_area: int,
    keep_components: int,
    summary: dict[str, Any],
) -> None:
    root_masks_dir = output_root / "contact_masks"
    save_final_contact_masks(
        contact_masks_dir=root_masks_dir,
        masks_by_part=masks_by_part,
        palette=palette,
        floor_parts=floor_parts,
        floor_mask_path=floor_mask_path,
        color_max_distance=color_max_distance,
        min_component_area=min_component_area,
        keep_components=keep_components,
    )
    shutil.copy2(composite_path, output_root / "contact_overlay.png")
    save_contact_visualization(
        canvas_rgb=load_rgb(canvas_path),
        masks_by_part=masks_by_part,
        palette=palette,
        visualization_path=output_root / "contact_overlay_visualization.png",
    )
    save_json(output_root / "agentic_summary.json", summary)


def run_agentic_loop(args: argparse.Namespace, assets: dict[str, Any]) -> int:
    if args.max_rounds < 1:
        raise ValueError("--max-rounds must be at least 1")
    output_root = resolve_paths(args)["output_root"]
    rounds_root = output_root / "rounds"
    rounds_root.mkdir(parents=True, exist_ok=True)
    gemini_api_key = read_provider_api_key(
        Path(args.api_key_file).resolve(),
        "Gemini",
    )
    openai_api_key = read_provider_api_key(
        Path(args.openai_api_key_file).resolve(),
        "OpenAI",
    )
    evaluator_model = str(args.eval_model)

    base_prompt = str(assets["base_prompt"])
    vlm_prompt_template = str(assets["vlm_prompt_template"])
    human_parts = list(assets["human_parts"])
    floor_parts = list(assets["floor_parts"])
    palette = list(assets["palette"])
    required_contacts = str(assets["required_contacts"])
    reference_path = Path(assets["reference_crop_path"])
    canvas_path = Path(assets["canvas_crop_path"])
    floor_mask_path = Path(assets["floor_mask_crop_path"])
    correction_instruction: str | None = None

    for round_index in range(1, int(args.max_rounds) + 1):
        round_dir = rounds_root / f"round_{round_index:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        round_prompt_dir = round_dir / "prompt"
        generated_path = round_dir / "generated_contact_overlay.png"
        resized_overlay_path = round_dir / "generated_contact_overlay_resized.png"
        composite_path = round_dir / "composite.png"
        openai_prompt_artifact_path = round_dir / "openai_generation.txt"
        evaluation_artifact_path = round_dir / "gemini_evaluation.txt"
        stale_round_masks_dir = round_dir / "contact_masks"
        if stale_round_masks_dir.exists():
            shutil.rmtree(stale_round_masks_dir)

        prompt = build_round_prompt(base_prompt, correction_instruction)
        previous_composite = (
            rounds_root / f"round_{round_index - 1:02d}" / "composite.png"
            if round_index > 1
            else None
        )
        prompt_path = write_round_prompt_package(
            prompt_dir=round_prompt_dir,
            prompt=prompt,
            reference_path=reference_path,
            canvas_path=canvas_path,
            previous_composite_path=previous_composite,
        )
        print("\n" + "=" * 72)
        print(f"Round {round_index:02d}/{args.max_rounds}")
        print(f"Prompt: {prompt_path}")
        print(f"OpenAI image 1: {round_prompt_dir / '01_reference_image.png'}")
        print(f"OpenAI image 2: {round_prompt_dir / '02_canvas_image.png'}")
        if previous_composite is not None:
            print(f"OpenAI image 3: {round_prompt_dir / '03_previous_composite.png'}")
        if generated_path.exists() and not args.overwrite:
            print(
                "Skipping OpenAI image edit; found existing generated "
                f"overlay: {generated_path}"
            )
            verify_generated_image(generated_path)
        else:
            if generated_path.exists():
                print(f"Overwriting generated overlay: {generated_path}")
            run_openai_image_edit(
                args=args,
                api_key=openai_api_key,
                prompt=prompt_path.read_text(encoding="utf-8"),
                canvas_path=canvas_path,
                reference_path=reference_path,
                previous_composite_path=previous_composite,
                output_path=generated_path,
                prompt_artifact_path=openai_prompt_artifact_path,
            )
            print(f"Wrote OpenAI generated overlay: {generated_path}")

        contact_masks = extract_contact_masks_from_overlay(
            overlay_path=generated_path,
            resized_overlay_path=resized_overlay_path,
            canvas_path=canvas_path,
            human_parts=human_parts,
            palette=palette,
            color_max_distance=float(args.color_max_distance),
            min_component_area=int(args.min_component_area),
            keep_components=int(args.keep_components),
        )
        recompose_masks(
            canvas_path=canvas_path,
            masks_by_part=contact_masks,
            palette=palette,
            output_path=composite_path,
        )

        print(f"Composite for VLM evaluation: {composite_path}")
        evaluation = evaluate_round_gemini(
            args=args,
            api_key=gemini_api_key,
            vlm_prompt_template=vlm_prompt_template,
            target_label=assets["target_label"],
            palette=palette,
            required_contacts=required_contacts,
            reference_path=reference_path,
            canvas_path=canvas_path,
            composite_path=composite_path,
            artifact_path=evaluation_artifact_path,
        )
        print(f"VLM result: done={evaluation['done']}")
        if evaluation["correction_instruction"]:
            print(f"Correction: {evaluation['correction_instruction']}")

        summary = {
            "interaction_name": args.interaction_name,
            "done": bool(evaluation["done"]),
            "accepted_round": round_index if evaluation["done"] else None,
            "latest_round": round_index,
            "max_rounds": int(args.max_rounds),
            "provider": "gemini",
            "model": evaluator_model,
            "image_provider": "openai",
            "image_model": args.image_model,
            "image_quality": args.image_quality,
            "image_size": args.image_size,
            "evaluator_provider": "gemini",
            "evaluator_model": evaluator_model,
            "target_object": assets["target_label"],
            "human_parts": human_parts,
            "floor_parts": floor_parts,
            "palette": palette,
            "required_contacts": required_contacts,
            "composite_includes_floor_contacts": False,
            "reference_crop": str(reference_path),
            "canvas_crop": str(canvas_path),
            "contact_spec": str(assets["contact_spec_path"]),
            "vlm_prompt": str(assets["vlm_prompt_path"]),
            "latest_evaluation": evaluation,
        }

        if evaluation["done"]:
            print(f"Accepted contact masks at round {round_index:02d}.")
            break
        correction_instruction = evaluation["correction_instruction"] or (
            "The previous result was not accepted. Improve the colored "
            "contact "
            "mask locations and sizes while preserving the original canvas."
        )

    if not evaluation["done"]:
        summary["stopped_reason"] = "max_rounds_reached"
        print(f"Reached max rounds ({args.max_rounds}) without VLM acceptance.")
    publish_round_outputs(
        output_root=output_root,
        composite_path=composite_path,
        canvas_path=canvas_path,
        masks_by_part=contact_masks,
        palette=palette,
        floor_parts=floor_parts,
        floor_mask_path=floor_mask_path,
        color_max_distance=args.color_max_distance,
        min_component_area=args.min_component_area,
        keep_components=args.keep_components,
        summary=summary,
    )
    return 0 if evaluation["done"] else 1



def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = resolve_paths(args)
    assets = prepare_assets(args, paths)
    print(f"Wrote/loaded reference crop: {assets['reference_crop_path']}")
    print(f"Wrote/loaded canvas crop: {assets['canvas_crop_path']}")
    if assets["floor_parts"]:
        print(f"Wrote/loaded floor mask crop: {assets['floor_mask_crop_path']}")
    print(f"Wrote/loaded contact spec: {assets['contact_spec_path']}")
    print(f"Wrote/loaded base prompt: {assets['base_prompt_path']}")
    print(f"Loaded VLM prompt: {assets['vlm_prompt_path']}")
    return run_agentic_loop(args, assets)


if __name__ == "__main__":
    raise SystemExit(main())
