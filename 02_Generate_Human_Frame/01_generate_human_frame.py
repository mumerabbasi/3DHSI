from __future__ import annotations

import argparse
import json
import shutil
import time
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the prompt, generate human-frame candidates, and select the best frame."
    )
    parser.add_argument("--interaction_name", default="interaction_01")
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--sig-json", type=Path)
    parser.add_argument("--scene-image", type=Path)
    parser.add_argument("--system-prompt", type=Path)
    parser.add_argument("--selector-prompt", type=Path)
    parser.add_argument(
        "--api-key-file", type=Path, default=PROJECT_DIR / ".secrets" / "gemini_api_key"
    )
    parser.add_argument("--image-model", default="gemini-3.1-flash-image")
    parser.add_argument("--selection-model", default="gemini-3.7-flash")
    parser.add_argument("--num-candidates", type=int, default=5)
    parser.add_argument("--selection-retries", type=int, default=3)
    parser.add_argument("--selection-retry-sleep-s", type=float, default=8.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def read_api_key(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Gemini API key file not found: {path}")
    key = path.read_text(encoding="utf-8").strip()
    if not key:
        raise ValueError(f"Gemini API key file is empty: {path}")
    return key


def open_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def extract_response_image(response: Any) -> Image.Image:
    for part in response.parts or []:
        if part.inline_data is not None:
            with Image.open(BytesIO(part.inline_data.data)) as image:
                return image.convert("RGB")
    raise RuntimeError("Gemini response did not include an image.")


def generate_candidate(api_key: str, model: str, prompt: str, scene: Image.Image) -> Image.Image:
    from google import genai

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model=model, contents=[prompt, scene])
    return extract_response_image(response)


def build_selection_prompt(template: str, interaction: str, candidates: list[Path]) -> str:
    listing = "\n".join(
        f"Image {index}: {path.name}" for index, path in enumerate(candidates, 1)
    )
    return (
        template.replace("{candidate_listing}", listing)
        .replace("{interaction}", interaction.strip())
        .strip()
    )


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


def save_selection_artifact(path: Path, prompt: str, response: str) -> None:
    path.write_text(
        "PROMPT\n======\n"
        f"{prompt.rstrip()}\n\n"
        "RAW RESPONSE\n============\n"
        f"{response.rstrip()}\n",
        encoding="utf-8",
    )


def parse_selection_response(raw_response: str) -> dict[str, Any]:
    parsed = json.loads(raw_response)
    if not isinstance(parsed, dict):
        raise ValueError("Gemini selection response must be a JSON object.")
    return parsed


def select_candidate(
    api_key: str,
    model: str,
    prompt: str,
    candidates: list[Path],
    artifact_path: Path,
    retries: int,
    retry_sleep_s: float,
) -> dict[str, Any]:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        temperature=0.0,
        seed=1,
        maxOutputTokens=512,
        responseMimeType="application/json",
    )
    response = request_gemini(
        client,
        model=model,
        contents=[prompt] + [open_rgb_image(path) for path in candidates],
        config=config,
        retries=retries,
        retry_sleep_s=retry_sleep_s,
    )
    raw_response = response.text or ""
    save_selection_artifact(artifact_path, prompt, raw_response)
    return parse_selection_response(raw_response)



def resize_to_scene(image: Image.Image, scene_size: tuple[int, int]) -> Image.Image:
    if image.size == scene_size:
        return image
    return ImageOps.fit(
        image,
        scene_size,
        method=Image.Resampling.BICUBIC,
        centering=(0.5, 0.5),
    )


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    if args.num_candidates < 1:
        raise ValueError("--num-candidates must be at least 1")

    output_root = (args.outdir or SCRIPT_DIR / "output" / args.interaction_name).resolve()
    sig_path = (
        args.sig_json
        or PROJECT_DIR / "01_Generate_SIG" / "output" / args.interaction_name / "sig.json"
    ).resolve()
    scene_path = (args.scene_image or sig_path.parent / "scene_image.png").resolve()
    system_prompt_path = (
        args.system_prompt or SCRIPT_DIR / "system_prompt_human_inpaint.md"
    ).resolve()
    selector_prompt_path = (
        args.selector_prompt or SCRIPT_DIR / "prompt_select_best_human_frame.md"
    ).resolve()

    sig = json.loads(sig_path.read_text(encoding="utf-8"))
    interaction = str(sig["interaction"])
    system_prompt = system_prompt_path.read_text(encoding="utf-8")
    scene = open_rgb_image(scene_path)

    prompt_dir = output_root / "prompt"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    prompt = f"{system_prompt.strip()}\n\nInteraction:\n{interaction}"
    (prompt_dir / "prompt.md").write_text(prompt + "\n", encoding="utf-8")
    prompt_scene_path = prompt_dir / "scene_image.png"
    if scene_path != prompt_scene_path.resolve():
        shutil.copy2(scene_path, prompt_scene_path)

    inpainted_path = output_root / "inpainted_frame.png"
    resized_path = output_root / "inpainted_frame_resized.png"
    if inpainted_path.exists() and not args.overwrite:
        resize_to_scene(open_rgb_image(inpainted_path), scene.size).save(resized_path)
        print(f"Reused existing human frame: {inpainted_path}")
        print(f"Wrote resized human frame: {resized_path}")
        return inpainted_path

    api_key = read_api_key(args.api_key_file.resolve())
    frames_dir = output_root / "human_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    candidates = [frames_dir / f"frame_{index:02d}.png" for index in range(args.num_candidates)]
    for candidate in candidates:
        if candidate.exists() and not args.overwrite:
            print(f"Reusing candidate: {candidate}")
            continue
        generated = generate_candidate(api_key, args.image_model, prompt, scene)
        generated.save(candidate)
        print(f"Generated candidate: {candidate}")

    template = selector_prompt_path.read_text(encoding="utf-8")
    selection_prompt = build_selection_prompt(template, interaction, candidates)
    selection = select_candidate(
        api_key=api_key,
        model=args.selection_model,
        prompt=selection_prompt,
        candidates=candidates,
        artifact_path=output_root / "gemini_select_best_human_frame.txt",
        retries=args.selection_retries,
        retry_sleep_s=args.selection_retry_sleep_s,
    )
    candidate_by_name = {path.name: path for path in candidates}
    selected_name = selection.get("selected_frame")
    if selected_name not in candidate_by_name:
        raise ValueError(
            f"Gemini selected invalid frame {selected_name!r}; "
            f"expected one of: {', '.join(candidate_by_name)}"
        )
    selected_path = candidate_by_name[selected_name]
    shutil.copy2(selected_path, inpainted_path)
    resize_to_scene(open_rgb_image(selected_path), scene.size).save(resized_path)
    (output_root / "selected_human_frame.json").write_text(
        json.dumps(
            {
                "selected_frame": selected_name,
                "reason": str(selection.get("reason", "")).strip(),
                "selected_path": str(selected_path),
                "image_model": args.image_model,
                "selection_model": args.selection_model,
                "candidates": [path.name for path in candidates],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Selected human frame: {selected_name}")
    print(f"Wrote human frame: {inpainted_path}")
    print(f"Wrote resized human frame: {resized_path}")
    return inpainted_path


if __name__ == "__main__":
    main()
