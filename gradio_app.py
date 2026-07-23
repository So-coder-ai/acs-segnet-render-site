"""Hugging Face Spaces entrypoint for ACS-SegNet inference."""

from __future__ import annotations

import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image
import torch


BASE_DIR = Path(__file__).resolve().parent
ACS_REPO_DIR = BASE_DIR / "ACS-SegNet"
SOURCE_CHECKPOINT = BASE_DIR / "checkpoints" / "ACSSegNet_fold1_best.pth"
FP16_CHECKPOINT = BASE_DIR / "checkpoints" / "ACSSegNet_fold1_best.fp16.pth"

os.environ.setdefault("ACS_REPO_DIR", str(ACS_REPO_DIR))
os.environ.setdefault("MODEL_PATHS", str(FP16_CHECKPOINT))
os.environ.setdefault("CACHE_MODELS", "1")
os.environ.setdefault("MODEL_DTYPE", "float16")
os.environ.setdefault("TORCH_NUM_THREADS", "1")
os.environ.setdefault("IMG_SIZE", "256")
os.environ.setdefault("MASK_THRESHOLD", "0.5")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def patch_model_source() -> None:
    model_path = ACS_REPO_DIR / "model.py"
    if not model_path.exists():
        raise FileNotFoundError(f"ACS-SegNet model.py was not found at {model_path}")

    text = model_path.read_text(encoding="utf-8")
    text = text.replace("import matplotlib.pyplot as plt\n", "")
    text = text.replace(
        "self.segformer = SegformerModel.from_pretrained(segformer_variant, config=seg_cfg)",
        "self.segformer = SegformerModel(seg_cfg)",
    )
    model_path.write_text(text, encoding="utf-8")


def ensure_acs_segnet_source() -> None:
    if not (ACS_REPO_DIR / "model.py").exists():
        archive_url = "https://github.com/NimaTorbati/ACS-SegNet/archive/refs/heads/main.zip"
        with tempfile.TemporaryDirectory(prefix="acs-segnet-") as tmp_dir:
            tmp_path = Path(tmp_dir)
            archive_path = tmp_path / "acs-segnet.zip"
            urllib.request.urlretrieve(archive_url, archive_path)
            with zipfile.ZipFile(archive_path) as archive:
                archive.extractall(tmp_path)
            extracted = next(path for path in tmp_path.iterdir() if path.is_dir())
            shutil.move(str(extracted), str(ACS_REPO_DIR))
    patch_model_source()


def ensure_fp16_checkpoint() -> None:
    if FP16_CHECKPOINT.exists():
        return
    if not SOURCE_CHECKPOINT.exists():
        return
    from convert_checkpoint import main as convert_checkpoint

    convert_checkpoint(SOURCE_CHECKPOINT, FP16_CHECKPOINT)


def setup_runtime() -> str | None:
    try:
        ensure_acs_segnet_source()
        ensure_fp16_checkpoint()
        return None
    except Exception as exc:
        return str(exc)


SETUP_ERROR = setup_runtime()

import app as core  # noqa: E402


def selected_message(image: Image.Image | None) -> str:
    if image is None:
        return "Upload an H&E image tile to begin."
    return f"Image uploaded: {image.width}x{image.height}px. Ready to generate the mask."


def run_segmentation(image: Image.Image | None):
    if SETUP_ERROR:
        raise gr.Error(f"Startup failed: {SETUP_ERROR}")
    if image is None:
        raise gr.Error("Please upload an H&E image first.")
    if not FP16_CHECKPOINT.exists() and not SOURCE_CHECKPOINT.exists():
        raise gr.Error("Checkpoint missing. Upload ACSSegNet_fold1_best.pth into the checkpoints folder.")
    if not core.INFERENCE_LOCK.acquire(blocking=False):
        raise gr.Error("A segmentation is already running. Please wait for it to finish.")

    try:
        rgb = image.convert("RGB")
        input_image = np.array(rgb)
        original_h, original_w = input_image.shape[:2]
        max_dim = core.MAX_IMAGE_DIM
        if original_w > max_dim or original_h > max_dim:
            rgb.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
            input_image = np.array(rgb)
            original_h, original_w = input_image.shape[:2]

        resized = np.array(rgb.resize((core.CFG["img_size"], core.CFG["img_size"]), Image.Resampling.LANCZOS))
        tensor = torch.from_numpy((resized.astype(np.float32) / 255.0).transpose(2, 0, 1))
        tensor = tensor.unsqueeze(0).to(core.DEVICE, dtype=core.MODEL_DTYPE)

        prob, models_used = core.predict_probability(tensor)
    except Exception as exc:
        raise gr.Error(str(exc)) from exc
    finally:
        core.INFERENCE_LOCK.release()

    mask_small = (prob > core.CFG["threshold"]).astype(np.uint8) * 255
    mask_full = np.array(Image.fromarray(mask_small).resize((original_w, original_h), Image.Resampling.NEAREST))
    prob_full = np.array(
        Image.fromarray(prob.astype(np.float32), mode="F").resize((original_w, original_h), Image.Resampling.BILINEAR)
    )
    heatmap = core.colorize_probability(prob_full)
    overlay = core.make_overlay(input_image, mask_full)
    foreground = float((mask_full > 0).mean()) * 100.0

    status = (
        f"Done. Foreground: {foreground:.3f}% | Threshold: {core.CFG['threshold']} | "
        f"Checkpoint(s): {models_used} | Device: {core.DEVICE}"
    )
    return (
        Image.fromarray(mask_full),
        Image.fromarray(heatmap),
        Image.fromarray(overlay),
        status,
    )


with gr.Blocks(title="ACS-SegNet H&E Mask Segmentation") as demo:
    gr.Markdown("# ACS-SegNet H&E Mask Segmentation")
    gr.Markdown("Upload one browser-readable H&E tile, then generate a binary mask, probability heatmap, and overlay.")

    with gr.Row():
        image_input = gr.Image(type="pil", label="H&E image")
        with gr.Column():
            status = gr.Textbox(label="Status", value="Upload an H&E image tile to begin.", interactive=False)
            submit = gr.Button("Generate segmentation", variant="primary")

    with gr.Row():
        mask_output = gr.Image(label="Binary mask")
        heatmap_output = gr.Image(label="Probability heatmap")
        overlay_output = gr.Image(label="Overlay preview")

    image_input.change(selected_message, inputs=image_input, outputs=status)
    submit.click(
        run_segmentation,
        inputs=image_input,
        outputs=[mask_output, heatmap_output, overlay_output, status],
        api_name="predict",
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch()
