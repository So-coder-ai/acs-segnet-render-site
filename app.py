import os
import sys
import uuid
import gc
import threading
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "/tmp/acs-segnet-results" if os.getenv("VERCEL") else STATIC_DIR / "results"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ACS_REPO_DIR = Path(os.getenv("ACS_REPO_DIR", BASE_DIR / "ACS-SegNet"))
if ACS_REPO_DIR.exists():
    sys.path.insert(0, str(ACS_REPO_DIR))

CFG = {
    "img_size": int(os.getenv("IMG_SIZE", "256")),
    "num_classes": 1,
    "segformer_var": os.getenv("SEGFORMER_VARIANT", "nvidia/segformer-b2-finetuned-ade-512-512"),
    "resnet_enc": os.getenv("RESNET_ENCODER", "resnet34"),
    "decoder_ch": (256, 128, 64, 32, 16),
    "simple_fusion": int(os.getenv("SIMPLE_FUSION", "1")),
    "threshold": float(os.getenv("MASK_THRESHOLD", "0.5")),
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "1")))
MODEL_DTYPE_NAME = os.getenv("MODEL_DTYPE", "float16").lower()
MODEL_DTYPE = torch.float16 if MODEL_DTYPE_NAME == "float16" else torch.float32
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(8 * 1024 * 1024)))
MAX_IMAGE_DIM = int(os.getenv("MAX_IMAGE_DIM", "2048"))
DEFAULT_CHECKPOINTS = [
    BASE_DIR / "checkpoints" / "ACSSegNet_fold0_best.pth",
    BASE_DIR / "checkpoints" / "ACSSegNet_fold1_best.pth",
    BASE_DIR / "checkpoints" / "ACSSegNet_fold2_best.pth",
]
MODEL_PATHS_ENV = os.getenv("MODEL_PATHS")
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "checkpoints" / "ACSSegNet_best.pth"))
CACHE_MODELS = os.getenv("CACHE_MODELS", "1") == "1"


def project_path(path: Path):
    return path if path.is_absolute() else BASE_DIR / path


def configured_checkpoint_paths():
    if MODEL_PATHS_ENV:
        return [project_path(Path(path.strip())) for path in MODEL_PATHS_ENV.split(",") if path.strip()]
    existing_folds = [path for path in DEFAULT_CHECKPOINTS if path.exists()]
    if existing_folds:
        return existing_folds
    return [project_path(MODEL_PATH)]

app = FastAPI(title="ACS-SegNet H&E Mask Segmentation")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

_models = None
_model_error = None
INFERENCE_LOCK = threading.Lock()


def build_model():
    try:
        from model import DualEncoderUNet
    except Exception as exc:
        raise RuntimeError(
            "Could not import ACS-SegNet. Make sure build.sh cloned "
            "https://github.com/NimaTorbati/ACS-SegNet.git into ./ACS-SegNet."
        ) from exc

    # Construct directly in FP16: constructing FP32 then converting needs
    # roughly 300 MB for this model and can exceed Render free during startup.
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(MODEL_DTYPE)
    try:
        return DualEncoderUNet(
            unet_encoder_name=CFG["resnet_enc"],
            unet_encoder_weights=None,
            segformer_variant=CFG["segformer_var"],
            classes=CFG["num_classes"],
            decoder_channels=CFG["decoder_ch"],
            simple_fusion=CFG["simple_fusion"],
            regression=False,
            in_channels=3,
            model_depth=5,
            IgnoreBottleNeck=False,
            input_size=CFG["img_size"],
        ).to(device=DEVICE)
    finally:
        torch.set_default_dtype(previous_dtype)


def load_state_into_model(checkpoint_path: Path):
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at {checkpoint_path}. Upload your trained .pth files "
            "to checkpoints/ or set MODEL_PATHS on Render."
        )
    model = build_model()
    # mmap prevents a second, fully resident 202 MB checkpoint copy while its
    # tensors are copied into the model. This is essential on Render free.
    try:
        state = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=True)
    except (TypeError, RuntimeError):
        state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    cleaned = {k.replace("module.", "", 1): v for k, v in state.items()}
    cleaned = remap_segformer_stage_keys(cleaned)
    model.load_state_dict(cleaned, strict=True)
    model.eval()
    return model


def remap_segformer_stage_keys(state):
    remapped = {}
    for key, value in state.items():
        new_key = key
        if key.startswith("segformer.stages."):
            parts = key.split(".")
            stage = parts[2]
            rest = ".".join(parts[3:])
            replacements = {
                "patch_embeddings.": f"encoder.patch_embeddings.{stage}.",
                "layer_norm.": f"encoder.layer_norm.{stage}.",
                "blocks.": f"encoder.block.{stage}.",
            }
            for old_prefix, new_prefix in replacements.items():
                if rest.startswith(old_prefix):
                    rest = new_prefix + rest[len(old_prefix):]
                    break
            rest = rest.replace(".layernorm_before.", ".layer_norm_1.")
            rest = rest.replace(".layernorm_after.", ".layer_norm_2.")
            rest = rest.replace(".attention.q_proj.", ".attention.self.query.")
            rest = rest.replace(".attention.k_proj.", ".attention.self.key.")
            rest = rest.replace(".attention.v_proj.", ".attention.self.value.")
            rest = rest.replace(".attention.o_proj.", ".attention.output.dense.")
            rest = rest.replace(".attention.sequence_reduction.sequence_reduction.", ".attention.self.sr.")
            rest = rest.replace(".attention.sequence_reduction.layer_norm.", ".attention.self.layer_norm.")
            rest = rest.replace(".mlp.fc1.", ".mlp.dense1.")
            rest = rest.replace(".mlp.fc2.", ".mlp.dense2.")
            new_key = f"segformer.{rest}"
        remapped[new_key] = value
    return remapped


def get_models():
    global _models, _model_error
    if _models is not None:
        return _models
    if _model_error is not None:
        raise RuntimeError(_model_error)
    try:
        _models = [load_state_into_model(path) for path in configured_checkpoint_paths()]
        if not _models:
            raise RuntimeError("No checkpoints configured. Set MODEL_PATHS or add .pth files to checkpoints/.")
        return _models
    except Exception as exc:
        _model_error = str(exc)
        raise


def predict_probability(tensor):
    checkpoint_paths = configured_checkpoint_paths()
    if CACHE_MODELS:
        models = get_models()
        with torch.inference_mode():
            probs = [torch.sigmoid(model(tensor))[0, 0].detach().cpu().numpy() for model in models]
        return np.mean(probs, axis=0), len(models)

    probs = []
    for checkpoint_path in checkpoint_paths:
        model = load_state_into_model(checkpoint_path)
        with torch.inference_mode():
            probs.append(torch.sigmoid(model(tensor))[0, 0].detach().cpu().numpy())
        del model
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    return np.mean(probs, axis=0), len(checkpoint_paths)


def read_rgb_image(file_bytes: bytes):
    try:
        pil = Image.open(__import__("io").BytesIO(file_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Please upload a valid image file.") from exc
    if pil.width > MAX_IMAGE_DIM or pil.height > MAX_IMAGE_DIM:
        pil.thumbnail((MAX_IMAGE_DIM, MAX_IMAGE_DIM), Image.Resampling.LANCZOS)
    return np.array(pil)


def save_png(path: Path, array: np.ndarray):
    Image.fromarray(array).save(path)


def colorize_probability(prob: np.ndarray):
    # A small built-in heatmap avoids importing OpenCV just for applyColorMap.
    value = np.clip(prob, 0, 1)[..., None]
    return (np.concatenate((
        np.clip(2.2 * value - 0.2, 0, 1),
        np.clip(2.0 * value - 0.65, 0, 1),
        np.clip(1.25 - 2.4 * value, 0, 1),
    ), axis=-1) * 255).astype(np.uint8)


def make_overlay(image: np.ndarray, mask: np.ndarray):
    overlay = image.copy()
    red = np.zeros_like(image)
    red[..., 0] = 255
    alpha = (mask > 0)[..., None].astype(np.float32) * 0.45
    return (overlay * (1 - alpha) + red * alpha).astype(np.uint8)


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "model_path": ", ".join(str(path) for path in configured_checkpoint_paths()),
            "device": str(DEVICE),
            "img_size": CFG["img_size"],
        },
    )


@app.get("/health")
def health():
    checkpoint_paths = configured_checkpoint_paths()
    checkpoint_exists = all(path.exists() for path in checkpoint_paths)
    repo_exists = ACS_REPO_DIR.exists()
    return {
        "ok": checkpoint_exists and repo_exists,
        "device": str(DEVICE),
        "checkpoints": [str(path) for path in checkpoint_paths],
        "checkpoint_exists": checkpoint_exists,
        "acs_repo": str(ACS_REPO_DIR),
        "acs_repo_exists": repo_exists,
        "model_loaded": _models is not None,
        "cache_models": CACHE_MODELS,
        "torch_num_threads": torch.get_num_threads(),
        "model_dtype": MODEL_DTYPE_NAME,
        "model_error": _model_error,
    }


@app.get("/results/{filename}", include_in_schema=False)
def result_file(filename: str):
    path = RESULTS_DIR / filename
    if not path.exists() or path.parent != RESULTS_DIR:
        raise HTTPException(status_code=404, detail="Result file not found.")
    return FileResponse(path)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file, preferably a PNG/JPEG H&E tile.")

    file_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"Image must be {MAX_UPLOAD_BYTES // (1024 * 1024)} MB or smaller.")
    image = read_rgb_image(file_bytes)
    original_h, original_w = image.shape[:2]
    resized = np.array(Image.fromarray(image).resize((CFG["img_size"], CFG["img_size"]), Image.Resampling.LANCZOS))
    tensor = torch.from_numpy((resized.astype(np.float32) / 255.0).transpose(2, 0, 1)).unsqueeze(0).to(DEVICE, dtype=MODEL_DTYPE)

    if not INFERENCE_LOCK.acquire(blocking=False):
        raise HTTPException(
            status_code=429,
            detail="A segmentation is already running. Please wait for it to finish before uploading another image.",
        )
    try:
        prob, models_used = predict_probability(tensor)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    finally:
        INFERENCE_LOCK.release()

    mask_small = (prob > CFG["threshold"]).astype(np.uint8) * 255
    mask_full = np.array(Image.fromarray(mask_small).resize((original_w, original_h), Image.Resampling.NEAREST))
    prob_full = np.array(Image.fromarray(prob.astype(np.float32), mode="F").resize((original_w, original_h), Image.Resampling.BILINEAR))
    heatmap = colorize_probability(prob_full)
    overlay = make_overlay(image, mask_full)

    result_id = uuid.uuid4().hex
    input_path = RESULTS_DIR / f"{result_id}_input.png"
    mask_path = RESULTS_DIR / f"{result_id}_mask.png"
    heat_path = RESULTS_DIR / f"{result_id}_probability.png"
    overlay_path = RESULTS_DIR / f"{result_id}_overlay.png"

    save_png(input_path, image)
    save_png(mask_path, mask_full)
    save_png(heat_path, heatmap)
    save_png(overlay_path, overlay)

    foreground = float((mask_full > 0).mean())
    return JSONResponse(
        {
            "id": result_id,
            "foreground_percent": round(foreground * 100, 3),
            "threshold": CFG["threshold"],
            "models_used": models_used,
            "input": f"/results/{input_path.name}",
            "mask": f"/results/{mask_path.name}",
            "probability": f"/results/{heat_path.name}",
            "overlay": f"/results/{overlay_path.name}",
        }
    )
