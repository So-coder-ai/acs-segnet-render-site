---
title: ACS-SegNet
emoji: 🧫
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 4.44.1
app_file: gradio_app.py
pinned: false
---

# ACS-SegNet H&E Segmentation Website

FastAPI website for uploading H&E stained images and generating ACS-SegNet binary masks, probability heatmaps, and overlays.

## Hugging Face Space Deployment

Use a Gradio Space on the free CPU Basic hardware. It has much more memory than Render Free and can run this checkpoint without Docker.

Space repo:

```text
Shiv89dalkd/Acs_Segnet
```

Push this GitHub repo to the Space repo with Git LFS enabled so the `.pth` checkpoint is uploaded too. The Space runs `gradio_app.py`, downloads the ACS-SegNet source code on startup, patches it for offline SegFormer initialization, creates the FP16 checkpoint if needed, and queues requests one at a time.

Optional Space variables:

```text
MODEL_PATHS=checkpoints/ACSSegNet_fold1_best.fp16.pth
CACHE_MODELS=1
MODEL_DTYPE=float16
TORCH_NUM_THREADS=1
IMG_SIZE=256
MASK_THRESHOLD=0.5
HF_HUB_DISABLE_SYMLINKS_WARNING=1
```

## Render Deployment

This project is configured for Render with:

- `render.yaml` Blueprint configuration.
- `build.sh` to install dependencies, clone ACS-SegNet if needed, and patch SegFormer initialization.
- `.python-version` pinned to Python `3.11.9`.
- CPU-only PyTorch in `requirements.txt`, without unused OpenCV, Matplotlib, or
  Albumentations packages.
- An FP16 inference checkpoint generated during the build, then memory-mapped
  while loading, to fit within Render Free's 512 MB RAM limit.
- A single Uvicorn worker and a small upload limit so overlapping uploads cannot
  exhaust the instance.

## Checkpoint

Default deployed checkpoint:

```text
checkpoints/ACSSegNet_fold1_best.pth
```

During a Render build, `build.sh` creates
`checkpoints/ACSSegNet_fold1_best.fp16.pth` and the service loads that smaller
inference checkpoint. Do not add the generated file to Git.

The app supports multiple checkpoints through `MODEL_PATHS`, but a 3-fold ensemble is heavy on CPU hosting. Use one fold unless you upgrade the Render instance.

## Required Render Environment Variables

These are already included in `render.yaml`, but you can also set/override them manually in Render:

```text
PYTHON_VERSION=3.11.9
MODEL_PATHS=checkpoints/ACSSegNet_fold1_best.fp16.pth
CACHE_MODELS=1
MODEL_DTYPE=float16
TORCH_NUM_THREADS=1
IMG_SIZE=256
MASK_THRESHOLD=0.5
HF_HUB_DISABLE_SYMLINKS_WARNING=1
```

## Run Locally

PowerShell:

```powershell
cd C:\Users\narai\Documents\Codex\2026-07-08\build\outputs\acs-segnet
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
$env:MODEL_PATHS="checkpoints\ACSSegNet_fold1_best.fp16.pth"
$env:CACHE_MODELS="1"
$env:TORCH_NUM_THREADS="1"
python -m uvicorn app:app --reload
```

Open `http://127.0.0.1:8000`.

## Deploy on Render

1. Push this folder to GitHub.
2. Keep checkpoint files in Git LFS.
3. In Render, choose `New -> Blueprint`.
4. Select the GitHub repo.
5. Render reads `render.yaml` automatically.
6. Render reads the `free` plan from `render.yaml`. After deployment, visit
   `/health`; it should report `model_dtype: "float16"`.

Expected health fields:

```json
{
  "ok": true,
  "checkpoint_exists": true,
  "acs_repo_exists": true,
  "model_error": null
}
```

## Notes

- The free Render instance has only 0.1 CPU, so the first prediction can be
  slow. It also spins down after 15 minutes without traffic.
- The model is intentionally limited to one checkpoint and one worker. Do not
  add the other two checkpoints to `MODEL_PATHS` on a Free instance.
- Uploaded images are capped at 8 MB and downscaled to 2048 px before output
  generation to keep request memory bounded.
- If a particular CPU reports an unsupported FP16 operation, set
  `MODEL_DTYPE=float32`; that needs a larger paid instance.
