# ACS-SegNet H&E Segmentation Website

FastAPI website for uploading H&E stained images and generating ACS-SegNet binary masks, probability heatmaps, and overlays.

## Vercel Notes

This project is configured for Vercel with:

- `vercel.json` for the Python function settings.
- `pyproject.toml` to expose `app:app` and run `vercel_build.py`.
- `.python-version` pinned to Python `3.12`, which is supported by Vercel's Python runtime.
- CPU-only PyTorch in `requirements.txt`.
- Runtime result images written to `/tmp` through the `/results/{filename}` route.

## Checkpoints

For Vercel, use one checkpoint by default to reduce memory and bundle pressure:

```text
checkpoints/ACSSegNet_fold1_best.pth
```

The app still supports a comma-separated `MODEL_PATHS` list, but a 3-fold ensemble is heavy for serverless hosting.

## Required Vercel Environment Variables

Set these in Vercel Project Settings -> Environment Variables:

```text
PYTHON_VERSION=3.12
MODEL_PATHS=checkpoints/ACSSegNet_fold1_best.pth
CACHE_MODELS=0
TORCH_NUM_THREADS=1
IMG_SIZE=256
MASK_THRESHOLD=0.5
VERCEL_SUPPORT_LARGE_FUNCTIONS=1
HF_HUB_DISABLE_SYMLINKS_WARNING=1
```

`VERCEL_SUPPORT_LARGE_FUNCTIONS=1` is important because Vercel's standard Python function bundle limit is 500 MB, while Large Functions support Python bundles up to 5 GB when Fluid compute is enabled.

## Run Locally

PowerShell:

```powershell
cd C:\Users\narai\Downloads\acs-segnet-render-site
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
$env:MODEL_PATHS="checkpoints\ACSSegNet_fold1_best.pth"
$env:CACHE_MODELS="0"
$env:TORCH_NUM_THREADS="1"
uvicorn app:app --reload
```

Open `http://127.0.0.1:8000`.

## Deploy on Vercel

1. Push this folder to GitHub.
2. Keep checkpoint files in Git LFS, or keep only `ACSSegNet_fold1_best.pth` if you want the smallest deployment.
3. Import the GitHub repo in Vercel.
4. Leave Framework Preset as `Other` if Vercel does not auto-detect FastAPI.
5. Add the environment variables above.
6. Deploy.

After deployment, visit:

```text
https://your-vercel-app.vercel.app/health
```

Confirm:

```json
{
  "ok": true,
  "checkpoint_exists": true,
  "acs_repo_exists": true,
  "model_error": null
}
```

## Limits

- Vercel request payloads are limited; use reasonably small image tiles.
- First prediction is slow because the model loads lazily.
- If Vercel still fails due to memory/bundle size, this model is a better fit for Render, Hugging Face Spaces, Railway, or a small VM.
