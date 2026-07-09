# ACS-SegNet H&E Segmentation Website

A Render-ready FastAPI website for uploading H&E stained images and generating ACS-SegNet binary mask segmentation, probability maps, and overlays.

## Checkpoints

This package includes three trained fold checkpoints. By default, the app loads all three and averages their probability maps as a small ensemble:

```text
checkpoints/ACSSegNet_fold0_best.pth
checkpoints/ACSSegNet_fold1_best.pth
checkpoints/ACSSegNet_fold2_best.pth
```

To use different checkpoints, set Render's `MODEL_PATHS` environment variable to a comma-separated list of `.pth` paths.

The app expects the notebook architecture:

- `DualEncoderUNet`
- ResNet34 encoder
- SegFormer-B2 branch
- CBAM/simple fusion enabled
- 256x256 inference input
- one binary output class

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
git clone https://github.com/NimaTorbati/ACS-SegNet.git ACS-SegNet
uvicorn app:app --reload
```

Open `http://127.0.0.1:8000`.

## Deploy on Render

1. Push this folder to a GitHub repository.
2. Keep the three checkpoint files in `checkpoints/`, or store them externally and download them during build.
3. In Render, choose New -> Blueprint and select this repo. Render will read `render.yaml`.
4. Use at least a Standard instance. ACS-SegNet with SegFormer-B2 is heavy for free CPU instances.
5. After deployment, visit `/health` to confirm the ACS-SegNet repo and checkpoints are available.

## Notes

- First inference can be slow because the models load lazily.
- If your checkpoints were trained with `unet_encoder_weights='imagenet'`, inference still uses `None` before loading weights to avoid unnecessary downloads at startup.
- If strict checkpoint loading fails, verify that the training notebook config matches `CFG` in `app.py`.
- Three ACS-SegNet checkpoints are large; make sure your Git host/deployment path supports files around 200 MB each, or use Git LFS/external storage.