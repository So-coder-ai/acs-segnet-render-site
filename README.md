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
