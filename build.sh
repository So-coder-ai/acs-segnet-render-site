#!/usr/bin/env bash
set -euo pipefail
python -m pip install --upgrade pip --no-cache-dir
pip install --no-cache-dir -r requirements.txt
if [ ! -d "ACS-SegNet" ]; then
  git clone --depth 1 https://github.com/NimaTorbati/ACS-SegNet.git ACS-SegNet
fi
python -c "from pathlib import Path; p=Path('ACS-SegNet/model.py'); text=p.read_text(); text=text.replace('self.segformer = SegformerModel.from_pretrained(segformer_variant, config=seg_cfg)', 'self.segformer = SegformerModel(seg_cfg)'); p.write_text(text)"
python -c "from pathlib import Path; p=Path('ACS-SegNet/unet_decoder/decoder.py'); text=p.read_text(); text=text.replace('import matplotlib.pyplot as plt\n', ''); p.write_text(text)"
python convert_checkpoint.py checkpoints/ACSSegNet_fold1_best.pth checkpoints/ACSSegNet_fold1_best.fp16.pth
ls -lh checkpoints || true
