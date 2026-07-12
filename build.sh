#!/usr/bin/env bash
set -euo pipefail
python -m pip install --upgrade pip
pip install -r requirements.txt
if [ ! -d "ACS-SegNet" ]; then
  git clone https://github.com/NimaTorbati/ACS-SegNet.git ACS-SegNet
fi
python -c "from pathlib import Path; p=Path('ACS-SegNet/model.py'); text=p.read_text(); text=text.replace('self.segformer = SegformerModel.from_pretrained(segformer_variant, config=seg_cfg)', 'self.segformer = SegformerModel(seg_cfg)'); p.write_text(text)"
ls -lh checkpoints || true
