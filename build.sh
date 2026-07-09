#!/usr/bin/env bash
set -euo pipefail
python -m pip install --upgrade pip
pip install -r requirements.txt
if [ ! -d "ACS-SegNet" ]; then
  git clone https://github.com/NimaTorbati/ACS-SegNet.git ACS-SegNet
fi
