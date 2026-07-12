from pathlib import Path
import subprocess

REPO_URL = "https://github.com/NimaTorbati/ACS-SegNet.git"
repo_dir = Path("ACS-SegNet")
model_file = repo_dir / "model.py"

if not repo_dir.exists():
    subprocess.check_call(["git", "clone", "--depth", "1", REPO_URL, str(repo_dir)])

text = model_file.read_text(encoding="utf-8")
text = text.replace(
    "self.segformer = SegformerModel.from_pretrained(segformer_variant, config=seg_cfg)",
    "self.segformer = SegformerModel(seg_cfg)",
)
model_file.write_text(text, encoding="utf-8")

for path in Path("checkpoints").glob("*.pth"):
    size_mb = path.stat().st_size / 1024 / 1024
    print(f"checkpoint: {path} ({size_mb:.1f} MB)")

print("Vercel build preparation complete.")
