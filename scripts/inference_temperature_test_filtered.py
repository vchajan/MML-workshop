import torch
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import importlib.util

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT = Path(".")
MODEL_PATH = PROJECT / "models/codebook_colorizer_filtered_e10_b10/best.pt"
CODEBOOK_PATH = PROJECT / "models/codebook_filtered/ab_codebook_256_filtered.npz"
INPUT_DIR = PROJECT / "input_images"
OUT_DIR = PROJECT / "outputs/inference_temperature_test_filtered"
OUT_DIR.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location("train_mod", "scripts/train_codebook_colorizer.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
z = np.load(CODEBOOK_PATH)

centers_ab = z["centers_ab"].astype(np.float32) / 128.0
centers_ab = np.clip(centers_ab, -1.0, 1.0)
centers = torch.from_numpy(centers_ab).to(DEVICE)

model = m.ResNetCodebookColorizer(
    num_classes=centers.shape[0],
    pretrained=False,
).to(DEVICE)

model.load_state_dict(ckpt["model_state"])
model.eval()

def load_L(path, image_size=256):
    img = Image.open(path).convert("RGB")
    img = img.resize((image_size, image_size), Image.BICUBIC)
    rgb = np.asarray(img).astype(np.float32) / 255.0
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[:, :, 0:1] / 50.0 - 1.0
    L_t = torch.from_numpy(L).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
    return img, L_t

def lab_to_rgb(L_t, ab_t):
    L_np = L_t.detach().cpu().numpy()[0, 0]
    ab_np = ab_t.detach().cpu().numpy()[0]

    lab = np.zeros((L_np.shape[0], L_np.shape[1], 3), dtype=np.float32)
    lab[:, :, 0] = (L_np + 1.0) * 50.0
    lab[:, :, 1:3] = np.transpose(ab_np, (1, 2, 0)) * 128.0

    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    rgb = np.clip(rgb, 0.0, 1.0)
    return Image.fromarray((rgb * 255).astype(np.uint8))

temps = [0.22, 0.26, 0.30, 0.34, 0.38, 0.45]

image_paths = []
for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]:
    image_paths.extend(INPUT_DIR.glob(ext))

if not image_paths:
    raise RuntimeError("No images found in input_images/. Put test images there first.")

for path in image_paths:
    original, L = load_L(path)

    with torch.no_grad():
        logits, residual = model(L)

    gray = Image.fromarray(((L[0, 0].detach().cpu().numpy() + 1.0) * 0.5 * 255).astype(np.uint8)).convert("RGB")

    outputs = [gray]

    for t in temps:
        with torch.no_grad():
            ab_base = m.expected_ab_from_logits(logits, centers, temperature=t)
            ab_pred = torch.clamp(ab_base + residual, -1.0, 1.0)
        outputs.append(lab_to_rgb(L, ab_pred))

    outputs.append(original.resize((256, 256), Image.BICUBIC))

    grid = Image.new("RGB", (256 * len(outputs), 256), "white")
    for i, im in enumerate(outputs):
        grid.paste(im, (i * 256, 0))

    out_path = OUT_DIR / f"{path.stem}_temperature_grid.png"
    grid.save(out_path)
    print("Saved:", out_path)

print("Done.")
