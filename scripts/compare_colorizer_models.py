import argparse
import random
from pathlib import Path
import importlib.util

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_train_module():
    spec = importlib.util.spec_from_file_location(
        "train_mod",
        "scripts/train_codebook_colorizer.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_model(module, checkpoint_path, num_classes, device):
    ckpt = torch.load(checkpoint_path, map_location=device)

    model = module.ResNetCodebookColorizer(
        num_classes=num_classes,
        pretrained=False,
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    epoch = ckpt.get("epoch", "?")
    val_loss = ckpt.get("val_loss", None)

    return model, epoch, val_loss


def load_l_and_original(path, image_size, device):
    img = Image.open(path).convert("RGB")
    img = img.resize((image_size, image_size), Image.BICUBIC)

    rgb = np.asarray(img).astype(np.float32) / 255.0
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    L = lab[:, :, 0:1] / 50.0 - 1.0
    L_t = torch.from_numpy(L).permute(2, 0, 1).unsqueeze(0).float().to(device)

    gray = ((L[:, :, 0] + 1.0) * 0.5 * 255).clip(0, 255).astype(np.uint8)
    gray_rgb = Image.fromarray(gray).convert("RGB")

    return L_t, gray_rgb, img


def lab_to_rgb_image(L_t, ab_t):
    L_np = L_t.detach().cpu().numpy()[0, 0]
    ab_np = ab_t.detach().cpu().numpy()[0]

    lab = np.zeros((L_np.shape[0], L_np.shape[1], 3), dtype=np.float32)
    lab[:, :, 0] = (L_np + 1.0) * 50.0
    lab[:, :, 1:3] = np.transpose(ab_np, (1, 2, 0)) * 128.0

    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    rgb = np.clip(rgb, 0.0, 1.0)
    return Image.fromarray((rgb * 255).astype(np.uint8))


@torch.no_grad()
def predict(module, model, L_t, centers, temperature):
    logits, residual = model(L_t)
    ab_base = module.expected_ab_from_logits(logits, centers, temperature=temperature)
    ab_pred = torch.clamp(ab_base + residual, -1.0, 1.0)
    return lab_to_rgb_image(L_t, ab_pred)


def add_label(img, text):
    canvas = Image.new("RGB", (img.width, img.height + 28), "white")
    canvas.paste(img, (0, 28))
    d = ImageDraw.Draw(canvas)
    d.text((6, 7), text, fill=(0, 0, 0))
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/dataset_filtered")
    parser.add_argument("--codebook", default="models/codebook_filtered/ab_codebook_256_filtered.npz")
    parser.add_argument("--model-a", default="models/codebook_colorizer_filtered_e10_b10/best.pt")
    parser.add_argument("--model-b", default="models/codebook_colorizer_filtered_v3/best.pt")
    parser.add_argument("--name-a", default="e10_b10_best")
    parser.add_argument("--name-b", default="v3_best")
    parser.add_argument("--out", default="outputs/model_compare/e10_vs_v3_random.png")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.30)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    module = load_train_module()

    z = np.load(args.codebook)
    centers_ab = z["centers_ab"].astype(np.float32) / 128.0
    centers_ab = np.clip(centers_ab, -1.0, 1.0)
    centers = torch.from_numpy(centers_ab).to(device)

    model_a, epoch_a, val_a = load_model(module, args.model_a, centers.shape[0], device)
    model_b, epoch_b, val_b = load_model(module, args.model_b, centers.shape[0], device)

    print(f"A: {args.name_a} | epoch={epoch_a} | val_loss={val_a}")
    print(f"B: {args.name_b} | epoch={epoch_b} | val_loss={val_b}")
    print(f"Temperature: {args.temperature}")

    paths = [p for p in Path(args.data).rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    if len(paths) == 0:
        raise RuntimeError(f"No images found in {args.data}")

    rng = random.Random(args.seed)
    selected = rng.sample(paths, min(args.count, len(paths)))

    rows = []

    for p in selected:
        L_t, gray, original = load_l_and_original(p, args.image_size, device)

        pred_a = predict(module, model_a, L_t, centers, args.temperature)
        pred_b = predict(module, model_b, L_t, centers, args.temperature)

        cols = [
            add_label(gray, "gray"),
            add_label(pred_a, args.name_a),
            add_label(pred_b, args.name_b),
            add_label(original, "original"),
        ]

        row = Image.new("RGB", (args.image_size * 4, args.image_size + 28), "white")
        for i, im in enumerate(cols):
            row.paste(im, (i * args.image_size, 0))

        rows.append(row)
        print("Used:", p)

    grid = Image.new("RGB", (args.image_size * 4, (args.image_size + 28) * len(rows)), "white")

    for i, row in enumerate(rows):
        grid.paste(row, (0, i * (args.image_size + 28)))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out)

    print("Saved:", out)


if __name__ == "__main__":
    main()
