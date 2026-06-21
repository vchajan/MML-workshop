import argparse
import random
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ColorizationDataset(Dataset):
    def __init__(self, data_dir, image_size=256, bins=16, augment=True):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.bins = bins
        self.augment = augment

        self.paths = [
            p for p in self.data_dir.rglob("*")
            if p.suffix.lower() in IMAGE_EXTS
        ]

        if len(self.paths) == 0:
            raise RuntimeError(f"No images found in {self.data_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]

        img = Image.open(path).convert("RGB")
        img = img.resize((self.image_size, self.image_size), Image.BICUBIC)

        if self.augment and random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        rgb = np.asarray(img).astype(np.float32) / 255.0

        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

        # OpenCV float LAB:
        # L approximately 0..100
        # a,b approximately -128..127
        L = lab[:, :, 0:1] / 50.0 - 1.0
        ab = lab[:, :, 1:3] / 128.0
        ab = np.clip(ab, -1.0, 1.0)

        # color-bin target from ground-truth ab
        a_idx = np.floor((ab[:, :, 0] + 1.0) * 0.5 * self.bins).astype(np.int64)
        b_idx = np.floor((ab[:, :, 1] + 1.0) * 0.5 * self.bins).astype(np.int64)
        a_idx = np.clip(a_idx, 0, self.bins - 1)
        b_idx = np.clip(b_idx, 0, self.bins - 1)
        cls = a_idx * self.bins + b_idx

        L_t = torch.from_numpy(L).permute(2, 0, 1).float()
        ab_t = torch.from_numpy(ab).permute(2, 0, 1).float()
        cls_t = torch.from_numpy(cls).long()
        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).float()

        return L_t, ab_t, cls_t, rgb_t


def group_norm(channels):
    groups = min(8, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            group_norm(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            group_norm(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class HybridUNet(nn.Module):
    def __init__(self, base=48, bins=16):
        super().__init__()
        classes = bins * bins

        self.enc1 = ConvBlock(1, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.enc4 = ConvBlock(base * 4, base * 8)

        self.pool = nn.MaxPool2d(2)

        self.mid = ConvBlock(base * 8, base * 8)

        self.up4 = nn.ConvTranspose2d(base * 8, base * 8, 2, stride=2)
        self.dec4 = ConvBlock(base * 16, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = ConvBlock(base * 8, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)

        self.ab_head = nn.Conv2d(base, 2, kernel_size=1)
        self.cls_head = nn.Conv2d(base, classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        m = self.mid(self.pool(e4))

        d4 = self.up4(m)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        ab = torch.tanh(self.ab_head(d1))
        logits = self.cls_head(d1)

        return ab, logits


def hybrid_loss(ab_pred, logits, ab_true, cls_true, lambda_cls=0.30, chroma_weight=4.0):
    chroma = torch.sqrt(torch.sum(ab_true ** 2, dim=1, keepdim=True)).clamp(0.0, 1.5)
    weight = 1.0 + chroma_weight * chroma

    ab_loss_map = F.smooth_l1_loss(ab_pred, ab_true, reduction="none").mean(dim=1, keepdim=True)
    ab_loss = (ab_loss_map * weight).mean()

    cls_loss_map = F.cross_entropy(logits, cls_true, reduction="none").unsqueeze(1)
    cls_loss = (cls_loss_map * weight).mean()

    total = ab_loss + lambda_cls * cls_loss
    return total, ab_loss.detach(), cls_loss.detach()


def lab_to_rgb_batch(L, ab):
    L_np = L.detach().cpu().numpy()
    ab_np = ab.detach().cpu().numpy()

    out = []
    for i in range(L_np.shape[0]):
        lab = np.zeros((L_np.shape[2], L_np.shape[3], 3), dtype=np.float32)
        lab[:, :, 0] = (L_np[i, 0] + 1.0) * 50.0
        lab[:, :, 1:3] = np.transpose(ab_np[i], (1, 2, 0)) * 128.0

        rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        rgb = np.clip(rgb, 0.0, 1.0)
        out.append(rgb)

    return out


@torch.no_grad()
def save_samples(model, loader, device, out_dir, epoch, max_images=4):
    model.eval()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    L, ab_true, cls_true, rgb_true = next(iter(loader))
    L = L.to(device)
    rgb_true = rgb_true[:max_images].cpu().numpy()

    ab_pred, _ = model(L)
    pred_rgbs = lab_to_rgb_batch(L[:max_images], ab_pred[:max_images])

    rows = []
    for i in range(min(max_images, L.shape[0])):
        gray = ((L[i, 0].detach().cpu().numpy() + 1.0) * 0.5)
        gray_rgb = np.stack([gray, gray, gray], axis=2)

        pred = pred_rgbs[i]
        true = np.transpose(rgb_true[i], (1, 2, 0))

        row = np.concatenate([gray_rgb, pred, true], axis=1)
        rows.append(row)

    grid = np.concatenate(rows, axis=0)
    grid = (np.clip(grid, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(grid).save(out_dir / f"epoch_{epoch:03d}.png")


def train_one_epoch(model, loader, optimizer, scaler, device, args):
    model.train()
    total_loss = 0.0
    total_ab = 0.0
    total_cls = 0.0

    for L, ab, cls, _ in loader:
        L = L.to(device, non_blocking=True)
        ab = ab.to(device, non_blocking=True)
        cls = cls.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=args.amp):
            ab_pred, logits = model(L)
            loss, ab_l, cls_l = hybrid_loss(
                ab_pred,
                logits,
                ab,
                cls,
                lambda_cls=args.lambda_cls,
                chroma_weight=args.chroma_weight,
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_ab += ab_l.item()
        total_cls += cls_l.item()

    n = len(loader)
    return total_loss / n, total_ab / n, total_cls / n


@torch.no_grad()
def validate(model, loader, device, args):
    model.eval()
    total_loss = 0.0
    total_ab = 0.0
    total_cls = 0.0

    for L, ab, cls, _ in loader:
        L = L.to(device, non_blocking=True)
        ab = ab.to(device, non_blocking=True)
        cls = cls.to(device, non_blocking=True)

        with autocast(enabled=args.amp):
            ab_pred, logits = model(L)
            loss, ab_l, cls_l = hybrid_loss(
                ab_pred,
                logits,
                ab,
                cls,
                lambda_cls=args.lambda_cls,
                chroma_weight=args.chroma_weight,
            )

        total_loss += loss.item()
        total_ab += ab_l.item()
        total_cls += cls_l.item()

    n = len(loader)
    return total_loss / n, total_ab / n, total_cls / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/dataset")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--base", type=int, default=48)
    parser.add_argument("--bins", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambda-cls", type=float, default=0.30)
    parser.add_argument("--chroma-weight", type=float, default=4.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--model-dir", type=str, default="models/lab_hybrid")
    parser.add_argument("--sample-dir", type=str, default="outputs/lab_hybrid_samples")
    args = parser.parse_args()

    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    dataset = ColorizationDataset(
        data_dir=args.data,
        image_size=args.image_size,
        bins=args.bins,
        augment=True,
    )

    val_size = max(1, int(len(dataset) * 0.10))
    train_size = len(dataset) - val_size

    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    # disable augmentation for validation dataset object
    val_ds.dataset.augment = False

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    print(f"Images: {len(dataset)} | train: {train_size} | val: {val_size}")
    print(f"Image size: {args.image_size} | batch: {args.batch_size} | base: {args.base}")
    print(f"Bins: {args.bins}x{args.bins} = {args.bins * args.bins} classes")

    model = HybridUNet(base=args.base, bins=args.bins).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = GradScaler(enabled=args.amp and device.type == "cuda")

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        start = time.time()

        train_loss, train_ab, train_cls = train_one_epoch(
            model, train_loader, optimizer, scaler, device, args
        )

        val_loss, val_ab, val_cls = validate(model, val_loader, device, args)

        elapsed = time.time() - start

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train {train_loss:.4f} ab {train_ab:.4f} cls {train_cls:.4f} | "
            f"val {val_loss:.4f} ab {val_ab:.4f} cls {val_cls:.4f} | "
            f"{elapsed:.1f}s"
        )

        save_samples(model, val_loader, device, args.sample_dir, epoch)

        last_path = model_dir / "last.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "args": vars(args),
                "val_loss": val_loss,
            },
            last_path,
        )

        if val_loss < best_val:
            best_val = val_loss
            best_path = model_dir / "best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "args": vars(args),
                    "val_loss": val_loss,
                },
                best_path,
            )
            print(f"Saved best model: {best_path}")

    print("Done.")


if __name__ == "__main__":
    main()
