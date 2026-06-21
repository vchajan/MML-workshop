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
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torchvision.models import resnet18, ResNet18_Weights

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_images(data_dir: str):
    paths = [p for p in Path(data_dir).rglob('*') if p.suffix.lower() in IMAGE_EXTS]
    if not paths:
        raise RuntimeError(f'No images found in {data_dir}')
    return sorted(paths)


class ColorizationDataset(Dataset):
    def __init__(self, paths, image_size=256, augment=True):
        self.paths = list(paths)
        self.image_size = image_size
        self.augment = augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert('RGB')
        img = img.resize((self.image_size, self.image_size), Image.BICUBIC)

        if self.augment and random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        rgb = np.asarray(img).astype(np.float32) / 255.0
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

        L = lab[:, :, 0:1] / 50.0 - 1.0      # roughly [-1, 1]
        ab = lab[:, :, 1:3] / 128.0           # roughly [-1, 1]
        ab = np.clip(ab, -1.0, 1.0)

        L_t = torch.from_numpy(L).permute(2, 0, 1).float()
        ab_t = torch.from_numpy(ab).permute(2, 0, 1).float()
        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).float()
        return L_t, ab_t, rgb_t


def gn(channels):
    groups = min(8, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        gn(out_ch),
        nn.SiLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        gn(out_ch),
        nn.SiLU(inplace=True),
    )


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = conv_block(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class ResNetCodebookColorizer(nn.Module):
    def __init__(self, num_classes=256, residual_scale=0.25, pretrained=True):
        super().__init__()
        self.residual_scale = residual_scale

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        try:
            enc = resnet18(weights=weights)
            print('Encoder: ResNet18 pretrained' if pretrained else 'Encoder: ResNet18 random')
        except Exception as e:
            print(f'Could not load pretrained ResNet18 weights: {e}')
            print('Encoder: ResNet18 random')
            enc = resnet18(weights=None)

        self.register_buffer('im_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('im_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.stem = nn.Sequential(enc.conv1, enc.bn1, enc.relu)  # /2, 64
        self.maxpool = enc.maxpool                               # /4
        self.layer1 = enc.layer1                                 # /4, 64
        self.layer2 = enc.layer2                                 # /8, 128
        self.layer3 = enc.layer3                                 # /16, 256
        self.layer4 = enc.layer4                                 # /32, 512

        self.up4 = UpBlock(512, 256, 256)
        self.up3 = UpBlock(256, 128, 128)
        self.up2 = UpBlock(128, 64, 64)
        self.up1 = UpBlock(64, 64, 64)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 2, stride=2),
            gn(64),
            nn.SiLU(inplace=True),
            conv_block(64, 64),
        )

        self.logits_head = nn.Conv2d(64, num_classes, 1)
        self.residual_head = nn.Conv2d(64, 2, 1)

    def forward(self, L):
        # L is [-1, 1]; convert to ImageNet-like 3-channel input for pretrained encoder
        x = ((L + 1.0) * 0.5).repeat(1, 3, 1, 1)
        x = (x - self.im_mean) / self.im_std

        s0 = self.stem(x)
        x = self.maxpool(s0)
        s1 = self.layer1(x)
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        s4 = self.layer4(s3)

        d3 = self.up4(s4, s3)
        d2 = self.up3(d3, s2)
        d1 = self.up2(d2, s1)
        d0 = self.up1(d1, s0)
        out = self.final_up(d0)

        logits = self.logits_head(out)
        residual = torch.tanh(self.residual_head(out)) * self.residual_scale
        return logits, residual


def load_codebook(path, device):
    z = np.load(path)
    centers = z['centers_ab'].astype(np.float32)
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise RuntimeError(f'Bad centers_ab shape: {centers.shape}. Expected (K, 2). Regenerate codebook.')

    centers_norm = np.clip(centers / 128.0, -1.0, 1.0)
    weights = z['weights'].astype(np.float32)
    if weights.shape[0] != centers.shape[0]:
        raise RuntimeError(f'Bad weights shape: {weights.shape}. Expected ({centers.shape[0]},).')

    centers_t = torch.from_numpy(centers_norm).to(device)
    weights_t = torch.from_numpy(weights).to(device)

    print(f'Loaded codebook: {path}')
    print(f'Classes: {centers_t.shape[0]}')
    print(f'Weight range: {weights_t.min().item():.3f} - {weights_t.max().item():.3f}')
    return centers_t, weights_t


@torch.no_grad()
def assign_codebook_targets(ab_true, centers_norm, chunk_size=65536):
    B, _, H, W = ab_true.shape
    x = ab_true.permute(0, 2, 3, 1).reshape(-1, 2)
    c = centers_norm
    c2 = (c ** 2).sum(dim=1).view(1, -1)
    labels = []
    for start in range(0, x.shape[0], chunk_size):
        xb = x[start:start + chunk_size]
        x2 = (xb ** 2).sum(dim=1, keepdim=True)
        d2 = x2 + c2 - 2.0 * xb @ c.t()
        labels.append(torch.argmin(d2, dim=1))
    return torch.cat(labels, dim=0).view(B, H, W)


def expected_ab_from_logits(logits, centers_norm, temperature=0.32):
    probs = F.softmax(logits / temperature, dim=1)
    centers = centers_norm.view(1, -1, 2, 1, 1)
    return (probs.unsqueeze(2) * centers).sum(dim=1)


def colorization_loss(logits, residual, ab_true, labels, centers_norm, class_weights,
                      lambda_ce=1.0, lambda_ab=8.0, chroma_weight=4.0, temperature=0.32):
    ab_base = expected_ab_from_logits(logits, centers_norm, temperature=temperature)
    ab_pred = torch.clamp(ab_base + residual, -1.0, 1.0)

    ce_map = F.cross_entropy(logits, labels, reduction='none')
    class_w = class_weights[labels]

    chroma = torch.sqrt(torch.sum(ab_true ** 2, dim=1)).clamp(0.0, 1.5)
    sat_w = 1.0 + chroma_weight * chroma

    ce_loss = (ce_map * class_w * sat_w).mean()
    ab_map = F.smooth_l1_loss(ab_pred, ab_true, reduction='none').mean(dim=1)
    ab_loss = (ab_map * sat_w).mean()
    total = lambda_ce * ce_loss + lambda_ab * ab_loss
    return total, ce_loss.detach(), ab_loss.detach(), ab_pred.detach()


def lab_to_rgb_batch(L, ab):
    L_np = L.detach().cpu().numpy()
    ab_np = ab.detach().cpu().numpy()
    out = []
    for i in range(L_np.shape[0]):
        lab = np.zeros((L_np.shape[2], L_np.shape[3], 3), dtype=np.float32)
        lab[:, :, 0] = (L_np[i, 0] + 1.0) * 50.0
        lab[:, :, 1:3] = np.transpose(ab_np[i], (1, 2, 0)) * 128.0
        rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        out.append(np.clip(rgb, 0.0, 1.0))
    return out


@torch.no_grad()
def save_samples(model, loader, device, centers_norm, out_dir, epoch, max_images=4, temperature=0.26):
    model.eval()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    L, ab_true, rgb_true = next(iter(loader))
    L = L.to(device)
    logits, residual = model(L)
    ab_base = expected_ab_from_logits(logits, centers_norm, temperature=temperature)
    ab_pred = torch.clamp(ab_base + residual, -1.0, 1.0)
    pred_rgbs = lab_to_rgb_batch(L[:max_images], ab_pred[:max_images])

    rows = []
    for i in range(min(max_images, L.shape[0])):
        gray = ((L[i, 0].detach().cpu().numpy() + 1.0) * 0.5)
        gray_rgb = np.stack([gray, gray, gray], axis=2)
        pred = pred_rgbs[i]
        true = np.transpose(rgb_true[i].numpy(), (1, 2, 0))
        rows.append(np.concatenate([gray_rgb, pred, true], axis=1))

    grid = np.concatenate(rows, axis=0)
    grid = (np.clip(grid, 0.0, 1.0) * 255).astype(np.uint8)
    Image.fromarray(grid).save(out_dir / f'epoch_{epoch:03d}.png')


def run_epoch(model, loader, optimizer, scaler, device, centers_norm, class_weights, args, train=True):
    model.train(train)
    total_loss = total_ce = total_ab = 0.0

    for L, ab_true, _ in loader:
        L = L.to(device, non_blocking=True)
        ab_true = ab_true.to(device, non_blocking=True)
        labels = assign_codebook_targets(ab_true, centers_norm)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with autocast(enabled=args.amp and device.type == 'cuda'):
                logits, residual = model(L)
                loss, ce_l, ab_l, _ = colorization_loss(
                    logits, residual, ab_true, labels, centers_norm, class_weights,
                    lambda_ce=args.lambda_ce,
                    lambda_ab=args.lambda_ab,
                    chroma_weight=args.chroma_weight,
                    temperature=args.temperature,
                )

            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

        total_loss += loss.item()
        total_ce += ce_l.item()
        total_ab += ab_l.item()

    n = len(loader)
    return total_loss / n, total_ce / n, total_ab / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/dataset')
    parser.add_argument('--codebook', type=str, default='models/codebook/ab_codebook_256.npz')
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--lambda-ce', type=float, default=1.0)
    parser.add_argument('--lambda-ab', type=float, default=8.0)
    parser.add_argument('--chroma-weight', type=float, default=4.0)
    parser.add_argument('--temperature', type=float, default=0.32)
    parser.add_argument('--model-dir', type=str, default='models/codebook_colorizer')
    parser.add_argument('--sample-dir', type=str, default='outputs/codebook_samples')
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    if device.type == 'cuda':
        print('GPU:', torch.cuda.get_device_name(0))

    centers_norm, class_weights = load_codebook(args.codebook, device)

    paths = list_images(args.data)
    rng = random.Random(args.seed)
    rng.shuffle(paths)
    val_size = max(1, int(len(paths) * 0.10))
    val_paths = paths[:val_size]
    train_paths = paths[val_size:]

    train_ds = ColorizationDataset(train_paths, image_size=args.image_size, augment=True)
    val_ds = ColorizationDataset(val_paths, image_size=args.image_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(device.type == 'cuda'))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=(device.type == 'cuda'))

    model = ResNetCodebookColorizer(num_classes=centers_norm.shape[0], pretrained=args.pretrained).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f'Images: {len(paths)} | train: {len(train_paths)} | val: {len(val_paths)}')
    print(f'Image size: {args.image_size} | batch: {args.batch_size}')
    print(f'Parameters: {params / 1e6:.2f}M')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = GradScaler(enabled=args.amp and device.type == 'cuda')

    model_dir = Path(args.model_dir)
    sample_dir = Path(args.sample_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    best_val = float('inf')
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        tr_loss, tr_ce, tr_ab = run_epoch(model, train_loader, optimizer, scaler, device, centers_norm, class_weights, args, train=True)
        va_loss, va_ce, va_ab = run_epoch(model, val_loader, optimizer, scaler, device, centers_norm, class_weights, args, train=False)
        elapsed = time.time() - start

        print(f'Epoch {epoch:03d}/{args.epochs} | train {tr_loss:.4f} ce {tr_ce:.4f} ab {tr_ab:.4f} | val {va_loss:.4f} ce {va_ce:.4f} ab {va_ab:.4f} | {elapsed:.1f}s')
        save_samples(model, val_loader, device, centers_norm, sample_dir, epoch)

        ckpt = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'args': vars(args),
            'val_loss': va_loss,
        }
        torch.save(ckpt, model_dir / 'last.pt')
        if va_loss < best_val:
            best_val = va_loss
            torch.save(ckpt, model_dir / 'best.pt')
            print(f'Saved best model. val_loss={best_val:.4f}')

    print('Done.')


if __name__ == '__main__':
    main()
