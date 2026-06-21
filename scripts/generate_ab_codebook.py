import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import MiniBatchKMeans

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(data_dir: Path):
    return [p for p in data_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS]


def load_ab_pixels(path: Path, image_size: int):
    img = Image.open(path).convert("RGB")
    img = img.resize((image_size, image_size), Image.BICUBIC)
    rgb = np.asarray(img).astype(np.float32) / 255.0
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    ab = lab[:, :, 1:3].reshape(-1, 2)
    chroma = np.sqrt(np.sum(ab ** 2, axis=1))
    return ab, chroma


def weighted_sample_indices(chroma, n_take, rng):
    if len(chroma) <= n_take:
        return np.arange(len(chroma))

    # prefer more saturated pixels so codebook is not dominated by gray/brown averages
    weights = 0.15 + np.clip(chroma / 60.0, 0.0, 1.0)
    probs = weights / weights.sum()
    return rng.choice(len(chroma), size=n_take, replace=False, p=probs)


def build_codebook(
    data_dir,
    image_size=128,
    n_clusters=256,
    pixels_per_image=256,
    max_images=None,
    seed=42,
):
    rng = np.random.default_rng(seed)
    paths = list_images(Path(data_dir))

    if max_images is not None:
        paths = paths[:max_images]

    if len(paths) == 0:
        raise RuntimeError(f"No images found in {data_dir}")

    print(f"Found {len(paths)} images")

    sampled = []
    for i, path in enumerate(paths, 1):
        try:
            ab, chroma = load_ab_pixels(path, image_size=image_size)
            idx = weighted_sample_indices(chroma, pixels_per_image, rng)
            sampled.append(ab[idx])

            if i % 200 == 0 or i == len(paths):
                print(f"[{i}/{len(paths)}] sampled")
        except Exception as e:
            print(f"Skipping {path.name}: {e}")

    sampled = np.concatenate(sampled, axis=0).astype(np.float32)
    print("Total sampled pixels:", len(sampled))

    print("Fitting MiniBatchKMeans...")
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=seed,
        batch_size=8192,
        n_init=3,
        max_iter=200,
        verbose=0,
    )
    kmeans.fit(sampled)

    centers = kmeans.cluster_centers_.astype(np.float32)

    print("Computing cluster histogram...")
    labels = kmeans.predict(sampled)
    counts = np.bincount(labels, minlength=n_clusters).astype(np.float64)
    prior = counts / counts.sum()

    # class weights: inverse-frequency with smoothing
    alpha = 0.5
    weights = 1.0 / np.power(prior + 1e-6, alpha)
    weights = weights / weights.mean()

    # sort centers by chroma only for easier inspection
    chroma_centers = np.sqrt(np.sum(centers ** 2, axis=1))
    order = np.argsort(chroma_centers)

    centers = centers[order]
    prior = prior[order]
    weights = weights[order]
    chroma_centers = chroma_centers[order]

    return {
        "centers_ab": centers,
        "prior": prior.astype(np.float32),
        "weights": weights.astype(np.float32),
        "chroma": chroma_centers.astype(np.float32),
        "num_images": len(paths),
        "num_sampled_pixels": int(len(sampled)),
        "image_size": int(image_size),
        "pixels_per_image": int(pixels_per_image),
        "n_clusters": int(n_clusters),
        "seed": int(seed),
    }


def save_outputs(result, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        centers_ab=result["centers_ab"],
        prior=result["prior"],
        weights=result["weights"],
        chroma=result["chroma"],
    )

    meta = {
        "num_images": result["num_images"],
        "num_sampled_pixels": result["num_sampled_pixels"],
        "image_size": result["image_size"],
        "pixels_per_image": result["pixels_per_image"],
        "n_clusters": result["n_clusters"],
        "seed": result["seed"],
        "top10_rarest_weights": sorted(result["weights"], reverse=True)[:10],
    }

    meta_path = out_path.with_suffix(".json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved codebook: {out_path}")
    print(f"Saved metadata: {meta_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/dataset")
    parser.add_argument("--out", type=str, default="models/codebook/ab_codebook_256.npz")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--clusters", type=int, default=256)
    parser.add_argument("--pixels-per-image", type=int, default=256)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = build_codebook(
        data_dir=args.data,
        image_size=args.image_size,
        n_clusters=args.clusters,
        pixels_per_image=args.pixels_per_image,
        max_images=args.max_images,
        seed=args.seed,
    )
    save_outputs(result, Path(args.out))


if __name__ == "__main__":
    main()
