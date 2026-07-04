import argparse
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from huggingface_hub import hf_hub_download
import models_vit  # RETFound's model definition — must be on your PYTHONPATH

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

def get_args():
    parser = argparse.ArgumentParser("RETFound embedding extractor")
    parser.add_argument("--input_dir",  type=str, required=True,
                        help="Directory of input images (searched recursively)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save .pt embedding files")
    parser.add_argument("--hf_token",   type=str, default=None,
                        help="HuggingFace token for downloading gated model weights")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a local .pth checkpoint (skips HF download)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size",   type=int, default=224)
    parser.add_argument("--device",     type=str, default="cuda")
    return parser.parse_args()


def build_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def find_images(input_dir):
    paths = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
                paths.append(os.path.join(root, f))
    return sorted(paths)


def load_model(args, device):
    # Download from HuggingFace if no local checkpoint given
    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        print("Downloading RETFound_mae_natureCFP from HuggingFace...")
        ckpt_path = hf_hub_download(
            repo_id="YukunZhou/RETFound_mae_natureCFP",
            filename="RETFound_mae_natureCFP.pth",
            token=args.hf_token,
        )
        print(f"Checkpoint saved to: {ckpt_path}")

    # Build ViT-Large model (RETFound_mae architecture)
    model = models_vit.__dict__["vit_large_patch16"](
        img_size=args.img_size,
        num_classes=0,       # no classification head — we want raw features
        global_pool=False,   # return full sequence so we can grab CLS token
    )

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)

    # Strip keys that don't belong in a feature extractor
    for k in ["head.weight", "head.bias"]:
        state_dict.pop(k, None)

    msg = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint: {msg}")

    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def extract(model, image_paths, transform, batch_size, device, output_dir, input_dir):
    os.makedirs(output_dir, exist_ok=True)

    total = len(image_paths)
    for start in range(0, total, batch_size):
        batch_paths = image_paths[start:start + batch_size]
        imgs, valid_paths = [], []

        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                imgs.append(transform(img))
                valid_paths.append(p)
            except Exception as e:
                print(f"[skip] {p}: {e}")

        if not imgs:
            continue

        batch = torch.stack(imgs).to(device)

        # forward_features returns [B, seq_len, embed_dim]
        # CLS token is index 0
        features = model.forward_features(batch)   # [B, 197, 1024] for ViT-L/16
        embeddings = features[:, 0, :]             # [B, 1024]

        for emb, src_path in zip(embeddings, valid_paths):
            # Preserve subdirectory structure relative to input_dir
            rel = os.path.relpath(src_path, input_dir)
            stem = os.path.splitext(rel)[0]
            out_path = os.path.join(output_dir, stem + ".pt")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            torch.save(emb.cpu(), out_path)

        done = min(start + batch_size, total)
        print(f"[{done}/{total}] saved embeddings")

    print(f"\nDone. Embeddings saved to: {output_dir}")


def main():
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    image_paths = find_images(args.input_dir)
    print(f"Found {len(image_paths)} images in {args.input_dir}")
    if not image_paths:
        print("No images found — check --input_dir and supported extensions.")
        return

    transform = build_transform(args.img_size)
    model = load_model(args, device)
    extract(model, image_paths, transform, args.batch_size, device, args.output_dir, args.input_dir)


if __name__ == "__main__":
    main()
