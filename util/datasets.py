import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

# Extensions we'll look for when matching a CSV stem to an actual file on disk
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]


class AdversarialCSVDataset(Dataset):
    """
    Returns (image, task_label, private_label) triples.

    - task_label   comes from args.label_col          (what the classifier predicts)
    - private_label comes from args.private_label_col  (what the adversary predicts / we hide)
    Images are matched to CSV rows by filename *stem* (extension-agnostic).
    """
    def __init__(self, samples, transform=None):
        # samples: list of (image_path, task_label, private_label)
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, task_label, private_label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, task_label, private_label


def _build_stem_lookup(image_dir):
    """Map every image's filename-stem -> full path, so we can match regardless of extension."""
    lookup = {}
    for f in os.listdir(image_dir):
        stem, ext = os.path.splitext(f)
        if ext.lower() in IMAGE_EXTENSIONS:
            lookup[stem] = os.path.join(image_dir, f)
    return lookup


def _load_all_samples(args):
    """Read CSV, match rows to images by stem, return list of (path, task_label, private_label)."""
    df = pd.read_csv(args.csv_path)
    stem_lookup = _build_stem_lookup(args.data_path)

    filename_col = getattr(args, "filename_col", "filename")
    task_col     = args.label_col
    private_col  = args.private_label_col

    samples, missing = [], 0
    for _, row in df.iterrows():
        # strip any extension the CSV might carry, match on stem only
        stem = os.path.splitext(str(row[filename_col]))[0]
        path = stem_lookup.get(stem)
        if path is None:
            missing += 1
            continue
        task_label    = int(row[task_col])
        private_label = int(row[private_col])
        samples.append((path, task_label, private_label))

    if missing > 0:
        print(f"[build_dataset] Warning: {missing} CSV rows had no matching image and were skipped")
    print(f"[build_dataset] Matched {len(samples)} images to CSV rows")
    return samples


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    all_samples = _load_all_samples(args)

    # ---- Reproducible stratified 70/30 train/test split (identical across all calls) ----
    seed = int(getattr(args, "seed", 0))
    task_labels = [s[1] for s in all_samples]  # stratify on the task label
    indices = list(range(len(all_samples)))

    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.30,
        stratify=task_labels,
        random_state=seed,
    )

    # No separate val set -> "val" reuses the test split so evaluation code doesn't break
    if is_train == "train":
        chosen = train_idx
    else:  # "val" or "test"
        chosen = test_idx

    chosen_samples = [all_samples[i] for i in chosen]
    print(f"[build_dataset] split='{is_train}' -> {len(chosen_samples)} samples")

    return AdversarialCSVDataset(chosen_samples, transform=transform)
