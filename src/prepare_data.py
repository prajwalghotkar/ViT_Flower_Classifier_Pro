import os
import shutil
import random
import argparse
from pathlib import Path


def split_dataset(source_dir, output_dir, train_ratio=0.7, val_ratio=0.15, seed=42):
    random.seed(seed)
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    classes = sorted([d.name for d in source_dir.iterdir() if d.is_dir()])
    summary = {}

    for split in ["train", "val", "test"]:
        for cls in classes:
            (output_dir / split / cls).mkdir(parents=True, exist_ok=True)

    for cls in classes:
        images = list((source_dir / cls).glob("*"))
        images = [p for p in images if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        random.shuffle(images)

        n = len(images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_files = images[:n_train]
        val_files = images[n_train:n_train + n_val]
        test_files = images[n_train + n_val:]

        for f in train_files:
            shutil.copy2(f, output_dir / "train" / cls / f.name)
        for f in val_files:
            shutil.copy2(f, output_dir / "val" / cls / f.name)
        for f in test_files:
            shutil.copy2(f, output_dir / "test" / cls / f.name)

        summary[cls] = {"total": n, "train": len(train_files), "val": len(val_files), "test": len(test_files)}

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="data/flowers_raw")
    parser.add_argument("--output", type=str, default="data/flowers_split")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = split_dataset(args.source, args.output, args.train_ratio, args.val_ratio, args.seed)
    for cls, counts in result.items():
        print(cls, counts)
