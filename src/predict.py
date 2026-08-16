import argparse
import torch
from torchvision import transforms

from vit_transfer import build_pretrained_vit
import utils


CLASS_NAMES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, weights_transform = build_pretrained_vit(num_classes=len(CLASS_NAMES), freeze_backbone=False)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)

    result = utils.predict_image(model, args.image, CLASS_NAMES, weights_transform, device)
    print(result)


if __name__ == "__main__":
    main()
