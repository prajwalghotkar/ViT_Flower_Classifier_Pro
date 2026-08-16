import argparse
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from vit_scratch import vit_tiny, vit_base
from vit_transfer import build_pretrained_vit, unfreeze_last_n_blocks
import engine
import utils


def get_dataloaders(data_dir, transform, batch_size, num_workers=2):
    train_data = datasets.ImageFolder(f"{data_dir}/train", transform=transform)
    val_data = datasets.ImageFolder(f"{data_dir}/val", transform=transform)
    test_data = datasets.ImageFolder(f"{data_dir}/test", transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, train_data.classes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/flowers_split")
    parser.add_argument("--model", type=str, choices=["scratch_tiny", "scratch_base", "transfer"], default="transfer")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--unfreeze_blocks", type=int, default=2)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    utils.set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if args.model == "transfer":
        model, weights_transform = build_pretrained_vit(num_classes=5, freeze_backbone=True)
        unfreeze_last_n_blocks(model, n=args.unfreeze_blocks)
        eval_transform = weights_transform
    else:
        model = vit_tiny(num_classes=5) if args.model == "scratch_tiny" else vit_base(num_classes=5)
        eval_transform = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        args.data_dir, train_transform, args.batch_size
    )
    print("Classes:", class_names)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss()

    results = engine.train(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=args.epochs,
        device=device,
        scheduler=scheduler,
        checkpoint_path=args.checkpoint,
    )

    utils.plot_loss_curves(results, save_path="outputs/loss_curves.png")

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    utils.evaluate_model(model, test_loader, class_names, device, save_path="outputs/confusion_matrix.png")


if __name__ == "__main__":
    main()
