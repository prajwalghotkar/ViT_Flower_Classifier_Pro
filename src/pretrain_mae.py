import argparse
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm

from vit_mae import MaskedAutoencoderViT


class UnlabeledImageDataset(Dataset):
    def __init__(self, root_dir, transform):
        self.paths = glob.glob(f"{root_dir}/*/*.jpg") + glob.glob(f"{root_dir}/*/*.jpeg")
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


def pretrain(model, dataloader, optimizer, device, epochs, checkpoint_path=None):
    model.to(device)
    history = []

    for epoch in tqdm(range(epochs)):
        model.train()
        total_loss = 0

        for images in dataloader:
            images = images.to(device)
            loss, _, _ = model(images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        history.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} | reconstruction_loss {avg_loss:.4f}")

        if checkpoint_path is not None:
            torch.save(model.state_dict(), checkpoint_path)

    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/flowers_raw")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--mask_ratio", type=float, default=0.75)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/mae_pretrained.pth")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    dataset = UnlabeledImageDataset(args.data_dir, transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    print(f"Unlabeled images for pretraining: {len(dataset)}")

    model = MaskedAutoencoderViT(img_size=args.img_size, mask_ratio=args.mask_ratio)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)

    pretrain(model, dataloader, optimizer, device, args.epochs, args.checkpoint)


if __name__ == "__main__":
    main()
