import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from PIL import Image


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def plot_loss_curves(results, save_path=None):
    epochs = range(len(results["train_loss"]))
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, results["train_loss"], label="train_loss")
    plt.plot(epochs, results["val_loss"], label="val_loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, results["train_acc"], label="train_acc")
    plt.plot(epochs, results["val_acc"], label="val_acc")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def evaluate_model(model, dataloader, class_names, device, save_path=None):
    model.eval()
    all_preds, all_labels = [], []

    with torch.inference_mode():
        for X, y in dataloader:
            X = X.to(device)
            logits = model(X)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print(report)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    return cm, report


def predict_image(model, image_path, class_names, transform, device):
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    model.eval()
    with torch.inference_mode():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    top_prob, top_idx = torch.topk(probs, k=min(3, len(class_names)))

    plt.figure()
    plt.imshow(img)
    plt.axis(False)
    title = " | ".join(
        f"{class_names[i]} {p:.2f}" for p, i in zip(top_prob.cpu().numpy(), top_idx.cpu().numpy())
    )
    plt.title(title)
    plt.show()

    return {class_names[i]: float(p) for p, i in zip(top_prob.cpu().numpy(), top_idx.cpu().numpy())}


def visualize_attention(model, image_path, transform, device, patch_size=16):
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    model.eval()
    with torch.inference_mode():
        _, attn_maps = model(x, return_attn=True)

    last_attn = attn_maps[-1][0]
    cls_attn = last_attn[0, 1:]

    grid_size = int(len(cls_attn) ** 0.5)
    attn_grid = cls_attn.reshape(grid_size, grid_size).cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[0].axis(False)

    axes[1].imshow(img)
    axes[1].imshow(attn_grid, cmap="jet", alpha=0.5, extent=(0, img.width, img.height, 0))
    axes[1].set_title("Attention Map (CLS token, last layer)")
    axes[1].axis(False)

    plt.show()
