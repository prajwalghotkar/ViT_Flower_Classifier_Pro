import torch
from torch import nn
import torchvision


def build_pretrained_vit(num_classes=5, freeze_backbone=True):
    weights = torchvision.models.ViT_B_16_Weights.DEFAULT
    model = torchvision.models.vit_b_16(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.heads.head.in_features
    model.heads = nn.Sequential(
        nn.LayerNorm(in_features),
        nn.Linear(in_features, num_classes),
    )

    transforms = weights.transforms()
    return model, transforms


def unfreeze_last_n_blocks(model, n=2):
    blocks = model.encoder.layers
    for block in list(blocks)[-n:]:
        for param in block.parameters():
            param.requires_grad = True
    for param in model.encoder.ln.parameters():
        param.requires_grad = True
