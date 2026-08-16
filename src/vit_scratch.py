import torch
from torch import nn


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, embedding_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.patcher = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)

    def forward(self, x):
        image_resolution = x.shape[-1]
        assert image_resolution % self.patch_size == 0
        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched)
        return x_flattened.permute(0, 2, 1)


class MultiheadSelfAttentionBlock(nn.Module):
    def __init__(self, embedding_dim=768, num_heads=12, attn_dropout=0.0):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )

    def forward(self, x):
        x = self.layer_norm(x)
        attn_output, attn_weights = self.multihead_attn(
            query=x, key=x, value=x, need_weights=True, average_attn_weights=True
        )
        return attn_output, attn_weights


class MLPBlock(nn.Module):
    def __init__(self, embedding_dim=768, mlp_size=3072, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_size, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.layer_norm(x)
        return self.mlp(x)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embedding_dim=768, num_heads=12, mlp_size=3072, mlp_dropout=0.1, attn_dropout=0.0):
        super().__init__()
        self.msa_block = MultiheadSelfAttentionBlock(embedding_dim, num_heads, attn_dropout)
        self.mlp_block = MLPBlock(embedding_dim, mlp_size, mlp_dropout)

    def forward(self, x, return_attn=False):
        attn_out, attn_weights = self.msa_block(x)
        x = attn_out + x
        x = self.mlp_block(x) + x
        if return_attn:
            return x, attn_weights
        return x


class ViT(nn.Module):
    def __init__(
        self,
        img_size=224,
        in_channels=3,
        patch_size=16,
        num_transformer_layers=12,
        embedding_dim=768,
        mlp_size=3072,
        num_heads=12,
        attn_dropout=0.0,
        mlp_dropout=0.1,
        embedding_dropout=0.1,
        num_classes=5,
    ):
        super().__init__()
        assert img_size % patch_size == 0

        self.num_patches = (img_size * img_size) // patch_size ** 2
        self.class_embedding = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, embedding_dim))
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, embedding_dim)

        self.transformer_encoder = nn.ModuleList([
            TransformerEncoderBlock(embedding_dim, num_heads, mlp_size, mlp_dropout, attn_dropout)
            for _ in range(num_transformer_layers)
        ])

        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(embedding_dim, num_classes),
        )

    def forward(self, x, return_attn=False):
        batch_size = x.shape[0]
        class_token = self.class_embedding.expand(batch_size, -1, -1)
        x = self.patch_embedding(x)
        x = torch.cat((class_token, x), dim=1)
        x = self.position_embedding + x
        x = self.embedding_dropout(x)

        attn_maps = []
        for block in self.transformer_encoder:
            if return_attn:
                x, attn = block(x, return_attn=True)
                attn_maps.append(attn)
            else:
                x = block(x)

        logits = self.classifier(x[:, 0])
        if return_attn:
            return logits, attn_maps
        return logits


def vit_tiny(num_classes=5):
    return ViT(
        num_transformer_layers=6,
        embedding_dim=256,
        mlp_size=512,
        num_heads=4,
        num_classes=num_classes,
    )


def vit_base(num_classes=5):
    return ViT(num_classes=num_classes)
