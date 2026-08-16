import torch
from torch import nn

from vit_scratch import PatchEmbedding, TransformerEncoderBlock


def random_masking(x, mask_ratio):
    batch_size, num_patches, dim = x.shape
    num_keep = int(num_patches * (1 - mask_ratio))

    noise = torch.rand(batch_size, num_patches, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    ids_keep = ids_shuffle[:, :num_keep]
    x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, dim))

    mask = torch.ones(batch_size, num_patches, device=x.device)
    mask[:, :num_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_kept, mask, ids_restore


class MAEEncoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embedding_dim=384,
                 depth=6, num_heads=6, mlp_size=1536):
        super().__init__()
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, embedding_dim)
        self.num_patches = (img_size // patch_size) ** 2
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches, embedding_dim) * 0.02)

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embedding_dim, num_heads, mlp_size)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x, mask_ratio=0.75):
        x = self.patch_embedding(x)
        x = x + self.position_embedding

        x, mask, ids_restore = random_masking(x, mask_ratio)

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_full(self, x):
        x = self.patch_embedding(x)
        x = x + self.position_embedding
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class MAEDecoder(nn.Module):
    def __init__(self, num_patches, patch_size=16, in_channels=3, encoder_dim=384,
                 decoder_dim=192, depth=2, num_heads=6, mlp_size=768):
        super().__init__()
        self.decoder_embed = nn.Linear(encoder_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.position_embedding = nn.Parameter(torch.randn(1, num_patches, decoder_dim) * 0.02)

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(decoder_dim, num_heads, mlp_size)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(decoder_dim)
        self.head = nn.Linear(decoder_dim, patch_size * patch_size * in_channels)

    def forward(self, x, ids_restore):
        x = self.decoder_embed(x)

        batch_size, num_visible, dim = x.shape
        num_total = ids_restore.shape[1]
        num_masked = num_total - num_visible

        mask_tokens = self.mask_token.repeat(batch_size, num_masked, 1)
        x_full = torch.cat([x, mask_tokens], dim=1)
        x_full = torch.gather(x_full, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, dim))

        x_full = x_full + self.position_embedding

        for block in self.blocks:
            x_full = block(x_full)
        x_full = self.norm(x_full)

        return self.head(x_full)


def patchify(images, patch_size):
    batch_size, channels, height, width = images.shape
    h_patches = height // patch_size
    w_patches = width // patch_size

    x = images.reshape(batch_size, channels, h_patches, patch_size, w_patches, patch_size)
    x = x.permute(0, 2, 4, 3, 5, 1)
    x = x.reshape(batch_size, h_patches * w_patches, patch_size * patch_size * channels)
    return x


class MaskedAutoencoderViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 encoder_dim=384, encoder_depth=6, encoder_heads=6,
                 decoder_dim=192, decoder_depth=2, decoder_heads=6,
                 mask_ratio=0.75, norm_pix_loss=True):
        super().__init__()
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss

        self.encoder = MAEEncoder(
            img_size, patch_size, in_channels, encoder_dim, encoder_depth, encoder_heads
        )
        self.decoder = MAEDecoder(
            self.encoder.num_patches, patch_size, in_channels,
            encoder_dim, decoder_dim, decoder_depth, decoder_heads
        )

    def forward(self, images):
        latent, mask, ids_restore = self.encoder(images, self.mask_ratio)
        pred = self.decoder(latent, ids_restore)
        target = patchify(images, self.patch_size)

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6) ** 0.5

        loss_per_patch = (pred - target) ** 2
        loss_per_patch = loss_per_patch.mean(dim=-1)
        loss = (loss_per_patch * mask).sum() / mask.sum()

        return loss, pred, mask

    def encode_for_classification(self, images):
        return self.encoder.forward_full(images).mean(dim=1)
