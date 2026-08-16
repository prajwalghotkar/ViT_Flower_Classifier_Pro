import torch


def compute_rollout(attn_maps, discard_ratio=0.9, head_fusion="mean"):
    result = torch.eye(attn_maps[0].shape[-1])

    with torch.no_grad():
        for attn in attn_maps:
            if head_fusion == "mean":
                fused = attn
            else:
                fused = attn

            flat = fused.view(fused.shape[0], -1)
            _, indices = flat.topk(int(flat.shape[-1] * discard_ratio), dim=-1, largest=False)
            flat.scatter_(1, indices, 0)

            identity = torch.eye(fused.shape[-1])
            fused = fused + identity
            fused = fused / fused.sum(dim=-1, keepdim=True)

            result = torch.matmul(fused[0], result)

    return result


def rollout_attention_map(model, image_tensor, discard_ratio=0.9):
    model.eval()
    with torch.inference_mode():
        _, attn_maps = model(image_tensor, return_attn=True)

    attn_maps = [a.detach().cpu().clone() for a in attn_maps]
    rollout = compute_rollout(attn_maps, discard_ratio=discard_ratio)

    cls_attention = rollout[0, 1:]
    grid_size = int(len(cls_attention) ** 0.5)
    attn_grid = cls_attention.reshape(grid_size, grid_size).numpy()

    attn_grid = (attn_grid - attn_grid.min()) / (attn_grid.max() - attn_grid.min() + 1e-8)
    return attn_grid
