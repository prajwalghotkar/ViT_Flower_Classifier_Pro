import json
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def extract_embeddings(model, dataloader, device, use_mae=False):
    model.eval()
    embeddings, labels, paths = [], [], []

    with torch.inference_mode():
        for batch in dataloader:
            if len(batch) == 3:
                X, y, p = batch
            else:
                X, y = batch
                p = [None] * len(y)

            X = X.to(device)
            if use_mae:
                emb = model.encode_for_classification(X)
            else:
                emb = model(X, return_attn=False)
                if isinstance(emb, tuple):
                    emb = emb[0]

            embeddings.append(emb.cpu().numpy())
            labels.extend(y.tolist() if torch.is_tensor(y) else y)
            paths.extend(p)

    return np.concatenate(embeddings, axis=0), labels, paths


def project_2d(embeddings, method="tsne", seed=42):
    if method == "pca":
        reducer = PCA(n_components=2, random_state=seed)
    else:
        perplexity = min(30, max(5, len(embeddings) // 4))
        reducer = TSNE(n_components=2, random_state=seed, perplexity=perplexity, init="pca")

    coords = reducer.fit_transform(embeddings)
    return coords


def nearest_neighbors(query_embedding, all_embeddings, top_k=5):
    query = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
    corpus = all_embeddings / (np.linalg.norm(all_embeddings, axis=1, keepdims=True) + 1e-8)

    similarities = corpus @ query
    top_indices = np.argsort(-similarities)[:top_k]
    return top_indices, similarities[top_indices]


def export_embedding_map(embeddings, labels, class_names, paths, out_path, method="tsne"):
    coords = project_2d(embeddings, method=method)

    points = []
    for (x, y), label, path in zip(coords, labels, paths):
        points.append({
            "x": float(x),
            "y": float(y),
            "label": class_names[label] if isinstance(label, int) else label,
            "path": path,
        })

    with open(out_path, "w") as f:
        json.dump(points, f)

    return points
