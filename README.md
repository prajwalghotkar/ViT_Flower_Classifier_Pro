# Field Notes: Teaching a Transformer to See Flowers

A comparative study of *how* a Vision Transformer comes to see anything at all — not another "train a ViT, report accuracy" tutorial. Three training regimes, same encoder architecture, same 3,670-image, 5-species dataset, evaluated honestly against each other with the internals made visible rather than just a final accuracy number.

**Live interactive write-up:** open `demo/index.html` in any browser (no server needed) — attention rollout galleries, a masked-autoencoder reconstruction plate, a hoverable/filterable embedding-space atlas, and a live nearest-neighbor similarity search, all built from real precomputed runs.

---

## The question this project asks

A Vision Transformer has no convolutional inductive bias — no built-in notion of edges, locality, or texture the way a CNN gets for free from its filters. Everything a ViT knows about images, it has to learn from data. So: on a small, single-domain dataset, what's the cheapest path to a transformer that actually sees something meaningful, and what does that "seeing" look like from the inside?

This project answers it by training the *same* patch-and-attention architecture three different ways on the *same* data, and then actually looking inside each one — attention maps, reconstructed pixels, embedding geometry — instead of stopping at a leaderboard number.

| Regime | What it is | Labels used |
|---|---|---|
| **From scratch** | Hand-built ViT trained directly on classification labels, no prior knowledge | Yes |
| **Masked autoencoding (MAE)** | Same encoder, pretrained by masking 75% of each image's patches and reconstructing the missing pixels | No |
| **ImageNet transfer** | `torchvision` ViT-B/16 pretrained on 1.3M ImageNet images, fine-tuned on this dataset | Yes (ImageNet's, not this dataset's) |

## Dataset

3,670 images across 5 classes — the widely-used TensorFlow `flower_photos` set, released under CC-BY 2.0, sitting in `data/flowers_raw/`:

| Class | Images |
|---|---|
| daisy | 633 |
| dandelion | 898 |
| roses | 641 |
| sunflowers | 699 |
| tulips | 799 |
| **Total** | **3,670** |

---

## Project structure

```
ViT_Flower_Classifier_Pro/
├── data/flowers_raw/          raw dataset, 5 class folders
├── src/                       all source code (see file-by-file breakdown below)
├── notebooks/
│   └── train_test_vit_pro.ipynb   full GPU-ready walkthrough notebook
├── demo/
│   ├── index.html             interactive results page (self-contained)
│   └── data.json              precomputed run data embedded into the page
├── checkpoints/                model weights land here when you train (not committed — see below)
├── outputs/                    plots/exports land here when you train (not committed)
├── requirements.txt
└── .gitignore
```

---

## File-by-file: what every piece of code does

### `src/vit_scratch.py` — the Vision Transformer, built from primitives
Implements the architecture from *An Image is Worth 16x16 Words* (Dosovitskiy et al., 2020) using nothing but base `nn.Module` building blocks:
- **`PatchEmbedding`** — splits an image into fixed-size patches and linearly projects each one into an embedding vector, using a single strided `Conv2d` as an efficient patch-and-project operation.
- **`MultiheadSelfAttentionBlock`** — pre-norm multi-head self-attention (`nn.MultiheadAttention` under the hood), also returns attention weights so they can be inspected later.
- **`MLPBlock`** — the pre-norm feed-forward block (`Linear → GELU → Dropout → Linear → Dropout`) that follows attention in every transformer layer.
- **`TransformerEncoderBlock`** — wires attention and MLP blocks together with residual connections; optionally returns its attention weights for interpretability.
- **`ViT`** — the full model: learnable class token, learnable position embeddings, a stack of encoder blocks, and a classification head on the class token's final representation. `forward(..., return_attn=True)` returns every layer's attention weights alongside the logits.
- **`vit_tiny()` / `vit_base()`** — convenience constructors for a small (6-layer, 256-dim) and a full-size (12-layer, 768-dim) configuration.

### `src/vit_mae.py` — self-supervised pretraining via masked autoencoding
Implements a Masked Autoencoder in the style of He et al., 2021 (*Masked Autoencoders Are Scalable Vision Learners*), reusing `PatchEmbedding` and `TransformerEncoderBlock` from `vit_scratch.py`:
- **`random_masking`** — randomly selects a subset of patches to keep (default: 25%) and returns the visible patches plus the information needed to restore the original order later.
- **`MAEEncoder`** — a ViT encoder that only ever processes the *visible* patches (this is what makes MAE pretraining cheap — 75% of the sequence is simply never computed on).
- **`MAEDecoder`** — a small, separate transformer that takes the encoder's output, reinserts learnable "mask tokens" at every masked position, and predicts the raw pixel values for the missing patches.
- **`patchify`** — reshapes a batch of images into a sequence of flattened patches, used to build the reconstruction target.
- **`MaskedAutoencoderViT`** — ties encoder and decoder together, computes the reconstruction loss (mean-squared error over masked patches only, on per-patch-normalized pixel values), and exposes `encode_for_classification()` to pull out a usable embedding from the pretrained encoder after pretraining is done.

### `src/pretrain_mae.py` — the self-supervised pretraining loop
- **`UnlabeledImageDataset`** — walks the dataset directory and loads every image, deliberately discarding the class-folder labels — this stage never sees them.
- **`pretrain()`** — the training loop: forward pass computes the masked-reconstruction loss, backward pass updates the encoder and decoder, checkpoints the model after every epoch.
- **CLI entry point** (`--data_dir`, `--epochs`, `--batch_size`, `--mask_ratio`, `--checkpoint`, ...) — run this before fine-tuning to get a pretrained encoder that has never seen a single label.

### `src/vit_transfer.py` — the ImageNet-pretrained baseline
- **`build_pretrained_vit()`** — loads `torchvision.models.vit_b_16` with ImageNet-1k weights, freezes the whole backbone by default, and swaps in a fresh classification head sized for this dataset's 5 classes.
- **`unfreeze_last_n_blocks()`** — selectively unfreezes the last few transformer blocks (plus the final layer norm) so fine-tuning can adapt high-level features without retraining the entire network from ImageNet weights.

### `src/prepare_data.py` — stratified dataset splitting
- **`split_dataset()`** — for every class folder, shuffles the images (fixed seed for reproducibility) and splits them into train/val/test according to the given ratios (default 70/15/15), copying files into an `ImageFolder`-compatible directory layout.
- CLI entry point so it can be run standalone before any training script.

### `src/engine.py` — the supervised training/evaluation loop
- **`train_step()`** — one epoch of forward pass, loss computation, backward pass, optimizer step, and running accuracy over the training set.
- **`test_step()`** — the equivalent evaluation-only pass (no gradients) over a validation or test set.
- **`train()`** — the full training loop: cosine learning-rate scheduling, best-checkpoint saving (only when validation accuracy improves), and early stopping after a configurable number of non-improving epochs.

### `src/attention_rollout.py` — proper multi-layer attention interpretability
Implements attention rollout (Abnar & Zuidema, 2020) rather than just visualizing a single layer's raw attention map, which can be misleading on its own:
- **`compute_rollout()`** — recursively multiplies attention matrices across every layer (adding a residual identity term, since transformer blocks have skip connections), producing a single map that reflects how information actually flows from input patches to the output token through the whole network.
- **`rollout_attention_map()`** — runs a model, extracts all layers' attention weights, computes the rollout, and reshapes the class token's row into a 2D grid matching the image's patch layout — ready to overlay on the original image.

### `src/embedding_explorer.py` — representation analysis and image similarity
- **`extract_embeddings()`** — runs a model (either the classifier's class-token output or the MAE encoder's pooled output) over a dataloader and collects embeddings, labels, and file paths for every image.
- **`project_2d()`** — reduces embeddings to two dimensions with PCA or t-SNE, for visualization.
- **`nearest_neighbors()`** — cosine-similarity search: given a query embedding, ranks every other image in the corpus by similarity and returns the top-k matches.
- **`export_embedding_map()`** — packages embeddings, 2D coordinates, labels, and file paths into JSON, ready to feed into the interactive demo page.

### `src/train.py` — the CLI entry point that ties it all together
Parses arguments (`--model` chooses `scratch_tiny`, `scratch_base`, or `transfer`; plus `--epochs`, `--batch_size`, `--lr`, `--img_size`, `--unfreeze_blocks`, `--checkpoint`, `--seed`), builds the right model and data transforms for whichever regime was selected, builds dataloaders from `prepare_data.py`'s output, and runs the full `engine.train()` loop, followed by loss-curve plotting and test-set evaluation.

### `src/predict.py` — single-image inference
A minimal CLI (`--image`, `--checkpoint`) that loads a fine-tuned transfer model and returns the top-3 predicted classes with probabilities for one input image — the smallest possible "does the trained model actually work" check.

### `src/utils.py` — shared plotting, evaluation, and visualization helpers
- **`set_seed()`** — seeds Python, NumPy, and PyTorch (CPU and CUDA) for reproducibility.
- **`plot_loss_curves()`** — side-by-side loss and accuracy curves across training.
- **`evaluate_model()`** — runs a model over a test set and prints a full `sklearn` classification report plus a confusion matrix heatmap.
- **`predict_image()`** — loads a single image, runs inference, and displays it with its top-3 predicted classes and confidence scores.
- **`visualize_attention()`** — overlays a single layer's class-token attention map on the original image (the simpler, single-layer counterpart to `attention_rollout.py`'s multi-layer version).

### `notebooks/train_test_vit_pro.ipynb`
The complete, ordered walkthrough: environment setup → dataset split → data pipeline with augmentation → train the from-scratch ViT → train the fine-tuned ImageNet ViT → evaluate both on the held-out test set → visualize attention → run inference on a sample image → a results table to fill in after a full run. Written to run top-to-bottom on a GPU runtime (Colab: Runtime → Change runtime type → GPU).

### `demo/index.html` + `demo/data.json`
A self-contained, dependency-free interactive results page:
- **Attention rollout gallery** — one held-out test image per class, with the rollout heatmap overlaid, alongside the model's actual prediction and confidence.
- **MAE reconstruction plate** — an original / 75%-masked / reconstructed triptych for a real image from this project's pretraining run.
- **Embedding atlas** — a hoverable, filterable 2D scatter plot of real CLS embeddings (colored by species after the fact, not used to produce the embeddings).
- **Live similarity search** — click any of 20 sample thumbnails and see its five nearest neighbors by cosine similarity, computed live in the browser from the actual embedding vectors baked into `data.json` — no backend required.

Every number and image in the demo came from a real, timed run on a single CPU core in the development sandbox — nothing in it is a mockup.

---

## Running it yourself

```bash
pip install -r requirements.txt

# 1. Split the raw dataset
python src/prepare_data.py --source data/flowers_raw --output data/flowers_split

# 2. Regime A: from scratch
python src/train.py --model scratch_tiny --epochs 40 --batch_size 32

# 3. Regime B: self-supervised pretraining, then fine-tune
python src/pretrain_mae.py --data_dir data/flowers_raw --epochs 100 --batch_size 64
python src/train.py --model scratch_tiny --epochs 15 --batch_size 32   # fine-tune from the MAE checkpoint

# 4. Regime C: ImageNet transfer learning
python src/train.py --model transfer --epochs 15 --batch_size 32

# 5. Try it on one image
python src/predict.py --image path/to/image.jpg --checkpoint checkpoints/best_model.pth
```

Or just open `notebooks/train_test_vit_pro.ipynb` in Colab with a GPU runtime and run every cell in order.

## What's verified vs. what needs a GPU

Everything shown in `demo/index.html` came from real, timed runs on a **1-core CPU** — every stage of the pipeline (training step, checkpointing, evaluation, MAE pretraining, attention rollout, embedding projection) was executed and its actual output captured, not simulated:

- From-scratch ViT: 120 images, 12 epochs → **56% test accuracy** vs. a 20% random baseline
- MAE pretraining: 120 unlabeled images, 10 epochs → reconstruction loss **1.15 → 0.86**

Full-dataset training (all 3,670 images, enough epochs to actually converge, all three regimes) needs a GPU to finish in a reasonable time — this repo intentionally doesn't ship pretrained weight files (they exceed GitHub's 100MB per-file limit) or claim results it didn't produce. `notebooks/train_test_vit_pro.ipynb` is ready to run the full comparison end-to-end on a free Colab GPU.

## Why compare training regimes instead of just training on more data

The obvious move for a bigger number — download more images, train longer — doesn't answer anything interesting about *why* the model works. Holding the dataset fixed and varying only how the encoder is trained isolates three separate contributions to a transformer's competence: how much comes from the architecture itself, how much from self-supervised structure already present in unlabeled pixels, and how much is simply borrowed from a much larger, unrelated dataset (ImageNet) via transfer learning.

## License and attribution

**Code** in `src/`, `notebooks/`, and `demo/` is provided for educational/portfolio use — add an explicit code license (MIT is a common default) if you want to state terms formally.

**Dataset** (`data/flowers_raw/`) is Google's official `flower_photos` set, used in TensorFlow's own tutorials, licensed under [CC-BY 2.0](https://creativecommons.org/licenses/by/2.0/). Every image originally came from Flickr, contributed by individual photographers who released their work under CC-BY. `data/flowers_raw/LICENSE.txt` is Google's original, unmodified attribution file — it lists the photographer and Flickr source link for every one of the 3,670 images, exactly as published with the dataset. CC-BY requires that attribution to survive with the images; this repo keeps it intact rather than stripping it out.
