# R2T‑Net

**R2T‑Net** is a PyTorch‑Lightning framework that turns any 4‑D fMRI sample—resting‑state (rs‑fMRI) or task (t‑fMRI)—into a single **1024‑dimensional brain‑signature**.  
A Transformer encoder (Step 1) creates the signature; an NT‑Xent contrastive loss (Step 2) makes signatures from the *same* subject and *different* modalities attract, while pushing signatures from *different* subjects apart.  
A small supervised head can then predict cognition or behaviour from the signature.

---

## 🌐 Motivation

Traditional pipelines handle rs‑fMRI and t‑fMRI separately, often compressing t‑fMRI into static contrast maps—losing temporal dynamics and personalization.  
**R2T‑Net** instead:

* Learns a **modality‑invariant** embedding (rest ⇄ task).  
* Produces **person‑specific** vectors (positive pairs = same subject).  
* Shows good **test–retest stability**.  
* Lets you predict behaviour from *resting scans only*, cutting scan time.

---

## 🧱 Backbone Flexibility

Edit one line in `module/models/load_model.py` to plug in any encoder that outputs a `[B, embed_dim]` feature:

| Category           | Examples |
|--------------------|----------|
| Transformer‑4D     | `swin4d_ver7` (default) ·  ViT · TimeSformer |
| 3‑D CNN            | 3‑D ResNet, 3‑D DenseNet, UNet‑3D |
| Hybrid             | CNN + GRU, Perceiver IO, Temporal‑U‑Net |

---

## 🔧 Key Features

| Feature | Details |
|---------|---------|
| **Input‑agnostic** | Raw 4‑D volumes `[C,H,W,D,T]`, grayordinates `[91 282,T]`, or ROI series `[V,T]` |
| **Always contrastive** | NT‑Xent runs in every mode; you choose to freeze or unfreeze the prediction head |
| **Cosine‑warmup scheduler** | Enable with `--use_scheduler` |
| **Metrics out‑of‑the‑box** | Pearson / MSE / MAE (regression), Balanced‑Acc / AUROC (classification) |
| **Runs on a single GPU** | Batch accumulation + mixed precision available |

---

## 📁 Directory Layout

```

R2TNet/
├── train.py                # train / validate / test
├── inference.py            # batch inference on rs‑fMRI
│
├── module/
│   ├── r2tnet.py           # LightningModule (encoder + heads + losses)
│   ├── models/
│   │   ├── load\_model.py
│   │   ├── swin4d\_transformer\_ver7.py
│   │   └── swin\_transformer.py
│   └── utils/
│       ├── data\_module.py
│       ├── datasets.py
│       ├── patch\_embedding.py
│       └── lr\_scheduler.py
│
└── logs/                   # auto‑generated (TensorBoard & checkpoints)

````

---

## 🚀 Quick Start

### 1 · Install

```bash
pip install pytorch-lightning torch timm einops torchmetrics scikit-learn
````

### 2 · Prepare your data

```
data/S1200/
└── img/100307/frame_0.pt  frame_1.pt ...
```

Each `frame_*.pt` is `[C,H,W,D]` for one TR.

### 3 · Training Paradigms

| Mode                         | Contrastive | Supervised Head | Command flag                  |
| ---------------------------- | ----------- | --------------- | ----------------------------- |
| **Pre‑training**             | ✅           | ❌ (frozen)      | `--contrastive --pretraining` |
| **Full fine‑tune** (default) | ✅           | ✅               | `--contrastive`               |

> If you omit `--contrastive`, the script still runs but you lose the main advantage of R2T‑Net.

#### A. Self‑supervised pre‑training

```bash
python train.py \
  --data_dir data/S1200 \
  --dataset_type rest \
  --contrastive --pretraining \
  --model swin4d_ver7 \
  --batch_size 8 --max_epochs 50 --use_scheduler
```

#### B. Fine‑tune with labels (keeps contrastive loss)

```bash
python train.py \
  --data_dir data/S1200 \
  --dataset_type rest \
  --contrastive \
  --load_model logs/pretrain.ckpt \
  --downstream_task_type regression \
  --model swin4d_ver7 \
  --batch_size 4 --max_epochs 30 --use_scheduler
```

### 4 · Inference (rs‑fMRI → score)

```bash
python inference.py \
  --ckpt logs/epoch03-valid_loss=0.2100.ckpt \
  --input_dir data/S1200/img/ \
  --output rs_predictions.csv
```

---

## 🧠 Applications

* Predict fluid intelligence, memory, personality from *resting scans only*
* Shorten scan protocols in population studies
* Transfer‑learn to clinical cohorts (ADNI, ABCD, UK Biobank)
