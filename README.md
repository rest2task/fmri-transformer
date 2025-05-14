# R2T‑Net

**R2T‑Net** is a PyTorch‑Lightning framework for learning a *single* 2 048‑dimensional “brain‑signature” vector from both resting‑state (rs‑fMRI) and task (t‑fMRI) data.  
A Transformer encoder (Step 1) turns each 4‑D volume or ROI‑timeseries into that signature; a contrastive NT‑Xent loss (Step 2) makes signatures from the *same* person / different modalities attract, while pushing *different* people apart.

---

## 🌐 Motivation

Most pipelines treat rs‑fMRI and t‑fMRI separately—often collapsing t‑fMRI into static contrast maps. That discards temporal information and limits subject‑specific modelling.  
**R2T‑Net** jointly embeds both modalities so the signature is:

* **Modality‑invariant** (rest ⇄ task)  
* **Person‑specific** (positive pairs = same subject)  
* **Time‑stable** (good test–retest reliability)  
* **Predictive** for cognition / behaviour from rs‑fMRI alone  

---

## 🧱 Backbone Flexibility

Any encoder that outputs a `[B, embed_dim]` feature can plug in by editing  
`module/models/load_model.py`.

| Category           | Examples you can register |
|--------------------|---------------------------|
| **Transformer‑4D** | `swin4d_ver7` (default), `vit`, `transformer2d`, TimeSformer‑style |
| **3‑D CNN**        | `resnet3d`, `densenet3d`, `r3d_18`, `unet3d` |
| **Hybrid**         | CNN + GRU, Perceiver IO, Temporal‑U‑Net |

(The repo ships with Swin‑4D and a fallback to any ViT from `timm`.)

---

## 🔧 Key Features

| Feature | Details |
|---------|---------|
| **Flexible input** | Raw 4‑D volumes `[C,H,W,D,T]`, grayordinates `[91 282,T]`, or parcellated ROI `[V,T]` |
| **Transformer or CNN** | Swap backbone via `--model`. External 3‑D patch‑embedding provided for ViT‑style nets. |
| **Contrastive + Supervised** | Add `--contrastive` for NT‑Xent; downstream regression / classification head is always present. |
| **Cosine‑Warmup scheduler** | `--use_scheduler` activates `CosineAnnealingWarmUpRestarts`. |
| **Metrics** | Pearson / MSE / MAE for regression, Balanced‑Acc / AUROC for classification (Lightning’s `torchmetrics`). |

---

## 📁 Directory Layout

```

R2TNet/
├── train.py            # training / validation / test
├── inference.py        # batch inference on rs‑fMRI
│
├── module/
│   ├── r2tnet.py       # LightningModule (encoder + heads + loss)
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
└── logs/               # created automatically (checkpoints + TensorBoard)

````

---

## 🚀 Quick Start

### 1 · Install

```bash
pip install pytorch-lightning torch timm einops torchmetrics scikit-learn
````

### 2 · Prepare data

```
data/S1200/
└── img/100307/frame_0.pt  frame_1.pt ...
```

Each `frame_*.pt` is a tensor `[C,H,W,D]` for one TR.

### 3 · Supervised training

```bash
python train.py \
  --data_dir data/S1200 \
  --dataset_type rest \
  --batch_size 4 \
  --model swin4d_ver7 \
  --max_epochs 50
```

### 4 · Contrastive pre‑training

```bash
python train.py \
  --data_dir data/S1200 \
  --dataset_type rest \
  --contrastive --use_scheduler \
  --model swin4d_ver7
```

### 5 · Inference (rs‑fMRI → cognitive score)

```bash
python inference.py \
  --ckpt logs/epoch03-valid_loss=0.2100.ckpt \
  --input_dir data/S1200/img/ \
  --output rs_predictions.csv
```

---

## 🧠 Applications

* Predict fluid intelligence, memory, personality from rs‑fMRI
* Replace long multi‑task scans with one short resting scan
* Transfer‑learning to clinical cohorts (e.g., ADNI, ABCD)

---

## 📜 License

Released under the MIT License – free for research and commercial use.
