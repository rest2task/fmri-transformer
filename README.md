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
│   │   ├── load_model.py
│   │   ├── swin4d_transformer_ver7.py
│   │   └── swin_transformer.py
│   └── utils/
│       ├── data_module.py
│       ├── datasets.py
│       ├── patch_embedding.py
│       └── lr_scheduler.py
│
└── logs/                   # auto‑generated (TensorBoard & checkpoints)

````

---


## 🚀 Quick Start

Train and evaluate on 4D fMRI, ROI series, or grayordinates — all from one CLI.


### 1 · Install

```bash
# PyTorch 2.x with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Core dependencies
pip install pytorch-lightning timm einops torchmetrics scikit-learn

# Optional extras
pip install monai pandas matplotlib
```


### 2 · Prepare your data

```
data/S1200/
├── img/
│   ├── 100307/
│   │   ├── frame_0.pt
│   │   ├── frame_1.pt
│   │   └── ...
│   └── 103414/
└── meta/
    ├── subject_dict.json        # {"100307": [0, 83.2], ...}
    └── splits.json              # {"train": [...], "val": [...], "test": [...]}
```

* Each `frame_*.pt` is a tensor shaped `[C, H, W, D]` (for 4D fMRI).
* ROIs or grayordinates should be saved as `[V]` tensors per frame.
* Optional: `voxel_mean.pt` and `voxel_std.pt` for normalization.


### 3 · Training Paradigms

| Mode                | NT-Xent | Supervised | CLI Flags                                                |
| ------------------- | ------- | ---------- | -------------------------------------------------------- |
| **Self-supervised** | ✅       | ❌ (frozen) | `--contrastive --pretraining --freeze_head`              |
| **Full fine-tune**  | ✅       | ✅          | `--contrastive` *(default)*                              |
| **Linear probe**    | ❌       | ✅ (frozen) | `--freeze_encoder --downstream_task_type classification` |

> ⚠️ Omitting `--contrastive` disables NT-Xent loss (core to R2T‑Net).


### 4 · Example Commands

#### A. Self-supervised Pre-training

```bash
python train.py \
  --data_dir data/S1200 \
  --dataset_type rest \
  --contrastive --pretraining --freeze_head \
  --model swin4d_ver7 \
  --batch_size 8 --max_epochs 50 \
  --use_scheduler --total_steps 20000
```

#### B. Fine-tune with Labels (Regression)

```bash
python train.py \
  --data_dir data/S1200 \
  --dataset_type rest \
  --contrastive \
  --load_model logs/last.ckpt \
  --downstream_task_type regression \
  --label_scaling_method standardization \
  --model swin4d_ver7 \
  --batch_size 4 --max_epochs 30 --use_scheduler
```

#### C. Evaluate/Test Only

```bash
python train.py \
  --test_only --test_ckpt logs/epoch02-valid_loss=0.2100.ckpt \
  --data_dir data/S1200
```


### 5 · Inference (e.g., rs-fMRI → prediction)

```bash
python inference.py \
  --ckpt logs/epoch03-valid_loss=0.2100.ckpt \
  --input_dir data/S1200/img \
  --input_kind vol \
  --output rs_predictions.csv
```

Output: CSV file with columns `subject_id,prediction`.


### 6 · Recommended Flags

| Flag                              | Purpose                                        |
| --------------------------------- | ---------------------------------------------- |
| `--precision 16`                  | Mixed precision — saves memory                 |
| `--accumulate_grad_batches 2`     | Gradient accumulation (for small GPUs)         |
| `--resume_from_checkpoint ...`    | Resume training from last/best                 |
| `--balanced_sampling`             | Ensures equal subject exposure per epoch       |
| `--num_rois 360 --input_kind roi` | Switch to ROI-based input (instead of volumes) |
| `--grayordinates`                 | Use 91,282-dim grayordinate inputs             |


### 7 · Troubleshooting

| Symptom             | Likely Fix                                     |
| ------------------- | ---------------------------------------------- |
| **CUDA OOM**        | Reduce `--batch_size`, or add `--precision 16` |
| **Loss = NaN**      | Check `--total_steps` ≫ `--warmup_pct`         |
| **AUROC = 0.5**     | Check for constant or missing labels           |
| **Slow dataloader** | Increase `--num_workers`; pre-convert to `.pt` |


### 8 · Next Steps

* ✅ Try other encoders: `--model resnet3d18`, `vit`, `cnn_gru`, `temporal_unet`
* ✅ Multi-GPU: `--accelerator gpu --devices 8 --strategy ddp`
* ✅ Export to ONNX: `python export_onnx.py --ckpt logs/last.ckpt`

---

## 🧠 Applications

* Predict fluid intelligence, memory, personality from *resting scans only*
* Shorten scan protocols in population studies
* Transfer‑learn to clinical cohorts (ADNI, ABCD, UK Biobank)
