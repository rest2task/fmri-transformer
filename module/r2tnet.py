# module/r2tnet.py
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import pytorch_lightning as pl
from torchmetrics.classification import BinaryAccuracy
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from .models.load_model   import load_model
from .utils.patch_embedding import PositionalEmbedding3D, PositionalEmbedding1D
from .utils.lr_scheduler   import CosineAnnealingWarmUpRestarts


# --------------------------------------------------------------------- #
# 1.  Main LightningModule
# --------------------------------------------------------------------- #
class R2TNet(pl.LightningModule):
    """
    R2T‑Net
    -------
    • Step‑1  Transformer encoder (ViT / Swin‑4D) -> 2 048‑dim brain signature  
    • Step‑2  NT‑Xent contrastive learning (rest vs. task pairs)  
    • Also supports supervised regression / classification
    """
    def __init__(self, data_module, **hparams):
        super().__init__()
        self.save_hyperparameters(ignore=['data_module'])

        # --------------------------------- data‑driven target scaling
        targets = data_module.train_dataset.target_values
        scaler_cls = StandardScaler if self.hparams.label_scaling_method == 'standardization' else MinMaxScaler
        self.scaler = scaler_cls().fit(targets)

        # --------------------------------- backbone
        self.use_external_patch = not self.hparams.model.startswith("swin")
        if self.use_external_patch:
            self.patch_embed = PositionalEmbedding3D(
                img_size=self.hparams.img_size,
                patch_size=self.hparams.patch_size,
                embed_dim=self.hparams.embed_dim
            )
        self.encoder = load_model(self.hparams.model, self.hparams)

        # --------------------------------- ROI encoder (1‑D transformer)
        self.roi_encoder = None
        if self.hparams.num_rois > 0:
            self.pos1d = PositionalEmbedding1D(self.hparams.num_rois, self.hparams.embed_dim)
            layer = nn.TransformerEncoderLayer(
                d_model=self.hparams.embed_dim,
                nhead=self.hparams.roi_heads,
                dim_feedforward=self.hparams.roi_ff,
                dropout=self.hparams.head_dropout
            )
            self.roi_encoder = nn.TransformerEncoder(layer, self.hparams.roi_layers)

        # --------------------------------- heads
        self.modality_emb = nn.Embedding(2, self.hparams.embed_dim)
        self.sig_head = self._make_mlp(self.hparams.embed_dim,
                                       self.hparams.signature_size,
                                       self.hparams.head_dropout,
                                       self.hparams.head_version)
        self.pred_head = self._make_mlp(self.hparams.signature_size,
                                        1,
                                        self.hparams.head_dropout,
                                        self.hparams.head_version)

        self.temp = self.hparams.temperature   # NT‑Xent temperature

    # ------------------------------- helpers
    @staticmethod
    def _make_mlp(in_dim, out_dim, dropout, version):
        if version == "simple":
            return nn.Linear(in_dim, out_dim)

        layers = []
        if version == "v1":
            layers += [nn.Linear(in_dim, 512), nn.GELU(), nn.Dropout(dropout), nn.Linear(512, out_dim)]
        elif version == "v2":
            layers += [
                nn.Linear(in_dim, 1024), nn.BatchNorm1d(1024), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(1024, out_dim)
            ]
        else:
            raise ValueError(f"unknown head_version '{version}'")
        return nn.Sequential(*layers)

    # identity‑augmentation placeholder (replace with MONAI or TorchVision later)
    def augment(self, x):  # noqa: D401
        return x

    # ------------------------------- forward
    def forward(self, x: Tensor, modality: Tensor):
        # volumetric
        if x.ndim == 6:
            if self.use_external_patch:
                x = self.patch_embed(x)                # [B, N, embed_dim]
                feats = self.encoder(x)                # ViT/Transformer2D
            else:
                feats = self.encoder(x)                # Swin4D handles patches internally
        # ROI/vertex
        elif x.ndim == 3 and self.roi_encoder is not None:
            y = self.pos1d(x.permute(2, 0, 1))         # [T, B, V] → add pos‑emb
            feats = self.roi_encoder(y).mean(dim=0)    # global temporal average
        else:
            raise ValueError(f"Unsupported input shape {x.shape}")

        feats = feats + self.modality_emb(modality)
        sig   = self.sig_head(feats)
        logits= self.pred_head(sig).squeeze(-1)
        return sig, logits

    # ------------------------------- training‑step
    def training_step(self, batch, _):
        # contrastive
        if self.hparams.contrastive:
            z1,_ = self(batch['fmri1'], batch['modality1'])
            z2,_ = self(batch['fmri2'], batch['modality2'])
            z1, z2 = map(lambda z: F.normalize(z, dim=1), (z1, z2))
            sim   = torch.mm(z1, z2.T) / self.temp
            labels= torch.arange(sim.size(0), device=sim.device)
            loss  = 0.5 * (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels))
            self.log("train_contrastive_loss", loss)
            return loss

        # supervised
        logits, target = self._compute(batch, augment=True)
        loss = self._supervised_loss(logits, target)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    # ------------------------------- helpers for supervision
    def _compute(self, batch, augment=False):
        x, y, m = batch['fmri'], batch['target'].float().squeeze(-1), batch['modality']
        x = self.augment(x) if augment else x
        _, logits = self(x, m)
        if self.hparams.downstream_task_type != "classification":
            if self.hparams.label_scaling_method == "standardization":
                y = (y - self.scaler.mean_[0]) / self.scaler.scale_[0]
            else:
                rng = self.scaler.data_max_[0] - self.scaler.data_min_[0]
                y = (y - self.scaler.data_min_[0]) / rng
        return logits, y

    def _supervised_loss(self, logits, target):
        if self.hparams.downstream_task_type == "classification":
            self.log("train_acc",
                     BinaryAccuracy()(torch.sigmoid(logits), target.int()),
                     prog_bar=True)
            return F.binary_cross_entropy_with_logits(logits, target)
        return F.mse_loss(logits, target)

    # ------------------------------- optim / sched
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(),
                                lr=self.hparams.learning_rate,
                                weight_decay=self.hparams.weight_decay)
        if not self.hparams.use_scheduler:
            return opt

        sched = CosineAnnealingWarmUpRestarts(
            opt,
            first_cycle_steps=self.hparams.warmup_steps,
            max_lr=self.hparams.learning_rate,
            min_lr=self.hparams.min_lr,
            warmup_steps=self.hparams.warmup_steps,
            gamma=self.hparams.gamma
        )
        return [opt], [{"scheduler": sched, "interval": "step"}]

    # ------------------------------- CLI flags
    @staticmethod
    def add_model_specific_args(parent: ArgumentParser):
        p = ArgumentParser(parents=[parent], add_help=False,
                           formatter_class=ArgumentDefaultsHelpFormatter)

        # backbone / patching
        back = p.add_argument_group("Backbone")
        back.add_argument("--model",
                          choices=["swin4d_ver7", "swin4d_base", "vit", "transformer2d"],
                          default="swin4d_ver7")
        back.add_argument("--in_chans",  type=int, default=1)
        back.add_argument("--img_size",  nargs=5, type=int, default=[96, 96, 96, 1, 400])
        back.add_argument("--patch_size",nargs=5, type=int, default=[2, 2, 2, 1, 20])
        back.add_argument("--embed_dim", type=int, default=128)

        # ROI transformer
        roi = p.add_argument_group("ROI")
        roi.add_argument("--num_rois",   type=int, default=0)
        roi.add_argument("--roi_layers", type=int, default=2)
        roi.add_argument("--roi_heads",  type=int, default=8)
        roi.add_argument("--roi_ff",     type=int, default=2048)

        # heads
        head = p.add_argument_group("Heads")
        head.add_argument("--head_dropout", type=float, default=0.1)
        head.add_argument("--head_version",
                          choices=["simple", "v1", "v2"], default="v1")
        head.add_argument("--signature_size", type=int, default=2048)
        head.add_argument("--hidden_dim",     type=int, default=512)

        # contrastive
        contr = p.add_argument_group("Contrastive")
        contr.add_argument("--contrastive", action="store_true")
        contr.add_argument("--temperature", type=float, default=0.1)

        # training hyper‑params
        train = p.add_argument_group("Training")
        train.add_argument("--learning_rate", type=float, default=1e-3)
        train.add_argument("--weight_decay",  type=float, default=1e-2)
        train.add_argument("--use_scheduler", action="store_true")
        train.add_argument("--warmup_steps",  type=int, default=1000)
        train.add_argument("--min_lr",        type=float, default=1e-6)
        train.add_argument("--gamma",         type=float, default=0.99)

        # downstream task config
        down = p.add_argument_group("Downstream")
        down.add_argument("--downstream_task_type",
                          choices=["regression", "classification"],
                          default="regression")
        down.add_argument("--label_scaling_method",
                          choices=["standardization", "minmax"],
                          default="standardization")

        return p
