# module/utils/data_module.py
import os, random, itertools
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from .datasets import S1200        # your existing dataset class

# ---------------------------------------------------------------- #
# Lightning DataModule
# ---------------------------------------------------------------- #
class fMRIDataModule(pl.LightningDataModule):
    # ------------- CLI flags ----------------------------
    @staticmethod
    def add_data_specific_args(parent_parser):
        p = parent_parser.add_argument_group("Data")
        p.add_argument("--data_dir", default=".", type=str)
        p.add_argument("--batch_size", default=8,  type=int)
        p.add_argument("--num_workers", default=4, type=int)

        p.add_argument("--dataset_type", choices=["rest", "task"], default="rest")
        p.add_argument("--sequence_length",    type=int, default=20)
        p.add_argument("--stride_between_seq", type=int, default=1)
        p.add_argument("--stride_within_seq",  type=int, default=1)

        p.add_argument("--contrastive", action="store_true",
                       help="Return two augmented clips for NT-Xent")
        p.add_argument("--with_voxel_norm",     action="store_true")
        p.add_argument("--shuffle_time_sequence", action="store_true")

        # NEW ---------------------------------------------------------------
        p.add_argument("--grayordinates", action="store_true",
                       help="Treat ROI axis as 91 282 gray-ordinates")
        p.add_argument("--label_scaling_method",
                       choices=["standardization", "minmax", "none"],
                       default="standardization")
        p.add_argument("--balanced_sampling", action="store_true",
                       help="Resample subjects so each appears equally in a epoch")
        return parent_parser

    # ------------- ctor ---------------------------------
    def __init__(self, **hparams):
        super().__init__()
        self.save_hyperparameters()
        self.dataset_kwargs = dict(
            sequence_length   = hparams["sequence_length"],
            stride_between_seq= hparams["stride_between_seq"],
            stride_within_seq = hparams["stride_within_seq"],
            contrastive       = hparams["contrastive"],
            with_voxel_norm   = hparams["with_voxel_norm"],
            shuffle_time_sequence = hparams["shuffle_time_sequence"],
        )

        # decide number of ROIs automatically
        self.num_rois = 91_282 if hparams["grayordinates"] else None
        self.scaler = None                     # set in setup()

    # ------------- setup --------------------------------
    def setup(self, stage=None):
        root = lambda split: os.path.join(self.hparams["data_dir"], split)

        # -------- load splits
        self.train_dataset = S1200(root=root("train"),
                                   dataset_type=self.hparams["dataset_type"],
                                   num_rois=self.num_rois,
                                   **self.dataset_kwargs)
        self.val_dataset   = S1200(root=root("val"),
                                   dataset_type=self.hparams["dataset_type"],
                                   num_rois=self.num_rois,
                                   **self.dataset_kwargs)
        self.test_dataset  = S1200(root=root("test"),
                                   dataset_type=self.hparams["dataset_type"],
                                   num_rois=self.num_rois,
                                   **self.dataset_kwargs)

        # -------- label scaling (regression only)
        if self.hparams["label_scaling_method"] != "none" \
           and self.train_dataset.task_type == "regression":
            y = np.concatenate([self.train_dataset.targets,
                                self.val_dataset.targets])
            scaler_cls = StandardScaler if self.hparams["label_scaling_method"] == "standardization"\
                         else MinMaxScaler
            self.scaler = scaler_cls().fit(y.reshape(-1, 1))
            for ds in (self.train_dataset, self.val_dataset, self.test_dataset):
                ds.targets = self.scaler.transform(ds.targets.reshape(-1, 1)).ravel()

        # expose for the model (needed for inverse-transform at test time)
        self.train_dataset.target_values = torch.tensor(
            self.train_dataset.targets, dtype=torch.float)

    # ------------- loaders ------------------------------
    def train_dataloader(self):
        if self.hparams["balanced_sampling"]:
            subj_ids = self.train_dataset.subject_ids
            indices_by_subj = {s: [] for s in subj_ids}
            for idx, s in enumerate(subj_ids):
                indices_by_subj[s].append(idx)
            # round-robin sampler
            rr = list(itertools.chain.from_iterable(
                zip(*indices_by_subj.values())))
            sampler = RandomSampler(rr, replacement=False)
        else:
            sampler = RandomSampler(self.train_dataset)

        return DataLoader(self.train_dataset,
                          batch_size=self.hparams["batch_size"],
                          sampler=sampler,
                          num_workers=self.hparams["num_workers"],
                          pin_memory=True,
                          persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.hparams["batch_size"],
                          sampler=SequentialSampler(self.val_dataset),
                          num_workers=self.hparams["num_workers"],
                          pin_memory=True,
                          persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.hparams["batch_size"],
                          sampler=SequentialSampler(self.test_dataset),
                          num_workers=self.hparams["num_workers"],
                          pin_memory=True,
                          persistent_workers=True)
