import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .datasets import S1200

class fMRIDataModule(pl.LightningDataModule):
    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data")
        parser.add_argument(
            "--data_dir", type=str, default=".",
            help="Root directory containing 'train/', 'val/', and 'test/' subfolders"
        )
        parser.add_argument("--batch_size",    type=int, default=8)
        parser.add_argument("--num_workers",   type=int, default=4)
        parser.add_argument(
            "--dataset_type", choices=["rest","task"], default="rest",
            help="Whether to load resting-state or task fMRI"
        )
        parser.add_argument("--sequence_length",     type=int, default=20)
        parser.add_argument("--stride_between_seq",  type=int, default=1)
        parser.add_argument("--stride_within_seq",   type=int, default=1)
        parser.add_argument("--contrastive",         action="store_true")
        parser.add_argument("--with_voxel_norm",     action="store_true")
        parser.add_argument("--shuffle_time_sequence", action="store_true")
        return parent_parser

    def __init__(
        self,
        data_dir: str,
        subject_dict: dict,
        batch_size: int,
        num_workers: int,
        dataset_type: str,
        sequence_length: int = 20,
        stride_between_seq: int = 1,
        stride_within_seq: int = 1,
        contrastive: bool = False,
        with_voxel_norm: bool = False,
        shuffle_time_sequence: bool = False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.subject_dict = subject_dict
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_type = dataset_type
        self.dataset_kwargs = {
            "sequence_length": sequence_length,
            "stride_between_seq": stride_between_seq,
            "stride_within_seq": stride_within_seq,
            "contrastive": contrastive,
            "with_voxel_norm": with_voxel_norm,
            "shuffle_time_sequence": shuffle_time_sequence,
        }

    def setup(self, stage=None):
        # Called by Lightning to set up train/val/test datasets
        self.train_dataset = S1200(
            root=os.path.join(self.data_dir, "train"),
            subject_dict=self.subject_dict,
            dataset_type=self.dataset_type,
            **self.dataset_kwargs
        )
        self.val_dataset = S1200(
            root=os.path.join(self.data_dir, "val"),
            subject_dict=self.subject_dict,
            dataset_type=self.dataset_type,
            **self.dataset_kwargs
        )
        self.test_dataset = S1200(
            root=os.path.join(self.data_dir, "test"),
            subject_dict=self.subject_dict,
            dataset_type=self.dataset_type,
            **self.dataset_kwargs
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )
