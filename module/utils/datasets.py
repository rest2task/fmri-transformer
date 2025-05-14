import os
import random
import torch
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    """
    Base dataset for fMRI sequences.
    Supports optional contrastive pairs.
    """
    def __init__(
        self,
        root: str,
        subject_dict: dict,
        dataset_type: str = 'rest',
        sequence_length: int = 20,
        stride_between_seq: int = 1,
        stride_within_seq: int = 1,
        contrastive: bool = False,
        with_voxel_norm: bool = False,
        shuffle_time_sequence: bool = False,
    ):
        assert dataset_type in ('rest', 'task')
        self.modality = 0 if dataset_type == 'rest' else 1
        self.root = root
        self.subject_dict = subject_dict
        self.sequence_length = sequence_length
        self.stride_between_seq = stride_between_seq
        self.stride_within_seq = stride_within_seq
        self.contrastive = contrastive
        self.with_voxel_norm = with_voxel_norm
        self.shuffle_time_sequence = shuffle_time_sequence

        # Total frames per sample and step size
        self.sample_duration = self.sequence_length * self.stride_within_seq
        self.stride = max(int(self.stride_between_seq * self.sample_duration), 1)

        # Precompute indices
        self.data = self._set_data()

    def _set_data(self):
        """
        Build a list of tuples describing each valid segment:
        (subject_path, start_frame, num_frames, (sex, target))
        """
        raise NotImplementedError

    def load_sequence(self, subject_path: str, start: int, num_frames: int):
        """
        Load a temporal sequence of .pt frames from disk.
        Returns either a single tensor [1, C, X, Y, Z, T] or
        a list [seq1, seq2] if contrastive=True.
        """
        # Select frame indices
        if self.shuffle_time_sequence and not self.contrastive:
            frames = random.sample(range(num_frames), self.sequence_length)
        else:
            frames = range(start, start + self.sample_duration, self.stride_within_seq)

        # Load the primary sequence
        seq = []
        for f in frames:
            path = os.path.join(subject_path, f"frame_{f}.pt")
            seq.append(torch.load(path).unsqueeze(0))

        # Optionally append normalization stats
        if self.with_voxel_norm:
            seq.append(torch.load(os.path.join(subject_path, "voxel_mean.pt")).unsqueeze(0))
            seq.append(torch.load(os.path.join(subject_path, "voxel_std.pt")).unsqueeze(0))

        seq = torch.cat(seq, dim=4)  # shape [1, C, X, Y, Z, T]

        if not self.contrastive:
            return seq

        # Build a negative sample for contrastive learning
        full = set(range(num_frames - self.sample_duration + 1))
        exclude = set(range(start - self.sample_duration, start + self.sample_duration))
        choices = list(full - exclude)
        rand_start = random.choice(choices)

        seq2 = []
        for f in range(rand_start, rand_start + self.sample_duration, self.stride_within_seq):
            path = os.path.join(subject_path, f"frame_{f}.pt")
            seq2.append(torch.load(path).unsqueeze(0))
        if self.with_voxel_norm:
            seq2.append(torch.load(os.path.join(subject_path, "voxel_mean.pt")).unsqueeze(0))
            seq2.append(torch.load(os.path.join(subject_path, "voxel_std.pt")).unsqueeze(0))
        seq2 = torch.cat(seq2, dim=4)

        return seq, seq2

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns:
          - fmri: tensor [1, X, Y, Z, T] or pair when contrastive
          - modality: 0 or 1
          - target: float
          - (fmri2, modality2) if contrastive
        """
        raise NotImplementedError


class S1200(BaseDataset):
    """
    Dataset for the S1200 release:
    Assumes folder structure:
      root/
        img/
          <subject_id>/
            frame_0.pt, frame_1.pt, ...
            voxel_mean.pt, voxel_std.pt (optional)
    subject_dict maps ID -> (sex, target_value)
    """
    def _set_data(self):
        data = []
        img_root = os.path.join(self.root, "img")

        for subj, (sex, target) in self.subject_dict.items():
            subj_path = os.path.join(img_root, subj)
            # Count frames
            num_frames = len([f for f in os.listdir(subj_path) if f.startswith("frame_")])
            max_start = num_frames - self.sample_duration + 1

            for start in range(0, max_start, self.stride):
                data.append((subj_path, start, num_frames, (sex, target)))

        return data

    def __getitem__(self, idx):
        subj_path, start, num_frames, (sex, target) = self.data[idx]
        seq = self.load_sequence(subj_path, start, num_frames)

        # Pad & permute: [1,C,X,Y,Z,T] -> [C,T,X,Y,Z]
        def pad_and_permute(x: torch.Tensor):
            # Remove batch dim, permute, then pad
            x = x.permute(0, 4, 1, 2, 3)  # [1,T,C,X,Y,Z]
            bg = x.flatten()[0].item()
            pad = (0, 0, 0, 0, 0, 0)  # adjust if needed
            x = torch.nn.functional.pad(x, pad, value=bg)
            return x[0]  # [T,C,X,Y,Z]

        if self.contrastive:
            seq1, seq2 = seq
            fmri1 = pad_and_permute(seq1)
            fmri2 = pad_and_permute(seq2)
            return {
                "fmri1": fmri1,
                "modality1": torch.tensor(self.modality, dtype=torch.long),
                "fmri2": fmri2,
                "modality2": torch.tensor(self.modality, dtype=torch.long),
                "target": torch.tensor(target, dtype=torch.float32),
                "sex": sex,
            }
        else:
            fmri = pad_and_permute(seq)
            return {
                "fmri": fmri,
                "modality": torch.tensor(self.modality, dtype=torch.long),
                "target": torch.tensor(target, dtype=torch.float32),
                "sex": sex,
            }
