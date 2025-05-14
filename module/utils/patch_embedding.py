# module/utils/patch_embedding.py
"""
Patch + positional embedding helpers for R2T‑Net
-----------------------------------------------
* PositionalEmbedding3D : splits [B, C, H, W, D, T] into 3‑D+time patches,
                          projects each patch to embed_dim, and adds a learned
                          token‑wise positional embedding.
* PositionalEmbedding1D : adds a learned embedding across ROI/vertex tokens.
"""
import torch
from torch import nn
from einops import rearrange


# --------------------------------------------------------------------- #
# 3‑D + temporal patch embedding
# --------------------------------------------------------------------- #
class PositionalEmbedding3D(nn.Module):
    def __init__(self, img_size, patch_size, embed_dim):
        """
        Args
        ----
        img_size   : list/tuple of 5 ints [H, W, D, C, T]
        patch_size : list/tuple of 5 ints [pH, pW, pD, pC, pT]
        embed_dim  : output feature dimension per patch (sent to ViT/Swin)
        """
        super().__init__()
        self.patch_size = patch_size

        # total number of patch tokens
        patches_per_dim = [img_size[i] // patch_size[i] for i in range(5)]
        n_tokens = 1
        for n in patches_per_dim:
            n_tokens *= n

        # linear projection (flatten patch → embed_dim)
        flat_patch_dim = (
            patch_size[0] * patch_size[1] * patch_size[2] * patch_size[4] * img_size[3]
        )
        self.proj = nn.Linear(flat_patch_dim, embed_dim)

        # learned positional embedding
        self.pe = nn.Parameter(torch.zeros(1, n_tokens, embed_dim))
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x):
        """
        Args
        ----
        x : Tensor  [B, C, H, W, D, T]

        Returns
        -------
        tokens : Tensor  [B, n_tokens, embed_dim]
        """
        pH, pW, pD, _, pT = self.patch_size

        # Rearrange into patch tokens and flatten per‑patch features
        x = rearrange(
            x,
            "b c (h pH) (w pW) (d pD) (t pT) -> b (h w d t) (pH pW pD pT c)",
            pH=pH,
            pW=pW,
            pD=pD,
            pT=pT,
        )
        x = self.proj(x) + self.pe
        return x


# --------------------------------------------------------------------- #
# 1‑D positional embedding (for ROI/vertex tokens)
# --------------------------------------------------------------------- #
class PositionalEmbedding1D(nn.Module):
    """
    Adds a learned positional vector to each ROI/vertex token.
    Accepts an input shaped [T, B, V] (time first, tokens last).
    """
    def __init__(self, seq_len, embed_dim):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(seq_len, embed_dim))
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x):
        # x : [T, B, V]  → treat V as token axis
        return x + self.pe.unsqueeze(1)
