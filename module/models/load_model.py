from .swin4d_transformer_ver7 import SwinTransformer4D as SwinTransformer4D_ver7
from .swin_transformer        import SwinTransformer  as SwinTransformer4D_base
from timm import create_model

def load_model(model_name, hparams):
    """
    Returns backbone ready for R2T‑Net.
    """
    if model_name == "swin4d_ver7":
        return SwinTransformer4D_ver7(**vars(hparams))
    if model_name == "swin4d_base":
        return SwinTransformer4D_base(**vars(hparams))
    if model_name in ("vit", "transformer2d"):
        return create_model(model_name, pretrained=False, in_chans=hparams.in_chans)
    raise ValueError(f"Unsupported backbone: {model_name}")
