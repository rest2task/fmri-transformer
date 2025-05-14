# inference.py
# ------------------------------------------------------------
# Predict cognitive scores for resting‑state fMRI using a
# checkpointed R2T‑Net.  Example:
#
#   python inference.py \
#     --ckpt  logs/epoch03-valid_loss=0.2100.ckpt \
#     --input_dir  demo_data/rest/ \
#     --output predictions.csv
#
#  Each .pt file in input_dir is assumed to contain either
#   ▸ tensor [C,H,W,D,T]  (volumetric)   or
#   ▸ tensor [V,T]        (ROI series)
# ------------------------------------------------------------
import csv
import glob
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch
from tqdm import tqdm
from module.r2tnet import R2TNet


# ------------------------------------------------------------
def load_model(ckpt_path: str, num_rois: int, device: torch.device) -> R2TNet:
    # Load entire Lightning checkpoint (or plain .pth)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    hyper = ckpt["hyper_parameters"] if "hyper_parameters" in ckpt else {}

    # override ROI size if user passes --num_rois
    if num_rois > 0:
        hyper["num_rois"] = num_rois

    # Build skeleton model
    dummy_data_module = type("D", (), {"train_dataset": type("T", (), {"target_values": [[0.0]]})})()
    model = R2TNet(data_module=dummy_data_module, **hyper)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.eval().to(device)
    return model


# ------------------------------------------------------------
@torch.no_grad()
def predict_dir(model, input_dir: str, device: torch.device):
    results = []
    for p in tqdm(sorted(glob.glob(os.path.join(input_dir, "*.pt"))), desc="inference"):
        x = torch.load(p, map_location="cpu")
        if x.ndim == 5:                # [C,H,W,D,T]  -> add batch
            x = x.unsqueeze(0)         # [1,C,H,W,D,T]
            mod = torch.zeros(1, dtype=torch.long)  # modality=0 (rest)
            _, pred = model(x.to(device), mod.to(device))
        elif x.ndim == 2:              # [V,T]  ROI
            x = x.unsqueeze(0)         # [1,V,T]
            mod = torch.zeros(1, dtype=torch.long)
            _, pred = model(x.to(device), mod.to(device))
        else:
            raise RuntimeError(f"{p} has unexpected shape {tuple(x.shape)}")

        score = pred.cpu().item()
        results.append((os.path.basename(p), score))
    return results


# ------------------------------------------------------------
def main():
    ap = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="Trained .ckpt or .pth file")
    ap.add_argument("--input_dir", required=True,
                    help="Folder with *.pt tensors for each subject")
    ap.add_argument("--output", default="predictions.csv")
    ap.add_argument("--num_rois", type=int, default=0,
                    help="Set >0 if inputs are ROI matrices [V,T]")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    model  = load_model(args.ckpt, args.num_rois, device)

    preds = predict_dir(model, args.input_dir, device)

    with open(args.output, "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["file", "predicted_score"])
        writer.writerows(preds)

    print(f"✅  wrote {len(preds)} predictions to {args.output}")


if __name__ == "__main__":
    main()
