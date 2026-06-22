import argparse
import csv
import json
from pathlib import Path
from collections import OrderedDict, defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.era5_128x256_finetune import era5_128x256_finetune
from utils.builder import ConfigBuilder
from utils.config import load_config


@torch.jit.script
def lat_torch(j: torch.Tensor, num_lat: int) -> torch.Tensor:
    return 90.0 - j * 180.0 / float(num_lat - 1)


@torch.jit.script
def latitude_weighting_factor_torch(j: torch.Tensor, num_lat: int, s: torch.Tensor) -> torch.Tensor:
    return num_lat * torch.cos(torch.pi / 180.0 * lat_torch(j, num_lat)) / s


def latitude_weights(device, num_lat=128):
    lat_t = torch.arange(start=0, end=num_lat, device=device)
    s = torch.sum(torch.cos(torch.pi / 180.0 * lat_torch(lat_t, num_lat)))
    return latitude_weighting_factor_torch(lat_t, num_lat, s).reshape(1, 1, num_lat, 1)


def weighted_rmse_channels(pred, target, weight):
    return torch.sqrt(torch.mean(weight * (pred - target) ** 2, dim=(-1, -2)))


def load_state_dict(path, device):
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "MODEL_STATE" in checkpoint:
        state = checkpoint["MODEL_STATE"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    else:
        state = checkpoint

    cleaned = OrderedDict()
    for key, value in state.items():
        if key.startswith("module."):
            key = key[len("module."):]
        if key.startswith("_orig_mod."):
            key = key[len("_orig_mod."):]
        cleaned[key] = value
    return cleaned


def channel_names(dataset):
    names = [None] * dataset.data_element_num
    for (vname, level), idx in dataset.index_dict1.items():
        names[idx] = f"{vname}" if level == 0 else f"{vname}{level}"
    return names


def channel_groups(dataset):
    groups = [None] * dataset.data_element_num
    levels = [None] * dataset.data_element_num
    for (vname, level), idx in dataset.index_dict1.items():
        groups[idx] = vname
        levels[idx] = int(level)
    return groups, levels


def resolve_ckpt_path(cfg, ckpt_arg):
    if ckpt_arg:
        ckpt_path = Path(ckpt_arg).expanduser().resolve()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        return ckpt_path

    exp_name = cfg.get("experiment", {}).get("name", "")
    if not exp_name:
        raise ValueError("Missing experiment.name in config; cannot auto-locate checkpoint.")

    repo_root = Path(__file__).resolve().parent
    patterns = [
        f"output/{exp_name}/best.pth",
        f"output/{exp_name}/final.pth",
        f"output/{exp_name}/snapshot_latest.pth",
        f"output*/{exp_name}/best.pth",
        f"output*/{exp_name}/final.pth",
        f"output*/{exp_name}/snapshot_latest.pth",
        f"output*/{exp_name}/**/best.pth",
        f"output*/{exp_name}/**/final.pth",
        f"output*/{exp_name}/**/snapshot_latest.pth",
    ]

    candidates = []
    seen = set()
    for pattern in patterns:
        for p in repo_root.glob(pattern):
            p = p.resolve()
            if p.is_file() and p not in seen:
                seen.add(p)
                candidates.append(p)

    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint found for experiment '{exp_name}'. Searched under output*/{exp_name}/ ..."
        )

    priority = {"best.pth": 0, "final.pth": 1, "snapshot_latest.pth": 2}
    candidates.sort(key=lambda p: (priority.get(p.name, 9), -p.stat().st_mtime))
    return candidates[0]


def build_eval_dataset(cfg, year):
    valid_cfg = dict(cfg["dataset"].get("valid", {}))
    valid_cfg["years"] = {"valid": [f"{year}-01-01 00:00:00", f"{year}-12-31 23:00:00"]}
    valid_cfg["file_stride"] = 1
    valid_cfg["train_stride"] = 1
    return era5_128x256_finetune(split="valid", **valid_cfg)


def main():
    parser = argparse.ArgumentParser(description="Evaluate AE reconstruction WRMSE on one year of ERA5 data.")
    parser.add_argument("--cfg", default="configs/ae_kl_hybrid_1024_16_full.yaml")
    parser.add_argument("--ckpt", default="", help="Optional checkpoint path. If empty, auto-resolve from cfg experiment.name")
    parser.add_argument("--year", type=int, default=2017)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=0, help="0 means evaluate the full year.")
    parser.add_argument("--outdir", default="eval_outputs/wrmse_2017")
    parser.add_argument("--save_npy", action="store_true", help="Save per-sample per-channel WRMSE to npy.")
    parser.add_argument(
        "--npy_path",
        default="",
        help="Optional path for sample-wise npy. Defaults to <outdir>/wrmse_<year>_samples.npy",
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.cfg)
    ckpt_path = resolve_ckpt_path(cfg, args.ckpt)
    print(f"Using checkpoint: {ckpt_path}")

    device = torch.device(args.device)

    builder = ConfigBuilder(cfg)
    model = builder.build_model().to(device)
    missing, unexpected = model.load_state_dict(load_state_dict(str(ckpt_path), device), strict=False)
    if missing or unexpected:
        print(f"load_state_dict missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print("missing keys sample:", missing[:10])
        if unexpected:
            print("unexpected keys sample:", unexpected[:10])
    model.eval()

    dataset = build_eval_dataset(cfg, args.year)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    mean, std = dataset.get_meanstd()
    mean = mean.to(device=device, dtype=torch.float32).view(1, -1, 1, 1)
    std = std.to(device=device, dtype=torch.float32).view(1, -1, 1, 1)
    weight = latitude_weights(device=device, num_lat=128)

    names = channel_names(dataset)
    groups, levels = channel_groups(dataset)
    wrmse_sum = torch.zeros(dataset.data_element_num, dtype=torch.float64, device=device)
    sample_count = 0
    wrmse_samples = [] if args.save_npy else None

    with torch.no_grad():
        for batch_idx, x in enumerate(loader):
            if args.max_batches and batch_idx >= args.max_batches:
                break
            x = x.to(device=device, dtype=torch.float32, non_blocking=True)
            recon, _ = model(x)
            target_phys = x * std + mean
            recon_phys = recon * std + mean
            wrmse = weighted_rmse_channels(recon_phys, target_phys, weight).double()
            wrmse_sum += wrmse.sum(dim=0)
            if wrmse_samples is not None:
                wrmse_samples.append(wrmse.detach().cpu().numpy())
            sample_count += x.shape[0]
            if (batch_idx + 1) % 50 == 0:
                print(f"evaluated batches={batch_idx + 1}, samples={sample_count}", flush=True)

    if hasattr(dataset, "close"):
        dataset.close()

    if sample_count == 0:
        raise RuntimeError("No samples were evaluated.")

    wrmse_mean = (wrmse_sum / sample_count).detach().cpu().numpy()

    if wrmse_samples is not None:
        wrmse_samples_np = np.concatenate(wrmse_samples, axis=0).astype(np.float32, copy=False)
        npy_path = Path(args.npy_path) if args.npy_path else (outdir / f"wrmse_{args.year}_samples.npy")
        npy_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(npy_path, wrmse_samples_np)
        print(f"saved npy: {npy_path} shape={wrmse_samples_np.shape}")

    channel_csv = outdir / f"wrmse_{args.year}_channels.csv"
    with channel_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["channel", "name", "variable", "level", "wrmse"])
        for idx, value in enumerate(wrmse_mean):
            writer.writerow([idx, names[idx], groups[idx], levels[idx], float(value)])

    group_values = defaultdict(list)
    for idx, value in enumerate(wrmse_mean):
        group_values[groups[idx]].append(float(value))

    group_csv = outdir / f"wrmse_{args.year}_variables.csv"
    with group_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variable", "num_channels", "mean_wrmse"])
        for name in sorted(group_values.keys()):
            values = group_values[name]
            writer.writerow([name, len(values), sum(values) / len(values)])

    summary = {
        "cfg": str(Path(args.cfg).resolve()),
        "checkpoint": str(ckpt_path),
        "year": args.year,
        "sample_interval_hours": 1,
        "samples": sample_count,
        "batch_size": args.batch_size,
        "max_batches": args.max_batches,
        "channel_csv": str(channel_csv.resolve()),
        "variable_csv": str(group_csv.resolve()),
    }
    if wrmse_samples is not None:
        summary["samples_npy"] = str(
            (Path(args.npy_path) if args.npy_path else (outdir / f"wrmse_{args.year}_samples.npy")).resolve()
        )

    with (outdir / f"wrmse_{args.year}_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print("Variable mean WRMSE:")
    for name in sorted(group_values.keys()):
        values = group_values[name]
        print(f"{name}: {sum(values) / len(values):.6f}")


if __name__ == "__main__":
    main()
