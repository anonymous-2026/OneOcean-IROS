#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 0
    hidden: int = 64
    hidden2: int = 64
    epochs: int = 12
    batch_size: int = 4096
    lr: float = 1e-3
    weight_decay: float = 0.0
    val_frac: float = 0.1


def main() -> int:
    ap = argparse.ArgumentParser(description="Train a tiny MLP (BC) and export weights to a NumPy .npz for headless inference.")
    ap.add_argument("--dataset-npz", required=True)
    ap.add_argument("--dataset-meta", required=True)
    ap.add_argument("--out-dir", required=True, help="Output directory for logs + weights")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = TrainConfig(seed=int(args.seed))
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = np.load(Path(args.dataset_npz).expanduser().resolve())
    x = np.asarray(ds["x"], dtype=np.float32)
    y = np.asarray(ds["y"], dtype=np.float32)
    meta = json.loads(Path(args.dataset_meta).expanduser().read_text(encoding="utf-8"))
    if x.shape[0] == 0:
        raise SystemExit("Empty dataset.")

    # Torch import (training env only).
    import torch
    import torch.nn as nn

    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))

    idx = np.arange(x.shape[0])
    rng = np.random.default_rng(int(cfg.seed))
    rng.shuffle(idx)
    n_val = int(round(float(cfg.val_frac) * float(x.shape[0])))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    x_tr, y_tr = x[tr_idx], y[tr_idx]
    x_va, y_va = x[val_idx], y[val_idx]

    # Standardize.
    x_mean = x_tr.mean(axis=0, keepdims=True)
    x_std = x_tr.std(axis=0, keepdims=True) + 1e-6
    y_mean = y_tr.mean(axis=0, keepdims=True)
    y_std = y_tr.std(axis=0, keepdims=True) + 1e-6
    x_tr_n = (x_tr - x_mean) / x_std
    y_tr_n = (y_tr - y_mean) / y_std
    x_va_n = (x_va - x_mean) / x_std
    y_va_n = (y_va - y_mean) / y_std

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(
        nn.Linear(x.shape[1], int(cfg.hidden)),
        nn.ReLU(),
        nn.Linear(int(cfg.hidden), int(cfg.hidden2)),
        nn.ReLU(),
        nn.Linear(int(cfg.hidden2), 3),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    loss_fn = nn.MSELoss()

    def batches(n: int, bs: int) -> list[tuple[int, int]]:
        out = []
        for i in range(0, n, bs):
            out.append((i, min(n, i + bs)))
        return out

    xtr_t = torch.from_numpy(x_tr_n).to(device)
    ytr_t = torch.from_numpy(y_tr_n).to(device)
    xva_t = torch.from_numpy(x_va_n).to(device)
    yva_t = torch.from_numpy(y_va_n).to(device)

    log = {"train": [], "val": []}
    for ep in range(int(cfg.epochs)):
        model.train()
        perm = torch.randperm(xtr_t.shape[0], device=device)
        ep_loss = 0.0
        for i0, i1 in batches(int(xtr_t.shape[0]), int(cfg.batch_size)):
            ii = perm[i0:i1]
            xb = xtr_t[ii]
            yb = ytr_t[ii]
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            ep_loss += float(loss.detach().cpu().item()) * float(i1 - i0)
        ep_loss /= float(xtr_t.shape[0])

        model.eval()
        with torch.no_grad():
            va = float(loss_fn(model(xva_t), yva_t).detach().cpu().item())
        log["train"].append(ep_loss)
        log["val"].append(va)

    # Export weights for NumPy-only inference.
    # Layer order: 0 Linear, 2 Linear, 4 Linear
    w0 = model[0].weight.detach().cpu().numpy().astype(np.float32)
    b0 = model[0].bias.detach().cpu().numpy().astype(np.float32)
    w1 = model[2].weight.detach().cpu().numpy().astype(np.float32)
    b1 = model[2].bias.detach().cpu().numpy().astype(np.float32)
    w2 = model[4].weight.detach().cpu().numpy().astype(np.float32)
    b2 = model[4].bias.detach().cpu().numpy().astype(np.float32)

    weights_npz = out_dir / "bc_mlp_v1_weights.npz"
    np.savez_compressed(
        weights_npz,
        w0=w0,
        b0=b0,
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        x_mean=x_mean.astype(np.float32),
        x_std=x_std.astype(np.float32),
        y_mean=y_mean.astype(np.float32),
        y_std=y_std.astype(np.float32),
        task_vocab=np.array(meta.get("task_vocab", []), dtype=object),
    )

    report = {
        "time_local": datetime.now().isoformat(timespec="seconds"),
        "cfg": asdict(cfg),
        "dataset": {
            "npz": str(Path(args.dataset_npz)),
            "meta": str(Path(args.dataset_meta)),
            "n_samples": int(x.shape[0]),
            "x_dim": int(x.shape[1]),
        },
        "final": {"train_mse": float(log["train"][-1]), "val_mse": float(log["val"][-1])},
        "loss_curve": log,
        "weights_npz": str(weights_npz),
    }
    (out_dir / "train_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["final"], indent=2))
    print(f"Wrote: {weights_npz}")
    print(f"Wrote: {out_dir / 'train_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

