#!/usr/bin/env python3
"""Train a lightweight Aqua-D4RT retention scorer."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from aqua_retention_utils import build_scorer, load_candidate_npz  # noqa: E402
from src.core import seed_everything  # noqa: E402


def _binary_metrics(probs: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = probs >= float(threshold)
    lab = labels.astype(bool)
    tp = int(np.logical_and(pred, lab).sum())
    fp = int(np.logical_and(pred, ~lab).sum())
    fn = int(np.logical_and(~pred, lab).sum())
    tn = int(np.logical_and(~pred, ~lab).sum())
    precision = tp / float(max(1, tp + fp))
    recall = tp / float(max(1, tp + fn))
    f1 = 2.0 * precision * recall / float(max(1e-12, precision + recall))
    accuracy = (tp + tn) / float(max(1, tp + fp + fn + tn))
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "pred_positive_rate": float(pred.mean()) if pred.size else 0.0,
    }


def _best_threshold(probs: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    if probs.size == 0:
        return _binary_metrics(probs, labels, 0.5)
    best = _binary_metrics(probs, labels, 0.5)
    for threshold in np.linspace(0.01, 0.99, 99):
        metrics = _binary_metrics(probs, labels, float(threshold))
        if (metrics["f1"], metrics["precision"], metrics["recall"]) > (
            best["f1"],
            best["precision"],
            best["recall"],
        ):
            best = metrics
    return best


def _predict(
    model: torch.nn.Module,
    features: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    if features.size == 0:
        return np.zeros((0,), dtype=np.float32)
    out: list[np.ndarray] = []
    mean_t = torch.as_tensor(mean, dtype=torch.float32, device=device)
    std_t = torch.as_tensor(std, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        for start in range(0, features.shape[0], int(batch_size)):
            batch = torch.as_tensor(features[start : start + int(batch_size)], dtype=torch.float32, device=device)
            batch = (batch - mean_t) / torch.clamp(std_t, min=1e-6)
            out.append(torch.sigmoid(model(batch).reshape(-1)).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(out, axis=0)


def _split_by_clip(
    features: np.ndarray,
    labels: np.ndarray,
    candidate_meta: np.ndarray,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    clips = np.unique(candidate_meta[:, 0].astype(np.int64)) if candidate_meta.size else np.asarray([0], dtype=np.int64)
    if clips.size <= 1:
        rng = np.random.default_rng(int(seed))
        indices = np.arange(labels.shape[0])
        rng.shuffle(indices)
        n_val = max(1, int(round(float(val_fraction) * indices.size))) if indices.size > 1 else 0
        return indices[n_val:], indices[:n_val]
    rng = np.random.default_rng(int(seed))
    shuffled = clips.copy()
    rng.shuffle(shuffled)
    n_val_clips = max(1, int(round(float(val_fraction) * shuffled.size)))
    val_clips = set(int(v) for v in shuffled[:n_val_clips])
    val_mask = np.asarray([int(v) in val_clips for v in candidate_meta[:, 0].astype(np.int64)], dtype=bool)
    train_idx = np.where(~val_mask)[0]
    val_idx = np.where(val_mask)[0]
    if train_idx.size == 0 or val_idx.size == 0:
        indices = np.arange(labels.shape[0])
        rng.shuffle(indices)
        n_val = max(1, int(round(float(val_fraction) * indices.size))) if indices.size > 1 else 0
        return indices[n_val:], indices[:n_val]
    return train_idx, val_idx


def _merge_payloads(payloads: list[dict[str, Any]], seed: int, max_candidates_per_npz: int) -> dict[str, Any]:
    if not payloads:
        raise ValueError("Provide at least one --train-npz.")
    rng = np.random.default_rng(int(seed))
    merged_features: list[np.ndarray] = []
    merged_labels: list[np.ndarray] = []
    merged_meta: list[np.ndarray] = []
    feature_names = payloads[0]["feature_names"]
    candidate_meta_names = payloads[0]["candidate_meta_names"]
    clip_offset = 0
    for payload in payloads:
        features = payload["features"]
        labels = payload["labels"]
        candidate_meta = payload["candidate_meta"]
        if payload["feature_names"] != feature_names:
            raise ValueError("All training npz files must share the same feature schema.")
        if payload["candidate_meta_names"] != candidate_meta_names:
            raise ValueError("All training npz files must share the same candidate meta schema.")
        if int(max_candidates_per_npz) > 0 and features.shape[0] > int(max_candidates_per_npz):
            indices = rng.choice(features.shape[0], size=int(max_candidates_per_npz), replace=False)
            indices.sort()
            features = features[indices]
            labels = labels[indices]
            candidate_meta = candidate_meta[indices]
        candidate_meta = candidate_meta.copy()
        if candidate_meta.size:
            candidate_meta[:, 0] = candidate_meta[:, 0] + float(clip_offset)
            clip_offset += int(np.max(candidate_meta[:, 0])) - int(np.min(candidate_meta[:, 0])) + 1
        merged_features.append(features)
        merged_labels.append(labels)
        merged_meta.append(candidate_meta)

    features = np.concatenate(merged_features, axis=0)
    labels = np.concatenate(merged_labels, axis=0)
    candidate_meta = np.concatenate(merged_meta, axis=0)
    return {
        "features": features,
        "labels": labels,
        "candidate_meta": candidate_meta,
        "feature_names": feature_names,
        "candidate_meta_names": candidate_meta_names,
    }


def _write_threshold_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["split", "threshold", "precision", "recall", "f1", "accuracy", "pred_positive_rate", "tp", "fp", "fn", "tn"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-npz", action="append", required=True)
    parser.add_argument("--val-npz", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hidden-dim", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--pos-weight", type=float, default=0.0)
    parser.add_argument("--max-candidates-per-npz", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seed_everything(int(args.seed))
    if str(args.device) == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(str(args.device))

    train_payloads = [load_candidate_npz(path) for path in args.train_npz]
    merged_payload = _merge_payloads(
        train_payloads,
        seed=int(args.seed),
        max_candidates_per_npz=int(args.max_candidates_per_npz),
    )
    features = merged_payload["features"]
    labels = merged_payload["labels"]
    candidate_meta = merged_payload["candidate_meta"]
    feature_names = merged_payload["feature_names"]
    if features.shape[0] == 0:
        raise ValueError("No training candidates found.")

    if args.val_npz:
        val_payload = load_candidate_npz(args.val_npz)
        train_x = features
        train_y = labels
        val_x = val_payload["features"]
        val_y = val_payload["labels"]
        split_meta = {"mode": "external_val_npz", "val_npz": str(Path(args.val_npz).resolve())}
    else:
        train_idx, val_idx = _split_by_clip(
            features,
            labels,
            candidate_meta,
            val_fraction=float(args.val_fraction),
            seed=int(args.seed),
        )
        train_x = features[train_idx]
        train_y = labels[train_idx]
        val_x = features[val_idx]
        val_y = labels[val_idx]
        split_meta = {
            "mode": "clip_split" if len(np.unique(candidate_meta[:, 0])) > 1 else "random_split",
            "val_fraction": float(args.val_fraction),
            "num_train": int(train_idx.size),
            "num_val": int(val_idx.size),
            "num_train_npz": len(args.train_npz),
            "max_candidates_per_npz": int(args.max_candidates_per_npz),
        }

    mean = train_x.mean(axis=0).astype(np.float32)
    std = np.maximum(train_x.std(axis=0).astype(np.float32), 1e-6)
    model = build_scorer(input_dim=train_x.shape[1], hidden_dim=int(args.hidden_dim), dropout=float(args.dropout)).to(device)
    if float(args.pos_weight) > 0.0:
        pos_weight = torch.tensor([float(args.pos_weight)], dtype=torch.float32, device=device)
    else:
        num_pos = float(max(1, int(train_y.sum())))
        num_neg = float(max(1, int(train_y.shape[0] - train_y.sum())))
        pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32, device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    train_x_t = torch.as_tensor((train_x - mean) / std, dtype=torch.float32)
    train_y_t = torch.as_tensor(train_y.astype(np.float32), dtype=torch.float32)
    rng = np.random.default_rng(int(args.seed))
    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_val_f1 = -1.0
    best_epoch = -1

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        order = np.arange(train_x_t.shape[0])
        rng.shuffle(order)
        losses: list[float] = []
        for start in range(0, order.size, int(args.batch_size)):
            batch_idx = order[start : start + int(args.batch_size)]
            xb = train_x_t[batch_idx].to(device)
            yb = train_y_t[batch_idx].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb).reshape(-1)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        train_probs = _predict(model, train_x, mean, std, device=device, batch_size=int(args.batch_size))
        val_probs = _predict(model, val_x, mean, std, device=device, batch_size=int(args.batch_size))
        train_best = _best_threshold(train_probs, train_y)
        val_best = _best_threshold(val_probs, val_y)
        row = {
            "epoch": int(epoch),
            "loss": float(np.mean(losses)) if losses else 0.0,
            "train_f1": train_best["f1"],
            "train_precision": train_best["precision"],
            "train_recall": train_best["recall"],
            "val_f1": val_best["f1"],
            "val_precision": val_best["precision"],
            "val_recall": val_best["recall"],
            "val_threshold": val_best["threshold"],
        }
        history.append(row)
        if val_best["f1"] > best_val_f1:
            best_val_f1 = float(val_best["f1"])
            best_epoch = int(epoch)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0 or epoch == int(args.epochs):
            print(
                f"epoch {epoch:03d}: loss={row['loss']:.4f} "
                f"train_f1={row['train_f1']:.3f} val_f1={row['val_f1']:.3f}@{row['val_threshold']:.2f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    train_probs = _predict(model, train_x, mean, std, device=device, batch_size=int(args.batch_size))
    val_probs = _predict(model, val_x, mean, std, device=device, batch_size=int(args.batch_size))
    train_best = _best_threshold(train_probs, train_y)
    val_best = _best_threshold(val_probs, val_y)
    selected_threshold = float(val_best["threshold"])
    threshold_rows: list[dict[str, Any]] = []
    for split, probs, lab in (("train", train_probs, train_y), ("val", val_probs, val_y)):
        for threshold in np.linspace(0.05, 0.95, 19):
            row = _binary_metrics(probs, lab, float(threshold))
            row["split"] = split
            threshold_rows.append(row)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "feature_names": feature_names,
        "feature_mean": mean,
        "feature_std": std,
        "hidden_dim": int(args.hidden_dim),
        "dropout": float(args.dropout),
        "selected_threshold": selected_threshold,
        "metadata": {
            "train_npz": [str(Path(path).resolve()) for path in args.train_npz],
            "num_train": int(train_x.shape[0]),
            "num_val": int(val_x.shape[0]),
            "train_positive_rate": float(train_y.mean()) if train_y.size else 0.0,
            "val_positive_rate": float(val_y.mean()) if val_y.size else 0.0,
            "split": split_meta,
            "pos_weight": float(pos_weight.detach().cpu().item()),
            "best_epoch": int(best_epoch),
            "best_val_f1": float(best_val_f1),
        },
    }
    torch.save(ckpt, output_dir / "retention_scorer.pt")
    summary = {
        "metadata": ckpt["metadata"],
        "feature_names": feature_names,
        "selected_threshold": selected_threshold,
        "train_best": train_best,
        "val_best": val_best,
        "history": history,
    }
    (output_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (output_dir / "train_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    _write_threshold_csv(output_dir / "threshold_sweep.csv", threshold_rows)

    print("Retention scorer training summary:")
    print(f"- train candidates: {train_x.shape[0]} pos={float(train_y.mean()):.3f}")
    print(f"- val candidates: {val_x.shape[0]} pos={float(val_y.mean()):.3f}")
    print(f"- val best F1: {val_best['f1']:.4f} @ threshold {selected_threshold:.2f}")
    print(f"Saved: {output_dir / 'retention_scorer.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
