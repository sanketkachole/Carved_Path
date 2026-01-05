#!/usr/bin/env python3
"""
Create patient-level k-fold splits from meta/patients_all.csv|parquet.

Writes:
out_dir/splits/fold_00/{train,val,test}.csv

Notes:
- Split unit is patient_id (one row per patient).
- Stratify by project_id by default (recommended for pan-cancer).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("make_kfold_splits")


def setup_logging(level: str) -> None:
    numeric = getattr(logging, level.upper(), None)
    if not isinstance(numeric, int):
        raise ValueError(f"Invalid log level: {level}")
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _round_robin_stratified_indices(y: np.ndarray, k: int, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    folds: List[List[int]] = [[] for _ in range(k)]

    classes = np.unique(y)
    for c in classes:
        idx = np.where(y == c)[0].tolist()
        rng.shuffle(idx)
        for j, ii in enumerate(idx):
            folds[j % k].append(ii)

    out: List[np.ndarray] = []
    for f in folds:
        rng.shuffle(f)
        out.append(np.array(f, dtype=np.int64))
    return out


def _plain_kfold_indices(n: int, k: int, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n, dtype=np.int64)
    rng.shuffle(idx)
    return [arr for arr in np.array_split(idx, k)]


def _write_split(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def make_folds(df: pd.DataFrame, k: int, seed: int, stratify_col: Optional[str]) -> List[pd.DataFrame]:
    if k < 2:
        raise ValueError("k must be >= 2")
    n = len(df)
    if n < k:
        raise ValueError(f"Not enough rows ({n}) for k={k}")

    if stratify_col:
        if stratify_col not in df.columns:
            raise ValueError(f"--stratify_col '{stratify_col}' not found in meta columns")
        y = df[stratify_col].astype(str).to_numpy()
        folds_idx = _round_robin_stratified_indices(y=y, k=k, seed=seed)
    else:
        folds_idx = _plain_kfold_indices(n=n, k=k, seed=seed)

    return [df.iloc[idx].reset_index(drop=True) for idx in folds_idx]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create k-fold splits from patients_all meta table")
    p.add_argument("--meta_path", type=str, required=True, help="Path to meta/patients_all.parquet or .csv")
    p.add_argument("--out_dir", type=str, required=True, help="Directory to write splits/ into")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stratify_col", type=str, default="project_id", help="Stratification column (default project_id)")
    p.add_argument("--log_level", type=str, default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    meta_path = Path(args.meta_path)
    out_dir = Path(args.out_dir)

    if meta_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(meta_path)
    else:
        df = pd.read_csv(meta_path)

    required = ["case_id", "patient_id", "h5_path", "report_text"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Meta table missing required columns: {missing}")

    stratify_col = args.stratify_col.strip() or None
    if stratify_col and stratify_col not in df.columns:
        raise ValueError(f"Stratify col '{stratify_col}' not in meta columns")

    # Ensure one row per patient (should already be true)
    df = df.drop_duplicates(subset=["patient_id"]).reset_index(drop=True)

    folds = make_folds(df=df, k=int(args.k), seed=int(args.seed), stratify_col=stratify_col)

    splits_root = out_dir / "splits"
    for i in range(int(args.k)):
        test_df = folds[i]
        val_df = folds[(i + 1) % int(args.k)]
        train_df = pd.concat(
            [folds[j] for j in range(int(args.k)) if j not in (i, (i + 1) % int(args.k))],
            axis=0,
        ).reset_index(drop=True)

        fold_dir = splits_root / f"fold_{i:02d}"
        _write_split(train_df, fold_dir / "train.csv")
        _write_split(val_df, fold_dir / "val.csv")
        _write_split(test_df, fold_dir / "test.csv")

        LOGGER.info("fold_%02d: train=%d val=%d test=%d", i, len(train_df), len(val_df), len(test_df))

    LOGGER.info("Done. Wrote folds to: %s", splits_root)


if __name__ == "__main__":
    main()
