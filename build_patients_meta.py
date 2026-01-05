#!/usr/bin/env python3
"""
Build a pan-cancer patient-level meta table by merging:
- WSI feature table (h5_path, file_uuid, patient_id)
- Labels table (labels.csv)
- TCGA_Reports.csv (patient_filename, text)

Outputs:
- meta/patients_all.parquet (preferred) and meta/patients_all.csv (fallback)

Key behavior:
- Normalizes all IDs to TCGA-XX-YYYY (3-chunk) as patient_id_3
- Picks ONE slide per patient (prefer .DX1., else first sorted path)
- Drops rows missing report text / h5_path
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import List, Optional

import pandas as pd

LOGGER = logging.getLogger("build_patients_meta")

TCGA3_RE = re.compile(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", re.IGNORECASE)


def setup_logging(level: str) -> None:
    numeric = getattr(logging, level.upper(), None)
    if not isinstance(numeric, int):
        raise ValueError(f"Invalid log level: {level}")
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def to_tcga3(x: object) -> Optional[str]:
    if x is None:
        return None
    m = TCGA3_RE.search(str(x))
    return m.group(1).upper() if m else None


def pick_one_slide_per_patient(wsi_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prefer DX1 slide if present, else keep first (sorted) slide.
    """
    df = wsi_df.copy()
    df["is_dx1"] = df["h5_path"].astype(str).str.contains(r"\.DX1\.", case=False, regex=True).astype(int)
    df["h5_path_str"] = df["h5_path"].astype(str)

    df = df.sort_values(["patient_id_3", "is_dx1", "h5_path_str"], ascending=[True, False, True])
    df = df.drop_duplicates(subset=["patient_id_3"], keep="first").reset_index(drop=True)

    df = df.drop(columns=["is_dx1", "h5_path_str"])
    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build patient-level meta table for MICCAI_carved_path")
    p.add_argument("--wsi_csv", type=str, required=True, help="Path to wsi_virchow_features_filename_ID_table.csv")
    p.add_argument("--labels_csv", type=str, required=True, help="Path to labels.csv")
    p.add_argument("--reports_csv", type=str, required=True, help="Path to TCGA_Reports.csv (patient_filename,text)")
    p.add_argument("--out_dir", type=str, default="meta", help="Output directory for meta tables")
    p.add_argument(
        "--keep_label_cols",
        type=str,
        default="project_id,primary_site,disease_type,gender,race,ethnicity,vital_status,primary_diagnosis,morphology,"
                "tissue_or_organ_of_origin,age_at_diagnosis,ajcc_pathologic_stage,ajcc_pathologic_t,ajcc_pathologic_n,"
                "ajcc_pathologic_m,tumor_grade,smoking_status,alcohol_history,survival_event,survival_time_days",
        help="Comma-separated label columns to keep from labels.csv (plus patient_id always).",
    )
    p.add_argument("--log_level", type=str, default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    wsi_csv = Path(args.wsi_csv)
    labels_csv = Path(args.labels_csv)
    reports_csv = Path(args.reports_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    keep_label_cols: List[str] = [c.strip() for c in args.keep_label_cols.split(",") if c.strip()]
    if "patient_id" in keep_label_cols:
        keep_label_cols.remove("patient_id")

    LOGGER.info("Loading WSI table: %s", wsi_csv)
    wsi = pd.read_csv(wsi_csv)
    for col in ["h5_path", "file_uuid", "patient_id"]:
        if col not in wsi.columns:
            raise ValueError(f"WSI CSV missing required column: {col}")

    LOGGER.info("Loading labels table: %s", labels_csv)
    labels = pd.read_csv(labels_csv)
    if "patient_id" not in labels.columns:
        raise ValueError("labels.csv missing required column: patient_id")

    missing_keep = [c for c in keep_label_cols if c not in labels.columns]
    if missing_keep:
        LOGGER.warning("Some requested keep_label_cols not found in labels.csv and will be skipped: %s", missing_keep)
        keep_label_cols = [c for c in keep_label_cols if c in labels.columns]

    labels = labels[["patient_id"] + keep_label_cols].copy()

    LOGGER.info("Loading reports table: %s", reports_csv)
    reports = pd.read_csv(reports_csv)
    if "patient_filename" not in reports.columns or "text" not in reports.columns:
        raise ValueError("TCGA_Reports.csv must have columns: patient_filename,text")

    # Normalize IDs
    wsi["patient_id_3"] = wsi["patient_id"].map(to_tcga3)
    labels["patient_id_3"] = labels["patient_id"].map(to_tcga3)
    reports["patient_id_3"] = reports["patient_filename"].map(to_tcga3)

    # Drop invalid IDs early
    wsi = wsi.dropna(subset=["patient_id_3", "h5_path"]).copy()
    labels = labels.dropna(subset=["patient_id_3"]).copy()
    reports = reports.dropna(subset=["patient_id_3", "text"]).copy()

    # One slide per patient
    wsi_one = pick_one_slide_per_patient(wsi)

    # One report per patient (if duplicates, keep first)
    reports_one = (
        reports.sort_values(["patient_id_3"])
        .drop_duplicates(subset=["patient_id_3"], keep="first")
        .reset_index(drop=True)
    )[["patient_id_3", "text"]].rename(columns={"text": "report_text"})

    # Merge: inner join ensures all modalities/labels exist
    LOGGER.info("Merging tables on patient_id_3...")
    merged = wsi_one.merge(labels.drop(columns=["patient_id"]), how="inner", on="patient_id_3")
    merged = merged.merge(reports_one, how="inner", on="patient_id_3")

    merged["patient_id"] = merged["patient_id_3"]
    merged["case_id"] = merged["patient_id_3"]

    # Drop empties
    merged = merged.dropna(subset=["h5_path", "report_text"]).reset_index(drop=True)

    # Minimal + labels + report_text
    # Keep any extra columns you want; this is safe for downstream instance-building.
    LOGGER.info("Final merged patients: %d", len(merged))
    LOGGER.info("Unique patients: %d", merged["patient_id"].nunique())

    out_parquet = out_dir / "patients_all.parquet"
    out_csv = out_dir / "patients_all.csv"

    try:
        merged.to_parquet(out_parquet, index=False)
        LOGGER.info("Wrote: %s", out_parquet)
    except Exception as e:
        LOGGER.warning("Parquet failed (%s). Writing CSV instead.", e)
        merged.to_csv(out_csv, index=False)
        LOGGER.info("Wrote: %s", out_csv)

    # Always write CSV too for easy inspection
    merged.to_csv(out_csv, index=False)
    LOGGER.info("Also wrote: %s", out_csv)


if __name__ == "__main__":
    main()
