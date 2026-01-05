#!/usr/bin/env python3
"""
Build Bundle-1 training instances (QA + JSON extraction + masked QA + synthetic verification).

Input:
- meta/patients_all.parquet (or CSV) with columns: case_id, patient_id, h5_path, report_text, plus label columns.

Output:
- meta/instances_bundle1.parquet (preferred) and CSV fallback

Instance rows:
- case_id, patient_id, h5_path
- task_type: qa | extract_json | masked_qa | verify
- prompt: text prompt to feed the LLM (you'll append report_text separately or inline)
- report_text: (optionally included here; can also store separately in meta)
- target_text: answer string / JSON string / verification label
- meta fields: question_id, field_name, mask_strategy, verify_statement, verify_is_true

Masking strategy:
- For masked_qa: we try to remove the exact answer string from the report (case-insensitive).
- For diagnosis-like fields, also attempt to remove the DIAGNOSIS section with a heuristic regex.

Verification:
- Build true statement from field; build false by sampling another patient's value from same project_id
  (fallback: global pool) to create hard-ish negatives.

This is intentionally simple and deterministic so you can iterate.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("build_instances_bundle1")


def setup_logging(level: str) -> None:
    numeric = getattr(logging, level.upper(), None)
    if not isinstance(numeric, int):
        raise ValueError(f"Invalid log level: {level}")
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


DIAG_SECTION_RE = re.compile(
    r"(DIAGNOSIS\s*:.*?)(GROSS\s+DESCRIPTION\s*:|INTRAOPERATIVE\s+CONSULTATION\s*:|$)",
    flags=re.IGNORECASE | re.DOTALL,
)


def normalize_answer(x: object) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    # Drop obvious "nan"
    if s.lower() in {"nan", "none", "null", "not reported"}:
        return None
    return s


def safe_json_dumps(d: Dict[str, object]) -> str:
    return json.dumps(d, ensure_ascii=False, sort_keys=True)


def remove_answer_from_report(report: str, answer: str) -> Tuple[str, str]:
    """
    Returns (masked_report, strategy_name).
    """
    if not report:
        return report, "none"

    rep = report
    # Attempt 1: remove DIAGNOSIS section (helps for diagnosis fields)
    m = DIAG_SECTION_RE.search(rep)
    if m:
        rep2 = rep[: m.start(1)] + "[DIAGNOSIS_SECTION_MASKED]" + rep[m.end(1) :]
        if rep2 != rep:
            rep = rep2
            # still also remove answer string if present
            rep = re.sub(re.escape(answer), "[MASK]", rep, flags=re.IGNORECASE)
            return rep, "diagnosis_section+answer_sub"
    # Attempt 2: remove exact answer occurrences
    rep2 = re.sub(re.escape(answer), "[MASK]", rep, flags=re.IGNORECASE)
    if rep2 != rep:
        return rep2, "answer_sub"
    # Attempt 3: nothing changed
    return rep, "none"


@dataclass(frozen=True)
class QAField:
    field: str
    question: str


DEFAULT_QA_FIELDS: List[QAField] = [
    QAField("project_id", "What is the cancer project ID?"),
    QAField("primary_site", "What is the primary site?"),
    QAField("primary_diagnosis", "What is the primary diagnosis?"),
    QAField("morphology", "What is the morphology code?"),
    QAField("ajcc_pathologic_stage", "What is the AJCC pathologic stage?"),
    QAField("tumor_grade", "What is the tumor grade?"),
    QAField("vital_status", "What is the vital status?"),
    QAField("survival_event", "What is the survival event indicator?"),
    QAField("survival_time_days", "What is the survival time in days?"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Bundle-1 multimodal instances table")
    p.add_argument("--meta_path", type=str, required=True, help="Path to meta/patients_all.parquet or .csv")
    p.add_argument("--out_dir", type=str, default="meta", help="Output directory")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_patients", type=int, default=0, help="If >0, cap number of patients for quick testing")
    p.add_argument("--include_report_text", action="store_true", help="Store report_text in instances file too")
    p.add_argument("--qa_fields", type=str, default="", help="Optional JSON file to override QA fields/questions")
    p.add_argument(
        "--json_fields",
        type=str,
        default="project_id,primary_site,primary_diagnosis,morphology,ajcc_pathologic_stage,tumor_grade,vital_status",
        help="Comma-separated fields to include in synoptic JSON extraction target",
    )
    p.add_argument("--n_verify_per_patient", type=int, default=2, help="How many verify instances per patient")
    p.add_argument("--log_level", type=str, default="INFO")
    return p.parse_args()


def load_meta(meta_path: Path) -> pd.DataFrame:
    if meta_path.suffix.lower() == ".parquet":
        return pd.read_parquet(meta_path)
    return pd.read_csv(meta_path)


def load_qa_fields(path: str) -> List[QAField]:
    """
    JSON format:
    [
      {"field": "project_id", "question": "What is ...?"},
      ...
    ]
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out: List[QAField] = []
    for item in raw:
        out.append(QAField(field=str(item["field"]), question=str(item["question"])))
    return out


def build_negative_pools(df: pd.DataFrame, field: str) -> Dict[str, List[str]]:
    """
    Returns mapping: project_id -> list of possible values for `field` within that project.
    """
    pools: Dict[str, List[str]] = {}
    if "project_id" not in df.columns:
        return pools
    sub = df[["project_id", field]].copy()
    sub["project_id"] = sub["project_id"].astype(str)
    sub[field] = sub[field].apply(normalize_answer)

    sub = sub.dropna(subset=[field])
    for pid, grp in sub.groupby("project_id"):
        vals = sorted(set(grp[field].astype(str).tolist()))
        if vals:
            pools[pid] = vals
    return pools


def sample_negative(
    rng: np.random.Generator,
    true_val: str,
    pool_same_project: Sequence[str],
    pool_global: Sequence[str],
) -> Optional[str]:
    candidates = [v for v in pool_same_project if v != true_val]
    if not candidates:
        candidates = [v for v in pool_global if v != true_val]
    if not candidates:
        return None
    return str(candidates[int(rng.integers(0, len(candidates)))])


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    meta_path = Path(args.meta_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(args.seed))

    df = load_meta(meta_path)

    required = ["case_id", "patient_id", "h5_path", "report_text"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Meta table missing required columns: {missing}")

    if int(args.max_patients) > 0:
        df = df.head(int(args.max_patients)).copy()

    # QA fields
    qa_fields = DEFAULT_QA_FIELDS
    if args.qa_fields.strip():
        qa_fields = load_qa_fields(args.qa_fields.strip())

    # JSON extraction fields
    json_fields = [c.strip() for c in args.json_fields.split(",") if c.strip()]
    for c in json_fields:
        if c not in df.columns:
            LOGGER.warning("json_fields includes '%s' which is not in meta; it will be null in target JSON.", c)

    # Precompute verification pools for a couple key fields (diagnosis + stage by default if present)
    verify_fields = []
    for f in ["primary_diagnosis", "ajcc_pathologic_stage", "tumor_grade"]:
        if f in df.columns:
            verify_fields.append(f)

    global_pools: Dict[str, List[str]] = {}
    per_project_pools: Dict[str, Dict[str, List[str]]] = {}
    for f in verify_fields:
        vals = [normalize_answer(v) for v in df[f].tolist()]
        vals = [v for v in vals if v is not None]
        global_pools[f] = sorted(set(vals))
        per_project_pools[f] = build_negative_pools(df, f)

    instances: List[Dict[str, object]] = []

    def base_prompt(report_text: str, instruction: str) -> str:
        # Keep prompt simple. Your training code will prepend visual tokens.
        return (
            "You are given a pathology whole-slide image (WSI) and its clinical pathology report.\n"
            "Use BOTH sources when needed.\n\n"
            f"REPORT:\n{report_text}\n\n"
            f"TASK:\n{instruction}\n"
            "ANSWER:"
        )

    for idx, row in df.iterrows():
        case_id = str(row["case_id"])
        patient_id = str(row["patient_id"])
        h5_path = str(row["h5_path"])
        report_text = str(row["report_text"])

        # ---- (1) QA instances ----
        for q_i, q in enumerate(qa_fields):
            if q.field not in df.columns:
                continue
            ans = normalize_answer(row.get(q.field))
            if ans is None:
                continue

            prompt = base_prompt(report_text, f"Question: {q.question} Provide a short, direct answer.")
            inst = {
                "case_id": case_id,
                "patient_id": patient_id,
                "h5_path": h5_path,
                "task_type": "qa",
                "question_id": f"{q.field}__{q_i:02d}",
                "field_name": q.field,
                "prompt": prompt,
                "target_text": str(ans),
                "mask_strategy": "",
                "verify_statement": "",
                "verify_is_true": "",
            }
            if args.include_report_text:
                inst["report_text"] = report_text
            instances.append(inst)

            # ---- (2) Masked QA forcing (try to remove shortcut) ----
            masked_report, strategy = remove_answer_from_report(report_text, str(ans))
            masked_prompt = base_prompt(
                masked_report,
                f"Question: {q.question} Provide a short, direct answer. "
                "If the report is missing details, infer from the WSI.",
            )
            inst_m = inst.copy()
            inst_m.update(
                {
                    "task_type": "masked_qa",
                    "prompt": masked_prompt,
                    "mask_strategy": strategy,
                }
            )
            if args.include_report_text:
                inst_m["report_text"] = masked_report
            instances.append(inst_m)

        # ---- (3) JSON synoptic extraction ----
        syn = {f: (normalize_answer(row.get(f)) or "") for f in json_fields}
        json_target = safe_json_dumps(syn)
        prompt_j = base_prompt(
            report_text,
            "Extract the following synoptic fields as a strict JSON object with keys exactly as specified: "
            f"{json_fields}. Output JSON only.",
        )
        inst_j = {
            "case_id": case_id,
            "patient_id": patient_id,
            "h5_path": h5_path,
            "task_type": "extract_json",
            "question_id": "synoptic_json",
            "field_name": "|".join(json_fields),
            "prompt": prompt_j,
            "target_text": json_target,
            "mask_strategy": "",
            "verify_statement": "",
            "verify_is_true": "",
        }
        if args.include_report_text:
            inst_j["report_text"] = report_text
        instances.append(inst_j)

        # ---- (4) Synthetic verification ----
        n_verify = int(args.n_verify_per_patient)
        for _ in range(n_verify):
            if not verify_fields:
                break
            field = str(verify_fields[int(rng.integers(0, len(verify_fields)))])
            true_val = normalize_answer(row.get(field))
            if true_val is None:
                continue

            # true statement
            statement_true = f"{field} is {true_val}."
            prompt_v_true = base_prompt(
                report_text,
                f"Verification: Is the following statement supported by the WSI + report? "
                f"Answer with one token: supported or contradicted.\nStatement: {statement_true}",
            )
            inst_vt = {
                "case_id": case_id,
                "patient_id": patient_id,
                "h5_path": h5_path,
                "task_type": "verify",
                "question_id": f"verify__{field}",
                "field_name": field,
                "prompt": prompt_v_true,
                "target_text": "supported",
                "mask_strategy": "",
                "verify_statement": statement_true,
                "verify_is_true": "1",
            }
            if args.include_report_text:
                inst_vt["report_text"] = report_text
            instances.append(inst_vt)

            # false statement (negative)
            proj = str(row.get("project_id", ""))
            same_pool = per_project_pools.get(field, {}).get(proj, [])
            global_pool = global_pools.get(field, [])
            neg = sample_negative(rng, true_val=str(true_val), pool_same_project=same_pool, pool_global=global_pool)
            if neg is None:
                continue

            statement_false = f"{field} is {neg}."
            prompt_v_false = base_prompt(
                report_text,
                f"Verification: Is the following statement supported by the WSI + report? "
                f"Answer with one token: supported or contradicted.\nStatement: {statement_false}",
            )
            inst_vf = {
                "case_id": case_id,
                "patient_id": patient_id,
                "h5_path": h5_path,
                "task_type": "verify",
                "question_id": f"verify__{field}",
                "field_name": field,
                "prompt": prompt_v_false,
                "target_text": "contradicted",
                "mask_strategy": "",
                "verify_statement": statement_false,
                "verify_is_true": "0",
            }
            if args.include_report_text:
                inst_vf["report_text"] = report_text
            instances.append(inst_vf)

    inst_df = pd.DataFrame(instances)
    LOGGER.info("Built instances: %d", len(inst_df))
    LOGGER.info("Task type counts:\n%s", inst_df["task_type"].value_counts())

    out_parquet = out_dir / "instances_bundle1.parquet"
    out_csv = out_dir / "instances_bundle1.csv"

    try:
        inst_df.to_parquet(out_parquet, index=False)
        LOGGER.info("Wrote: %s", out_parquet)
    except Exception as e:
        LOGGER.warning("Parquet failed (%s). Writing CSV instead.", e)
        inst_df.to_csv(out_csv, index=False)
        LOGGER.info("Wrote: %s", out_csv)

    # Always write CSV too
    inst_df.to_csv(out_csv, index=False)
    LOGGER.info("Also wrote: %s", out_csv)


if __name__ == "__main__":
    main()
