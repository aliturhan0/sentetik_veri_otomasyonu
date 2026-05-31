"""
Baseline RCGAN ile V2 RCGAN modelini aynı seed verisi üstünde karşılaştırır.

Kullanım:
    python3 akilli_veri_arttirimi/scripts/08_compare_rcgan_models.py \
      --seed akilli_veri_arttirimi/waymo_seed_MASSIVE.csv \
      --baseline akilli_veri_arttirimi/outputs/waymo_rcgan_GODMODE_A100_STABLE.pth \
      --candidate akilli_veri_arttirimi/outputs/waymo_rcgan_GODMODE_V2_PHYSICS_AWARE.pth \
      --samples 1000

Çıktı:
    akilli_veri_arttirimi/outputs/rcgan_model_comparison.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from akilli_veri_arttirimi.backend import server  # noqa: E402


def load_generator(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model yok: {model_path}")
    if server.is_git_lfs_pointer(str(model_path)):
        raise RuntimeError(f"Model Git LFS pointer görünüyor: {model_path}")

    generator = server.RCGAN_Generator()
    checkpoint = server.safe_torch_load(str(model_path), map_location="cpu")
    state_dict = checkpoint["G"] if isinstance(checkpoint, dict) and "G" in checkpoint else checkpoint
    generator.load_state_dict(state_dict)
    generator.eval()
    return generator


def read_seed(seed_path: Path, sample_rows: int) -> pd.DataFrame:
    total_rows = server.count_csv_rows(str(seed_path))
    if total_rows <= 0:
        return pd.read_csv(seed_path).head(sample_rows)
    return server.read_seed_dataframe(str(seed_path), total_rows, sample_size=sample_rows)


def compact_physical_report(report: dict) -> dict:
    return {
        "applicable": report.get("applicable"),
        "score": report.get("score"),
        "negative_speed_ratio": report.get("negative_speed_ratio"),
        "high_speed_ratio": report.get("high_speed_ratio"),
        "high_accel_ratio": report.get("high_accel_ratio"),
        "jump_ratio": report.get("jump_ratio"),
        "coherence_penalty": report.get("coherence_penalty"),
        "detail": report.get("detail"),
    }


def evaluate_model(name: str, model_path: Path, seed_df: pd.DataFrame, samples: int) -> dict:
    server.rcgan = load_generator(model_path)
    df_gen = server.generate_waymo(seed_df, samples)
    if df_gen.empty:
        raise RuntimeError(f"{name} modeli üretim yapamadı.")

    numeric_cols = server.waymo_feature_columns()
    fidelity = server._compute_fidelity_report(seed_df, df_gen, numeric_cols, "label" if "label" in seed_df.columns else None)
    physical = server._physical_consistency_report(df_gen)
    label_dist = df_gen["label"].value_counts().to_dict() if "label" in df_gen.columns else {}
    post = df_gen.attrs.get("rcgan_postprocess", {})

    return {
        "name": name,
        "model_path": str(model_path),
        "generated_rows": int(len(df_gen)),
        "label_distribution": {str(k): int(v) for k, v in label_dist.items()},
        "fidelity": {
            "cosine_similarity": fidelity.get("cosine_similarity"),
            "column_correlation": fidelity.get("column_correlation"),
            "avg_columns_checked": len(fidelity.get("column_details", [])),
        },
        "physical": compact_physical_report(physical),
        "postprocess": {
            "severity": post.get("severity"),
            "diversity_removed": post.get("diversity_filter", {}).get("removed"),
            "output_rows_after_postprocess": post.get("output_rows_after_postprocess"),
        },
    }


def choose_winner(baseline: dict, candidate: dict) -> dict:
    b_phys = float(baseline.get("physical", {}).get("score") or 0)
    c_phys = float(candidate.get("physical", {}).get("score") or 0)
    b_fid = float(baseline.get("fidelity", {}).get("column_correlation") or 0)
    c_fid = float(candidate.get("fidelity", {}).get("column_correlation") or 0)

    baseline_score = (b_phys * 0.60) + (max(b_fid, 0) * 100 * 0.40)
    candidate_score = (c_phys * 0.60) + (max(c_fid, 0) * 100 * 0.40)
    margin = candidate_score - baseline_score

    if margin >= 2.0:
        decision = "candidate"
        detail = "V2 model fizik/fidelity bileşik skorunda baseline modelden anlamlı biçimde iyi."
    elif margin <= -2.0:
        decision = "baseline"
        detail = "Baseline model bileşik skorda daha güvenli; V2 şimdilik devreye alınmamalı."
    else:
        decision = "baseline"
        detail = "Skorlar yakın; güvenli rollback politikası gereği baseline korunmalı."

    return {
        "decision": decision,
        "baseline_score": round(baseline_score, 3),
        "candidate_score": round(candidate_score, 3),
        "margin": round(margin, 3),
        "detail": detail,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=str(ROOT / "akilli_veri_arttirimi" / "waymo_seed_MASSIVE.csv"))
    parser.add_argument("--baseline", default=str(ROOT / "akilli_veri_arttirimi" / "outputs" / "waymo_rcgan_GODMODE_A100_STABLE.pth"))
    parser.add_argument("--candidate", default=str(ROOT / "akilli_veri_arttirimi" / "outputs" / "waymo_rcgan_GODMODE_V2_PHYSICS_AWARE.pth"))
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed-sample-rows", type=int, default=5000)
    parser.add_argument("--out", default=str(ROOT / "akilli_veri_arttirimi" / "outputs" / "rcgan_model_comparison.json"))
    args = parser.parse_args()

    seed_path = Path(args.seed)
    baseline_path = Path(args.baseline)
    candidate_path = Path(args.candidate)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seed_df = read_seed(seed_path, args.seed_sample_rows)
    if "label" not in seed_df.columns:
        seed_df = seed_df.copy()
        seed_df["label"] = "normal"
    if not server.is_waymo_frame(seed_df):
        converted = server.try_convert_to_waymo(seed_df, "label")
        if converted is None or converted.empty:
            raise RuntimeError("Seed veri Waymo formatına çevrilemedi.")
        seed_df = converted

    baseline = evaluate_model("baseline", baseline_path, seed_df, args.samples)
    candidate = evaluate_model("candidate", candidate_path, seed_df, args.samples)
    decision = choose_winner(baseline, candidate)

    result = {
        "seed_rows": int(len(seed_df)),
        "samples": int(args.samples),
        "baseline": baseline,
        "candidate": candidate,
        "decision": decision,
        "activation_hint": (
            f"SENTETIK_RCGAN_MODEL_PATH={candidate_path}"
            if decision["decision"] == "candidate"
            else "Baseline ile devam et; V2 modeli devreye alma."
        ),
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps(result["decision"], ensure_ascii=False, indent=2))
    print(f"Karşılaştırma raporu: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
