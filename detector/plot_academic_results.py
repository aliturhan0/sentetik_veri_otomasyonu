import os

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = "results"
CSV_NAME = "robustness_metrics.csv"

CONDITION_ORDER = [
    "blur_high",
    "brightness_high",
    "occlusion_high",
]

CONDITION_LABELS = {
    "blur_high": "Blur High",
    "brightness_high": "Brightness High",
    "occlusion_high": "Occlusion High",
}


def academic_bar_plot(
    summary,
    values,
    errors,
    ylabel,
    title,
    output_path,
    ylim=None,
):
    plt.figure(figsize=(8, 5), dpi=300)

    bars = plt.bar(
        summary.index,
        values,
        yerr=errors,
        capsize=5,
        edgecolor="black",
        linewidth=1,
    )

    plt.ylabel(ylabel, fontsize=12)
    plt.xlabel("Synthetic Corruption Type", fontsize=12)
    plt.title(title, fontsize=13, pad=12)

    if ylim:
        plt.ylim(ylim)

    plt.grid(axis="y", linestyle="--", alpha=0.4)

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")
    return output_path


def build_summary(df):
    df = df.copy()
    df["condition_label"] = df["condition"].map(CONDITION_LABELS)

    return (
        df.groupby("condition_label")
        .agg(
            prediction_iou_mean=("prediction_iou", "mean"),
            prediction_iou_std=("prediction_iou", "std"),
            pixel_agreement_mean=("pixel_agreement", "mean"),
            pixel_agreement_std=("pixel_agreement", "std"),
            robustness_drop_mean=("robustness_drop", "mean"),
            robustness_drop_std=("robustness_drop", "std"),
            distribution_shift_mean=("distribution_shift", "mean"),
            distribution_shift_std=("distribution_shift", "std"),
        )
        .reindex([CONDITION_LABELS[c] for c in CONDITION_ORDER])
    )


def run_academic_plots(results_dir=RESULTS_DIR, csv_name=CSV_NAME):
    csv_path = os.path.join(results_dir, csv_name)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metrik CSV bulunamadı: {csv_path}")

    df = pd.read_csv(csv_path)
    summary = build_summary(df)

    summary_path = os.path.join(results_dir, "academic_summary_statistics.csv")
    summary.to_csv(summary_path)
    print(f"Saved: {summary_path}")

    outputs = [
        academic_bar_plot(
            summary=summary,
            values=summary["prediction_iou_mean"],
            errors=summary["prediction_iou_std"],
            ylabel="Prediction Consistency IoU",
            title="Semantic Segmentation Consistency under Synthetic Corruptions",
            output_path=os.path.join(results_dir, "academic_prediction_iou.png"),
            ylim=(0, 1),
        ),
        academic_bar_plot(
            summary=summary,
            values=summary["pixel_agreement_mean"],
            errors=summary["pixel_agreement_std"],
            ylabel="Pixel Agreement",
            title="Pixel-Level Agreement between Clean and Corrupted Predictions",
            output_path=os.path.join(results_dir, "academic_pixel_agreement.png"),
            ylim=(0, 1),
        ),
        academic_bar_plot(
            summary=summary,
            values=summary["robustness_drop_mean"],
            errors=summary["robustness_drop_std"],
            ylabel="Robustness Drop",
            title="Segmentation Robustness Degradation by Corruption Type",
            output_path=os.path.join(results_dir, "academic_robustness_drop.png"),
            ylim=(0, 1),
        ),
        academic_bar_plot(
            summary=summary,
            values=summary["distribution_shift_mean"],
            errors=summary["distribution_shift_std"],
            ylabel="Class Distribution Shift",
            title="Predicted Class Distribution Shift under Synthetic Corruptions",
            output_path=os.path.join(results_dir, "academic_distribution_shift.png"),
        ),
    ]

    return {
        "summary_csv": summary_path,
        "plots": outputs,
    }


def main():
    run_academic_plots()


if __name__ == "__main__":
    main()
