from __future__ import annotations

import ast
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

PYTHON_SOURCES = [
    "main_launcher.py",
    "akilli_veri_arttirimi/main.py",
    "akilli_veri_arttirimi/backend/server.py",
    "rcgan_qt_gui_app_v1/generate.py",
    "rcgan_qt_gui_app_v1/qt_gui_app_updated.py",
    "rcgan_qt_gui_app_v1/pipeline_bridge.py",
    "detector/upscale_generated_images.py",
    "detector/yolo_robustness_evaluation.py",
    "detector/semantic_segmentation_robustness.py",
    "detector/plot_academic_results.py",
]

REQUIRED_ARTIFACTS = [
    "rcgan_qt_gui_app_v1/checkpoint_epoch_29.pt",
    "detector/EDSR_x4.pb",
    "detector/yolov8n.pt",
    "akilli_veri_arttirimi/outputs/waymo_rcgan_GODMODE_A100_STABLE.pth",
    "akilli_veri_arttirimi/waymo_seed_MASSIVE.csv",
]


def is_lfs_pointer(path: Path) -> bool:
    if not path.exists() or path.stat().st_size > 1024:
        return False
    try:
        return path.read_bytes().startswith(b"version https://git-lfs.github.com/spec/v1")
    except OSError:
        return False


def check_python_syntax() -> list[str]:
    errors = []
    for relative in PYTHON_SOURCES:
        path = ROOT / relative
        try:
            source = path.read_text(encoding="utf-8")
            ast.parse(source, filename=str(path))
            compile(source, str(path), "exec")
        except Exception as exc:
            errors.append(f"{relative}: {exc}")
    return errors


def check_required_artifacts() -> list[str]:
    errors = []
    for relative in REQUIRED_ARTIFACTS:
        path = ROOT / relative
        if not path.exists():
            errors.append(f"{relative}: dosya yok")
        elif is_lfs_pointer(path):
            errors.append(f"{relative}: Git LFS pointer, `git lfs pull` gerekiyor")
    return errors


def main() -> int:
    errors = []
    errors.extend(check_python_syntax())
    errors.extend(check_required_artifacts())

    if errors:
        print("Smoke check failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Smoke check passed: syntax and required artifact checks are OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
