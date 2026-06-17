import os
import importlib.util
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

# PySide6/Shiboken installs an import hook for Qt feature detection. On some
# macOS environments that hook can trip over six.moves while pandas imports
# python-dateutil late in the pipeline, so preload this tiny dependency first.
from six.moves import _thread  # noqa: F401

from PIL import Image, ImageOps
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QSize
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QHeaderView,
    QSizePolicy,
    QStackedWidget,
    QStyle,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from generate import (
    FAULT_MAP,
    SEVERITY_MAP,
    collect_images_from_folder,
    ensure_local_file,
    generate_sequence,
    natural_sort_key,
)
from pipeline_bridge import copy_images, prepare_detector_dataset
from pipeline_bridge import prepare_project_clean_dataset


APP_TITLE = "RCGAN Robustness Pipeline"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results"
DEFAULT_CLEAN_DIR = PROJECT_ROOT / "clean"
DEFAULT_OUTPUT_DIR = str(DEFAULT_OUTPUTS_ROOT / "gan_generated")
DEFAULT_DETECTOR_DIR = str(PROJECT_ROOT / "detector")
DEFAULT_CHECKPOINT = str(Path(__file__).resolve().parent / "checkpoint_epoch_29.pt")
PREVIEW_CACHE_DIR = Path(tempfile.gettempdir()) / "sentetik_image_previews"
PREVIEW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = Path(os.getenv("SENTETIK_IMAGE_LOG", Path(__file__).resolve().parent / "image_runtime.log"))
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
TEXT_PREVIEW_SUFFIXES = {".csv", ".txt", ".md", ".log"}
CSV_COLUMN_DESCRIPTIONS = {
    "condition": "Uygulanan bozulma koşulu. Örn: blur_high, brightness_high, occlusion_medium.",
    "generated_file": "RCGAN ve upscale sonrası değerlendirilen sentetik görüntü dosyası.",
    "clean_reference": "Sentetik görüntünün karşılaştırıldığı orijinal clean referans görüntü.",
    "clean_object_count": "YOLO'nun clean görüntüde tespit ettiği nesne sayısı.",
    "generated_object_count": "YOLO'nun sentetik/bozulmuş görüntüde tespit ettiği nesne sayısı.",
    "clean_avg_confidence": "Clean görüntüdeki YOLO tahminlerinin ortalama güven skoru.",
    "generated_avg_confidence": "Sentetik görüntüdeki YOLO tahminlerinin ortalama güven skoru.",
    "detection_drop": "Clean nesne sayısı eksi sentetik nesne sayısı. Pozitif değer tespit kaybını gösterir.",
    "confidence_drop": "Clean güven skoru eksi sentetik güven skoru. Pozitif değer güven düşüşünü gösterir.",
    "detection_retention": "Sentetik görüntüde korunan tespit oranı. 1'e yakın değer daha iyidir.",
    "confidence_retention": "Sentetik görüntüde korunan güven oranı. 1'e yakın değer daha iyidir.",
    "pixel_agreement": "Clean ve sentetik segmentasyon tahminlerinin piksel bazlı uyuşma oranı.",
    "prediction_iou": "Clean ve sentetik segmentasyon tahminleri arasındaki IoU benzeri örtüşme skoru.",
    "distribution_shift": "Segmentasyon sınıf dağılımlarının clean ve sentetik görüntü arasında ne kadar değiştiği.",
    "robustness_drop": "Segmentasyon dayanıklılık kaybı. 1 - prediction_iou olarak hesaplanır.",
}


def debug_log(message):
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as log_file:
            log_file.write(str(message) + "\n")
    except OSError:
        pass
    print(str(message), flush=True)


def force_window_visible(window):
    screen = QApplication.primaryScreen()
    if screen is not None:
        available = screen.availableGeometry()
        width = min(1320, max(900, available.width() - 80))
        height = min(840, max(640, available.height() - 80))
        window.resize(width, height)
        window.move(available.x() + 40, available.y() + 40)

    window.show()
    window.setWindowState(window.windowState() & ~Qt.WindowState.WindowMinimized | Qt.WindowState.WindowActive)
    window.raise_()
    window.activateWindow()


def build_safe_preview(path):
    source = Path(ensure_local_file(path, timeout=10))
    target = PREVIEW_CACHE_DIR / f"{source.stem}_{abs(hash(str(source)))}.png"

    with Image.open(source) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        img.thumbnail((1800, 1800), Image.Resampling.LANCZOS)
        img.save(target, "PNG")

    return target


class ImagePreview(QLabel):
    def __init__(self, placeholder="Önizleme"):
        super().__init__(placeholder)

        self._image_path = None
        self._placeholder = placeholder

        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(260, 190)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.setObjectName("preview")

    def set_image(self, path):
        try:
            self._image_path = str(build_safe_preview(path))
        except Exception as exc:
            self.clear()
            self.setText(f"Görüntü açılamadı\n{exc}")
            self.setAlignment(Qt.AlignmentFlag.AlignCenter)
            return
        self._refresh_pixmap()

    def clear_image(self):
        self._image_path = None
        self.clear()
        self.setText(self._placeholder)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def resizeEvent(self, event):
        super().resizeEvent(event)

        if self._image_path:
            self._refresh_pixmap()

    def _refresh_pixmap(self):
        pixmap = QPixmap(self._image_path)

        if pixmap.isNull():
            self.clear()
            self.setText("Görüntü açılamadı")
            self.setAlignment(Qt.AlignmentFlag.AlignCenter)
            return

        scaled = pixmap.scaled(
            max(1, self.width() - 12),
            max(1, self.height() - 12),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        self.setPixmap(scaled)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)


class InferenceWorker(QThread):
    log = Signal(str)
    finished_ok = Signal(list)
    failed = Signal(str)

    def __init__(
        self,
        checkpoint,
        image_paths,
        output_dir,
        fault,
        severity,
        all_conditions,
        conditions=None,
        image_size=256,
    ):
        super().__init__()

        self.checkpoint = checkpoint
        self.image_paths = image_paths
        self.output_dir = output_dir
        self.fault = fault
        self.severity = severity
        self.all_conditions = all_conditions
        self.conditions = conditions
        self.image_size = image_size

    def run(self):
        try:
            output_paths = generate_sequence(
                checkpoint_path=self.checkpoint,
                image_paths=self.image_paths,
                output_dir=self.output_dir,
                fault=self.fault,
                severity=self.severity,
                all_conditions=self.all_conditions,
                conditions=self.conditions,
                image_size=self.image_size,
                log_callback=self.log.emit,
            )

            self.finished_ok.emit(output_paths)

        except Exception:
            self.failed.emit(traceback.format_exc())


class PipelineWorker(QThread):
    log = Signal(str)
    finished_ok = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        mode,
        detector_dir,
        clean_image_paths=None,
        generated_image_paths=None,
        checkpoint=None,
        output_dir=None,
        fault=None,
        severity=None,
        all_conditions=False,
        conditions=None,
        image_size=256,
    ):
        super().__init__()

        self.mode = mode
        self.detector_dir = Path(detector_dir)
        self.clean_image_paths = clean_image_paths or []
        self.generated_image_paths = generated_image_paths or []
        self.checkpoint = checkpoint
        self.output_dir = output_dir
        self.fault = fault
        self.severity = severity
        self.all_conditions = all_conditions
        self.conditions = conditions
        self.image_size = image_size

    def run(self):
        try:
            result = self.run_pipeline_step()
            self.finished_ok.emit(result)

        except Exception:
            self.failed.emit(traceback.format_exc())

    def run_pipeline_step(self):
        detector_dir = self.detector_dir.resolve()
        dataset_root = detector_dir / "robustness_dataset"
        outputs_root = detector_dir.parent / "outputs"
        results_root = detector_dir.parent / "results"

        if self.mode == "full":
            self.log.emit("\n--- Tam pipeline basladi ---")
            self.reset_pipeline_outputs(outputs_root, results_root)

            clean_prepared = prepare_project_clean_dataset(
                clean_image_paths=self.clean_image_paths,
                project_root=detector_dir.parent,
                clear_existing=True,
                log_callback=self.log.emit,
            )
            self.clean_image_paths = [Path(p) for p in clean_prepared["copied_clean"]]

            generated_paths = generate_sequence(
                checkpoint_path=self.checkpoint,
                image_paths=self.clean_image_paths,
                output_dir=self.output_dir,
                fault=self.fault,
                severity=self.severity,
                all_conditions=self.all_conditions,
                conditions=self.conditions,
                image_size=self.image_size,
                log_callback=self.log.emit,
            )

            prepared = self.prepare_dataset(
                detector_dir=detector_dir,
                generated_paths=generated_paths,
            )
            self.log_prepare_summary(prepared)

            generated_dirs = self.generated_dirs_from_prepare_result(
                dataset_root=dataset_root,
                prepared=prepared,
                root_name="generated",
            )

            upscaled = self.run_upscale(
                detector_dir=detector_dir,
                dataset_root=dataset_root,
                generated_dirs=generated_dirs,
                upscaled_root=outputs_root / "gan_upscaled",
            )
            upscaled_dirs = self.upscaled_dirs_from_result(
                upscaled=upscaled,
            )

            yolo = self.run_yolo(
                detector_dir=detector_dir,
                dataset_root=dataset_root,
                generated_dirs=upscaled_dirs,
                results_dir=results_root / "yolo",
            )
            segmentation = self.run_segmentation(
                detector_dir=detector_dir,
                dataset_root=dataset_root,
                generated_dirs=upscaled_dirs,
                output_dir=outputs_root / "segmentation_outputs",
                results_dir=results_root / "segmentation",
            )
            yolo_visuals = self.run_yolo_visuals(
                detector_dir=detector_dir,
                dataset_root=dataset_root,
                generated_dirs=upscaled_dirs,
                output_dir=outputs_root / "yolo_comparisons",
            )
            segmentation_visuals = self.run_segmentation_visuals(
                detector_dir=detector_dir,
                dataset_root=dataset_root,
                generated_dirs=upscaled_dirs,
                seg_output_root=outputs_root / "segmentation_outputs",
                output_dir=outputs_root / "segmentation_comparisons",
            )

            self.log.emit("\n--- Tam pipeline tamamlandi ---")

            return {
                "mode": self.mode,
                "generated_paths": generated_paths,
                "clean_prepared": clean_prepared,
                "prepared": prepared,
                "upscaled": upscaled,
                "yolo": yolo,
                "segmentation": segmentation,
                "yolo_visuals": yolo_visuals,
                "segmentation_visuals": segmentation_visuals,
            }

        if self.mode == "prepare":
            clean_prepared = prepare_project_clean_dataset(
                clean_image_paths=self.clean_image_paths,
                project_root=detector_dir.parent,
                clear_existing=True,
                log_callback=self.log.emit,
            )
            self.clean_image_paths = [Path(p) for p in clean_prepared["copied_clean"]]

            return {
                "mode": self.mode,
                "clean_prepared": clean_prepared,
                "prepared": self.prepare_dataset(
                    detector_dir=detector_dir,
                    generated_paths=self.generated_image_paths,
                ),
            }

        if self.mode == "upscale":
            return {
                "mode": self.mode,
                "upscaled": self.run_upscale(
                    detector_dir=detector_dir,
                    dataset_root=dataset_root,
                    upscaled_root=outputs_root / "gan_upscaled",
                ),
            }

        if self.mode == "yolo":
            generated_dirs = self.generated_dirs_from_root(outputs_root / "gan_upscaled")
            self.ensure_clean_staging(dataset_root)

            return {
                "mode": self.mode,
                "yolo": self.run_yolo(
                    detector_dir=detector_dir,
                    dataset_root=dataset_root,
                    generated_dirs=generated_dirs,
                    results_dir=results_root / "yolo",
                ),
            }

        if self.mode == "segmentation":
            generated_dirs = self.generated_dirs_from_root(outputs_root / "gan_upscaled")
            self.ensure_clean_staging(dataset_root)

            return {
                "mode": self.mode,
                "segmentation": self.run_segmentation(
                    detector_dir=detector_dir,
                    dataset_root=dataset_root,
                    generated_dirs=generated_dirs,
                    output_dir=outputs_root / "segmentation_outputs",
                    results_dir=results_root / "segmentation",
                ),
            }

        raise ValueError(f"Bilinmeyen pipeline modu: {self.mode}")

    def reset_pipeline_outputs(self, outputs_root, results_root):
        for folder in [
            outputs_root / "gan_generated",
            outputs_root / "gan_upscaled",
            outputs_root / "segmentation_outputs",
            outputs_root / "segmentation_comparisons",
            outputs_root / "yolo_comparisons",
            results_root / "yolo",
            results_root / "segmentation",
        ]:
            if folder.exists():
                shutil.rmtree(folder)

            folder.mkdir(parents=True, exist_ok=True)

        self.log.emit("Eski outputs/results pipeline klasorleri temizlendi.")

    def prepare_dataset(self, detector_dir, generated_paths):
        return prepare_detector_dataset(
            clean_image_paths=self.clean_image_paths,
            generated_image_paths=generated_paths,
            detector_root=detector_dir,
            log_callback=self.log.emit,
        )

    def log_prepare_summary(self, prepared):
        self.log.emit(f"Detector clean staging: {prepared['clean_dir']}")
        self.log.emit(f"Detector generated staging: {prepared['generated_root']}")
        self.log.emit(f"Kopyalanan clean sayisi: {len(prepared['copied_clean'])}")

        for condition, paths in prepared["copied_generated"].items():
            self.log.emit(f"Kopyalanan {condition} sayisi: {len(paths)}")

    def ensure_clean_staging(self, dataset_root):
        clean_dir = Path(dataset_root) / "clean"

        if clean_dir.exists() and any(clean_dir.iterdir()):
            return

        if not self.clean_image_paths:
            project_clean_dir = Path(dataset_root).parent.parent / "clean"

            if project_clean_dir.exists():
                self.clean_image_paths = [
                    path for path in sorted(project_clean_dir.iterdir())
                    if path.is_file() and path.suffix.lower() in {
                        ".png",
                        ".jpg",
                        ".jpeg",
                        ".bmp",
                        ".webp",
                    }
                ]

        if not self.clean_image_paths:
            raise FileNotFoundError(
                "Clean staging klasoru hazir degil. Once clean frame listesini secip "
                "`Detector Veri Setini Hazirla` veya tam pipeline adimini calistirmalisin."
            )

        self.log.emit("Clean staging eksik; secili clean frame listesi kopyalaniyor...")
        copied = copy_images(
            image_paths=self.clean_image_paths,
            target_dir=clean_dir,
            log_callback=self.log.emit,
        )
        self.log.emit(f"Clean staging hazirlandi: {len(copied)} dosya")

    def run_upscale(self, detector_dir, dataset_root, generated_dirs=None, upscaled_root=None):
        run_upscale_generated_images = self.import_detector_function(
            detector_dir,
            "upscale_generated_images",
            "run_upscale_generated_images",
        )

        return run_upscale_generated_images(
            dataset_root=str(dataset_root),
            generated_dirs=generated_dirs,
            upscaled_root=str(upscaled_root) if upscaled_root else None,
            model_path=str(detector_dir / "EDSR_x4.pb"),
            log_callback=self.log.emit,
        )

    def run_yolo(self, detector_dir, dataset_root, generated_dirs, results_dir):
        run_yolo_evaluation = self.import_detector_function(
            detector_dir,
            "yolo_robustness_evaluation",
            "run_yolo_evaluation",
        )

        return run_yolo_evaluation(
            dataset_root=str(dataset_root),
            results_dir=str(results_dir),
            yolo_model_path=str(detector_dir / "yolov8n.pt"),
            generated_dirs=generated_dirs,
            log_callback=self.log.emit,
        )

    def run_segmentation(
        self,
        detector_dir,
        dataset_root,
        generated_dirs,
        output_dir,
        results_dir,
    ):
        run_segmentation_evaluation = self.import_detector_function(
            detector_dir,
            "semantic_segmentation_robustness",
            "run_segmentation_evaluation",
        )

        return run_segmentation_evaluation(
            dataset_root=str(dataset_root),
            output_dir=str(output_dir),
            results_dir=str(results_dir),
            generated_dirs=generated_dirs,
            log_callback=self.log.emit,
        )

    def run_yolo_visuals(self, detector_dir, dataset_root, generated_dirs, output_dir):
        run_yolo_visual_failure_analysis = self.import_detector_function(
            detector_dir,
            "yolo_visual_failure_analysis",
            "run_yolo_visual_failure_analysis",
        )

        return run_yolo_visual_failure_analysis(
            dataset_root=str(dataset_root),
            generated_dirs=generated_dirs,
            output_dir=str(output_dir),
            yolo_model_path=str(detector_dir / "yolov8n.pt"),
            log_callback=self.log.emit,
        )

    def run_segmentation_visuals(
        self,
        detector_dir,
        dataset_root,
        generated_dirs,
        seg_output_root,
        output_dir,
    ):
        run_segmentation_visual_comparison = self.import_detector_function(
            detector_dir,
            "segmentation_visual_comparison",
            "run_segmentation_visual_comparison",
        )

        return run_segmentation_visual_comparison(
            dataset_root=str(dataset_root),
            generated_image_dirs=generated_dirs,
            seg_output_root=str(seg_output_root),
            output_dir=str(output_dir),
            log_callback=self.log.emit,
        )

    def generated_dirs_from_prepare_result(self, dataset_root, prepared, root_name):
        return {
            condition: str(dataset_root / root_name / condition)
            for condition in prepared["copied_generated"]
        }

    def upscaled_dirs_from_result(self, upscaled):
        return {
            condition: str(Path(upscaled["upscaled_root"]) / condition)
            for condition in upscaled["outputs"]
        }

    def generated_dirs_from_root(self, generated_root):
        generated_root = Path(generated_root)

        if generated_root.exists():
            dirs = {
                path.name: str(path)
                for path in sorted(generated_root.iterdir())
                if path.is_dir()
            }

            if dirs:
                return dirs

        return {
            "blur_high": str(generated_root / "blur_high"),
            "brightness_high": str(generated_root / "brightness_high"),
            "occlusion_high": str(generated_root / "occlusion_high"),
        }

    def import_detector_function(self, detector_dir, module_name, function_name):
        detector_path = str(detector_dir)

        if detector_path not in sys.path:
            sys.path.insert(0, detector_path)

        module = __import__(module_name, fromlist=[function_name])
        return getattr(module, function_name)


class RCGANQtApp(QWidget):
    def __init__(self):
        super().__init__()
        debug_log("[Görüntü] RCGANQtApp kuruluyor...")

        self.setWindowTitle(APP_TITLE)
        self.resize(1320, 840)

        self.worker = None
        self.pipeline_worker = None
        self.output_paths = []
        self.selected_images = []
        self.active_artifact_group = "outputs"

        self._build_ui()
        self._apply_styles()
        self.refresh_pipeline_outputs()
        debug_log("[Görüntü] Arayüz bileşenleri hazır.")

    def closeEvent(self, event):
        app = QApplication.instance()
        if app is not None:
            app.quit()
        super().closeEvent(event)

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        main_shell = QHBoxLayout()
        main_shell.setSpacing(10)
        root.addLayout(main_shell)

        nav_panel = QWidget()
        nav_panel.setObjectName("mainNav")
        nav_layout = QVBoxLayout(nav_panel)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(10)

        self.files_nav_btn = QPushButton("1  Dosyalar")
        self.conditions_nav_btn = QPushButton("2  Kosullar")
        self.generate_nav_btn = QPushButton("3  Uret")
        self.run_nav_btn = QPushButton("4  Calistir")
        self.results_nav_btn = QPushButton("Sonuclar")
        self.main_nav_buttons = [
            self.files_nav_btn,
            self.conditions_nav_btn,
            self.generate_nav_btn,
            self.run_nav_btn,
            self.results_nav_btn,
        ]

        for button in self.main_nav_buttons:
            button.setObjectName("navButton")
            button.setCheckable(True)

        self.files_nav_btn.setChecked(True)
        self.set_button_icon(self.files_nav_btn, QStyle.StandardPixmap.SP_FileDialogDetailedView)
        self.set_button_icon(self.conditions_nav_btn, QStyle.StandardPixmap.SP_FileDialogContentsView)
        self.set_button_icon(self.generate_nav_btn, QStyle.StandardPixmap.SP_MediaPlay)
        self.set_button_icon(self.run_nav_btn, QStyle.StandardPixmap.SP_ComputerIcon)
        self.set_button_icon(self.results_nav_btn, QStyle.StandardPixmap.SP_FileDialogInfoView)

        for button in self.main_nav_buttons:
            nav_layout.addWidget(button)

        nav_layout.addStretch(1)

        self.main_tabs = QStackedWidget()
        self.main_tabs.setObjectName("mainStack")

        main_shell.addWidget(nav_panel, 0)
        main_shell.addWidget(self.main_tabs, 1)

        pipeline_page = QWidget()
        pipeline_root = QHBoxLayout(pipeline_page)
        pipeline_root.setContentsMargins(0, 0, 0, 0)
        pipeline_root.setSpacing(10)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setObjectName("controlScroll")
        left_scroll.setMinimumWidth(520)
        left_scroll.setMaximumWidth(620)

        left_panel = QWidget()
        left_panel.setObjectName("controlPanel")
        left_panel.setMinimumWidth(500)

        left = QVBoxLayout(left_panel)
        right = QVBoxLayout()
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(8)
        right.setSpacing(8)

        left_scroll.setWidget(left_panel)

        pipeline_root.addWidget(left_scroll, 0)
        pipeline_root.addLayout(right, 1)

        title = QLabel("RCGAN Pipeline")
        title.setObjectName("title")
        left.addWidget(title)

        self.control_stack = QStackedWidget()
        self.control_stack.setObjectName("controlStack")

        file_group = QGroupBox("1) Dosyaları Seç")
        file_layout = QGridLayout(file_group)
        file_layout.setColumnMinimumWidth(0, 120)
        file_layout.setColumnStretch(0, 0)
        file_layout.setColumnStretch(1, 1)
        file_layout.setColumnStretch(2, 0)
        file_layout.setHorizontalSpacing(8)
        file_layout.setVerticalSpacing(8)

        self.checkpoint_edit = QLineEdit(DEFAULT_CHECKPOINT)
        self.output_edit = QLineEdit(DEFAULT_OUTPUT_DIR)
        self.detector_edit = QLineEdit(DEFAULT_DETECTOR_DIR)
        for edit in (self.checkpoint_edit, self.output_edit, self.detector_edit):
            edit.setMinimumWidth(0)

        self._add_file_row(
            file_layout,
            0,
            "Checkpoint",
            self.checkpoint_edit,
            self.pick_checkpoint,
        )

        self.select_images_btn = QPushButton("Fotograf Sec")
        self.select_folder_btn = QPushButton("Klasorden Al")

        self.set_button_icon(self.select_images_btn, QStyle.StandardPixmap.SP_FileDialogDetailedView)
        self.set_button_icon(self.select_folder_btn, QStyle.StandardPixmap.SP_DirOpenIcon)

        self.select_images_btn.clicked.connect(self.pick_multiple_images)
        self.select_folder_btn.clicked.connect(self.pick_image_folder)

        file_layout.addWidget(QLabel("Zaman serisi:"), 1, 0)
        file_layout.addWidget(self.select_images_btn, 1, 1)
        file_layout.addWidget(self.select_folder_btn, 1, 2)

        self.input_list = QListWidget()
        self.input_list.itemSelectionChanged.connect(self.preview_selected_input)

        file_layout.addWidget(QLabel("Seçilen clean frame listesi:"), 2, 0)
        file_layout.addWidget(self.input_list, 2, 1, 1, 2)

        self._add_file_row(
            file_layout,
            3,
            "Output",
            self.output_edit,
            self.pick_output_dir,
            directory=True,
        )

        self._add_file_row(
            file_layout,
            4,
            "Detector",
            self.detector_edit,
            self.pick_detector_dir,
            directory=True,
        )

        self.control_stack.addWidget(file_group)

        condition_group = QGroupBox("2) Üretim Koşulları")
        condition_layout = QGridLayout(condition_group)

        self.fault_combo = QComboBox()
        self.fault_combo.addItems(list(FAULT_MAP.keys()))

        self.severity_combo = QComboBox()
        self.severity_combo.addItems(list(SEVERITY_MAP.keys()))

        self.blur_severity_combo = QComboBox()
        self.blur_severity_combo.addItems(list(SEVERITY_MAP.keys()))

        self.occlusion_severity_combo = QComboBox()
        self.occlusion_severity_combo.addItems(list(SEVERITY_MAP.keys()))

        self.brightness_severity_combo = QComboBox()
        self.brightness_severity_combo.addItems(list(SEVERITY_MAP.keys()))

        self.image_size_combo = QComboBox()
        self.image_size_combo.addItems(["128 - Hızlı", "192 - Dengeli", "256 - Kalite"])
        self.image_size_combo.setCurrentIndex(0)

        condition_layout.addWidget(QLabel("Tek hata tipi:"), 0, 0)
        condition_layout.addWidget(self.fault_combo, 0, 1)

        condition_layout.addWidget(QLabel("Tek hata seviyesi:"), 1, 0)
        condition_layout.addWidget(self.severity_combo, 1, 1)

        condition_layout.addWidget(QLabel("Blur seviyesi:"), 2, 0)
        condition_layout.addWidget(self.blur_severity_combo, 2, 1)

        condition_layout.addWidget(QLabel("Occlusion seviyesi:"), 3, 0)
        condition_layout.addWidget(self.occlusion_severity_combo, 3, 1)

        condition_layout.addWidget(QLabel("Brightness seviyesi:"), 4, 0)
        condition_layout.addWidget(self.brightness_severity_combo, 4, 1)

        condition_layout.addWidget(QLabel("Üretim çözünürlüğü:"), 5, 0)
        condition_layout.addWidget(self.image_size_combo, 5, 1)

        self.control_stack.addWidget(condition_group)

        action_group = QGroupBox("3) Üret")
        action_layout = QVBoxLayout(action_group)
        action_layout.setContentsMargins(12, 18, 12, 12)
        action_layout.setSpacing(10)

        self.generate_one_btn = QPushButton("Tek Kosulla Uret")
        self.generate_selected_faults_btn = QPushButton("3 Hata Tipini Uret")
        self.open_output_btn = QPushButton("GAN Klasoru")
        self.open_outputs_root_btn = QPushButton("Outputs")
        self.open_results_root_btn = QPushButton("Results")
        self.clear_btn = QPushButton("Temizle")

        self.generate_one_btn.setObjectName("primaryButton")
        self.generate_selected_faults_btn.setObjectName("primaryButton")
        self.clear_btn.setObjectName("mutedButton")
        self.set_button_icon(self.generate_one_btn, QStyle.StandardPixmap.SP_MediaPlay)
        self.set_button_icon(self.generate_selected_faults_btn, QStyle.StandardPixmap.SP_MediaPlay)
        self.set_button_icon(self.open_output_btn, QStyle.StandardPixmap.SP_DirOpenIcon)
        self.set_button_icon(self.open_outputs_root_btn, QStyle.StandardPixmap.SP_DirOpenIcon)
        self.set_button_icon(self.open_results_root_btn, QStyle.StandardPixmap.SP_DirOpenIcon)
        self.set_button_icon(self.clear_btn, QStyle.StandardPixmap.SP_TrashIcon)

        self.generate_one_btn.clicked.connect(self.start_generation_single)
        self.generate_selected_faults_btn.clicked.connect(self.start_generation_selected_faults)
        self.open_output_btn.clicked.connect(self.open_output_folder)
        self.open_outputs_root_btn.clicked.connect(lambda: self.open_folder(DEFAULT_OUTPUTS_ROOT))
        self.open_results_root_btn.clicked.connect(lambda: self.open_folder(DEFAULT_RESULTS_ROOT))
        self.clear_btn.clicked.connect(self.clear_outputs)

        action_layout.addWidget(self.generate_one_btn)
        action_layout.addWidget(self.generate_selected_faults_btn)
        action_layout.addWidget(self.open_output_btn)
        action_layout.addWidget(self.open_outputs_root_btn)
        action_layout.addWidget(self.open_results_root_btn)
        action_layout.addWidget(self.clear_btn)
        action_layout.addStretch(1)

        self.control_stack.addWidget(action_group)

        pipeline_group = QGroupBox("4) Pipeline")
        pipeline_layout = QVBoxLayout(pipeline_group)
        pipeline_layout.setContentsMargins(12, 18, 12, 12)
        pipeline_layout.setSpacing(10)

        self.pipeline_one_btn = QPushButton("Tam Pipeline: Tek Kosul")
        self.pipeline_selected_faults_btn = QPushButton("Tam Pipeline: 3 Hata")
        self.prepare_detector_btn = QPushButton("Veri Setini Hazirla")
        self.upscale_btn = QPushButton("Upscale")
        self.yolo_btn = QPushButton("YOLO")
        self.segmentation_btn = QPushButton("Segmentasyon")

        self.pipeline_one_btn.setObjectName("accentButton")
        self.pipeline_selected_faults_btn.setObjectName("accentButton")
        self.set_button_icon(self.pipeline_one_btn, QStyle.StandardPixmap.SP_MediaPlay)
        self.set_button_icon(self.pipeline_selected_faults_btn, QStyle.StandardPixmap.SP_MediaPlay)
        self.set_button_icon(self.prepare_detector_btn, QStyle.StandardPixmap.SP_DriveHDIcon)
        self.set_button_icon(self.upscale_btn, QStyle.StandardPixmap.SP_ArrowUp)
        self.set_button_icon(self.yolo_btn, QStyle.StandardPixmap.SP_FileDialogInfoView)
        self.set_button_icon(self.segmentation_btn, QStyle.StandardPixmap.SP_FileDialogContentsView)

        self.pipeline_one_btn.clicked.connect(self.start_full_pipeline_single)
        self.pipeline_selected_faults_btn.clicked.connect(self.start_full_pipeline_selected_faults)
        self.prepare_detector_btn.clicked.connect(self.start_prepare_detector)
        self.upscale_btn.clicked.connect(self.start_upscale)
        self.yolo_btn.clicked.connect(self.start_yolo)
        self.segmentation_btn.clicked.connect(self.start_segmentation)

        pipeline_layout.addWidget(self.pipeline_one_btn)
        pipeline_layout.addWidget(self.pipeline_selected_faults_btn)
        pipeline_layout.addWidget(self.prepare_detector_btn)
        pipeline_layout.addWidget(self.upscale_btn)
        pipeline_layout.addWidget(self.yolo_btn)
        pipeline_layout.addWidget(self.segmentation_btn)
        pipeline_layout.addStretch(1)

        self.control_stack.addWidget(pipeline_group)

        status_group = QGroupBox("5) Pipeline Durumu")
        status_layout = QVBoxLayout(status_group)

        self.refresh_status_btn = QPushButton("Model ve Paketleri Kontrol Et")
        self.set_button_icon(self.refresh_status_btn, QStyle.StandardPixmap.SP_BrowserReload)
        self.refresh_status_btn.clicked.connect(self.refresh_pipeline_status)

        self.pipeline_status_box = QTextEdit()
        self.pipeline_status_box.setReadOnly(True)
        self.pipeline_status_box.setMinimumHeight(150)
        self.pipeline_status_box.setPlaceholderText("Pipeline modeli ve paket durumu burada gorunecek...")

        status_layout.addWidget(self.refresh_status_btn)
        status_layout.addWidget(self.pipeline_status_box)

        left.addWidget(self.control_stack, 1)

        preview_title = QLabel("Önizleme ve Çıktılar")
        preview_title.setObjectName("subtitle")
        right.addWidget(preview_title)

        preview_row = QHBoxLayout()

        self.curr_preview = ImagePreview("Seçilen clean frame önizleme")
        self.out_preview = ImagePreview("Üretilen görüntü önizleme")
        self.curr_preview.setMaximumHeight(320)
        self.out_preview.setMaximumHeight(320)

        preview_row.addWidget(self.curr_preview, 1)
        preview_row.addWidget(self.out_preview, 1)

        right.addLayout(preview_row, 2)

        self.output_list = QListWidget()
        self.output_list.itemSelectionChanged.connect(self.preview_selected_output)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setPlaceholderText("İşlem kayıtları burada görünecek...")

        tabs = QTabWidget()
        self.pipeline_tabs = tabs
        output_tab = QWidget()
        output_layout = QVBoxLayout(output_tab)
        output_layout.addWidget(self.output_list)

        log_tab = QWidget()
        log_layout = QVBoxLayout(log_tab)
        log_layout.addWidget(self.log_box)

        tabs.addTab(output_tab, "Çıktılar")
        results_tab = QWidget()
        artifacts_layout = QVBoxLayout(results_tab)

        results_title = QLabel("Model Ciktilari ve Sayisal Sonuclar")
        results_title.setObjectName("subtitle")
        artifacts_layout.addWidget(results_title)

        artifact_filter_buttons = QHBoxLayout()
        self.outputs_filter_btn = QPushButton("Outputs")
        self.results_filter_btn = QPushButton("Results")
        self.outputs_filter_btn.setObjectName("filterButton")
        self.results_filter_btn.setObjectName("filterButton")
        self.outputs_filter_btn.setCheckable(True)
        self.results_filter_btn.setCheckable(True)
        self.outputs_filter_btn.setChecked(True)
        self.set_button_icon(self.outputs_filter_btn, QStyle.StandardPixmap.SP_DirIcon)
        self.set_button_icon(self.results_filter_btn, QStyle.StandardPixmap.SP_FileDialogInfoView)
        self.outputs_filter_btn.clicked.connect(lambda: self.set_artifact_group("outputs"))
        self.results_filter_btn.clicked.connect(lambda: self.set_artifact_group("results"))
        artifact_filter_buttons.addWidget(self.outputs_filter_btn)
        artifact_filter_buttons.addWidget(self.results_filter_btn)
        artifacts_layout.addLayout(artifact_filter_buttons)

        artifact_buttons = QHBoxLayout()
        self.refresh_artifacts_btn = QPushButton("Pipeline Ciktilarini Yenile")
        self.open_artifact_btn = QPushButton("Secili Dosyayi Ac")
        self.open_artifact_folder_btn = QPushButton("Secili Klasoru Ac")

        self.refresh_artifacts_btn.setObjectName("primaryButton")
        self.set_button_icon(self.refresh_artifacts_btn, QStyle.StandardPixmap.SP_BrowserReload)
        self.set_button_icon(self.open_artifact_btn, QStyle.StandardPixmap.SP_FileIcon)
        self.set_button_icon(self.open_artifact_folder_btn, QStyle.StandardPixmap.SP_DirOpenIcon)

        self.refresh_artifacts_btn.clicked.connect(self.refresh_pipeline_outputs)
        self.open_artifact_btn.clicked.connect(self.open_selected_artifact)
        self.open_artifact_folder_btn.clicked.connect(self.open_selected_artifact_folder)

        artifact_buttons.addWidget(self.refresh_artifacts_btn)
        artifact_buttons.addWidget(self.open_artifact_btn)
        artifact_buttons.addWidget(self.open_artifact_folder_btn)

        self.artifact_list = QListWidget()
        self.artifact_list.itemSelectionChanged.connect(self.preview_selected_artifact)

        self.artifact_preview = ImagePreview("Pipeline ciktisi onizleme")
        self.artifact_text_preview = QTextEdit()
        self.artifact_text_preview.setReadOnly(True)
        self.artifact_text_preview.setMinimumHeight(130)
        self.artifact_text_preview.setPlaceholderText("CSV ve metin ciktisi burada gorunecek...")

        self.artifact_table_preview = QTableWidget()
        self.artifact_table_preview.setObjectName("metricTable")
        self.artifact_table_preview.setAlternatingRowColors(True)
        self.artifact_table_preview.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.artifact_table_preview.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.artifact_table_preview.horizontalHeader().setStretchLastSection(True)
        self.artifact_table_preview.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.artifact_table_preview.verticalHeader().setVisible(False)
        self.artifact_table_preview.setVisible(False)

        results_body = QHBoxLayout()
        results_left = QVBoxLayout()
        results_right = QVBoxLayout()

        results_left.addLayout(artifact_buttons)
        results_left.addWidget(self.artifact_list, 1)
        results_right.addWidget(self.artifact_preview, 2)
        results_right.addWidget(self.artifact_table_preview, 2)
        results_right.addWidget(self.artifact_text_preview, 1)

        results_body.addLayout(results_left, 2)
        results_body.addLayout(results_right, 3)
        artifacts_layout.addLayout(results_body, 1)

        tabs.addTab(log_tab, "Log")

        right.addWidget(tabs, 1)

        hint = QLabel(
            "Not: Seçilen görüntüler dosya adına göre sıralanır. "
            "İlk görüntü için çıktı üretilmez çünkü önceki frame yoktur. "
            "2. görüntüden itibaren prev + curr şeklinde zaman serisi üretimi yapılır."
        )
        hint.setWordWrap(True)
        hint.setObjectName("hint")

        right.addWidget(hint)

        self.main_tabs.addWidget(pipeline_page)
        self.main_tabs.addWidget(results_tab)
        self.files_nav_btn.clicked.connect(lambda: self.switch_main_page(0))
        self.conditions_nav_btn.clicked.connect(lambda: self.switch_main_page(1))
        self.generate_nav_btn.clicked.connect(lambda: self.switch_main_page(2))
        self.run_nav_btn.clicked.connect(lambda: self.switch_main_page(3))
        self.results_nav_btn.clicked.connect(lambda: self.switch_main_page(4))

        self._action_buttons = [
            self.generate_one_btn,
            self.generate_selected_faults_btn,
            self.pipeline_one_btn,
            self.pipeline_selected_faults_btn,
            self.prepare_detector_btn,
            self.upscale_btn,
            self.yolo_btn,
            self.segmentation_btn,
            self.open_output_btn,
            self.open_outputs_root_btn,
            self.open_results_root_btn,
            self.refresh_artifacts_btn,
            self.open_artifact_btn,
            self.open_artifact_folder_btn,
            self.outputs_filter_btn,
            self.results_filter_btn,
            self.clear_btn,
            self.select_images_btn,
            self.select_folder_btn,
        ]

    def switch_main_page(self, index):
        if index == 4:
            self.main_tabs.setCurrentIndex(1)
        else:
            self.main_tabs.setCurrentIndex(0)
            self.control_stack.setCurrentIndex(index)

        for button_index, button in enumerate(self.main_nav_buttons):
            button.setChecked(button_index == index)

    def show_log_tab(self):
        if hasattr(self, "pipeline_tabs"):
            self.pipeline_tabs.setCurrentIndex(1)

    def append_log(self, message):
        self.show_log_tab()
        self.log_box.append(str(message))
        QApplication.processEvents()

    def set_button_icon(self, button, standard_pixmap):
        button.setIcon(self.style().standardIcon(standard_pixmap))
        button.setIconSize(QSize(18, 18))

    def _add_file_row(self, layout, row, label_text, edit, picker, directory=False):
        layout.addWidget(QLabel(label_text + ":"), row, 0)
        layout.addWidget(edit, row, 1)

        btn = QPushButton("Seç")
        self.set_button_icon(btn, QStyle.StandardPixmap.SP_DialogOpenButton)
        btn.setMinimumWidth(72)
        btn.setMaximumWidth(92)
        btn.clicked.connect(picker)

        layout.addWidget(btn, row, 2)

    def _apply_styles(self):
        self.setStyleSheet("""
            QWidget {
                font-family: 'SF Pro Display', 'Segoe UI', sans-serif;
                font-size: 14px;
                color: #e8ecf4;
                background: #06080f;
            }

            QWidget#controlPanel {
                background: transparent;
            }

            QScrollArea#controlScroll {
                border: none;
                background: transparent;
            }

            QScrollBar:vertical {
                background: transparent;
                width: 6px;
                margin: 0;
                border-radius: 5px;
            }

            QScrollBar::handle:vertical {
                background: rgba(34, 211, 238, 0.35);
                min-height: 32px;
                border-radius: 3px;
            }

            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0;
            }

            QLabel#title {
                font-size: 23px;
                font-weight: 700;
                margin: 2px 0 6px 0;
                color: #ffffff;
            }

            QLabel#subtitle {
                font-size: 17px;
                font-weight: 700;
                color: #ffffff;
            }

            QLabel#hint {
                color: #a0aec0;
                padding: 11px 12px;
                background: #0c1017;
                border: 1px solid rgba(255, 255, 255, 0.06);
                border-radius: 8px;
            }

            QLabel#preview {
                border: 1px solid rgba(148, 163, 184, 0.16);
                border-radius: 8px;
                background: #10151f;
            }

            QPushButton {
                min-height: 20px;
                padding: 8px 12px;
                border-radius: 10px;
                border: 1px solid rgba(255, 255, 255, 0.06);
                background: #111620;
                color: #a0aec0;
                font-weight: 600;
            }

            QPushButton:hover {
                background: #181e2a;
                border-color: rgba(99, 102, 241, 0.3);
                color: #e8ecf4;
            }

            QPushButton:pressed {
                background: #2f5fbf;
            }

            QPushButton:disabled {
                color: #687180;
                background: #171a20;
                border-color: #272c35;
            }

            QPushButton#primaryButton {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #6366f1, stop:1 #8b5cf6
                );
                border-color: rgba(99, 102, 241, 0.4);
                color: #ffffff;
                font-weight: 700;
            }

            QPushButton#primaryButton:hover {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #818cf8, stop:1 #a78bfa
                );
                border-color: rgba(139, 92, 246, 0.6);
            }

            QPushButton#accentButton {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #6d5dfc, stop:1 #22d3ee
                );
                border-color: rgba(34, 211, 238, 0.35);
                color: #ffffff;
                font-weight: 700;
            }

            QPushButton#accentButton:hover {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #818cf8, stop:1 #67e8f9
                );
                border-color: rgba(34, 211, 238, 0.55);
            }

            QPushButton#mutedButton {
                color: #c4cad4;
                background: #191c22;
                border-color: #303744;
            }

            QWidget#mainNav {
                min-width: 150px;
                max-width: 170px;
                background: #0d121b;
                border: 1px solid rgba(148, 163, 184, 0.16);
                border-radius: 12px;
                padding: 10px;
            }

            QPushButton#navButton {
                min-height: 40px;
                padding: 0 10px;
                text-align: left;
                border-radius: 8px;
                border: 1px solid transparent;
                background: transparent;
                color: #a8b3c7;
                font-weight: 800;
            }

            QPushButton#navButton:hover {
                color: #eef2ff;
                background: rgba(109, 93, 252, 0.10);
                border-color: rgba(109, 93, 252, 0.24);
            }

            QPushButton#navButton:checked {
                color: #ffffff;
                background: rgba(109, 93, 252, 0.18);
                border-color: rgba(109, 93, 252, 0.45);
            }

            QPushButton#filterButton {
                min-height: 38px;
                text-align: left;
                border-radius: 8px;
                background: #111620;
                color: #a8b3c7;
                border: 1px solid rgba(148, 163, 184, 0.16);
                font-weight: 800;
            }

            QPushButton#filterButton:hover {
                color: #eef2ff;
                border-color: rgba(34, 211, 238, 0.35);
            }

            QPushButton#filterButton:checked {
                color: #ffffff;
                background: rgba(34, 211, 238, 0.12);
                border-color: rgba(34, 211, 238, 0.45);
            }

            QLineEdit,
            QComboBox {
                min-height: 20px;
                padding: 6px 8px;
                border-radius: 8px;
                border: 1px solid #343b48;
                background: #10151f;
                color: #e8ecf4;
                selection-background-color: rgba(99, 102, 241, 0.35);
            }

            QLineEdit:focus,
            QComboBox:focus {
                border-color: rgba(34, 211, 238, 0.55);
            }

            QComboBox::drop-down {
                border: none;
                padding-right: 8px;
            }

            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid #22d3ee;
            }

            QComboBox QAbstractItemView {
                background: #10151f;
                color: #e8ecf4;
                border: 1px solid rgba(148, 163, 184, 0.16);
                selection-background-color: rgba(99, 102, 241, 0.35);
            }

            QTextEdit,
            QListWidget {
                border: 1px solid rgba(148, 163, 184, 0.16);
                border-radius: 8px;
                background: #10151f;
                color: #e7edf6;
                selection-background-color: rgba(99, 102, 241, 0.35);
                selection-color: #ffffff;
                padding: 4px;
            }

            QTableWidget#metricTable {
                border: 1px solid rgba(148, 163, 184, 0.16);
                border-radius: 8px;
                gridline-color: rgba(148, 163, 184, 0.16);
                background: #10151f;
                alternate-background-color: #151b27;
                color: #e8ecf4;
                selection-background-color: rgba(109, 93, 252, 0.35);
                selection-color: #ffffff;
            }

            QTableWidget#metricTable::item {
                padding: 6px;
            }

            QHeaderView::section {
                padding: 8px 10px;
                border: none;
                border-right: 1px solid rgba(148, 163, 184, 0.16);
                border-bottom: 1px solid rgba(148, 163, 184, 0.16);
                background: #151b27;
                color: #22d3ee;
                font-weight: 800;
            }

            QTabWidget::pane {
                border: 1px solid rgba(148, 163, 184, 0.16);
                border-radius: 8px;
                background: #10151f;
            }

            QTabBar::tab {
                padding: 9px 14px;
                border: 1px solid rgba(148, 163, 184, 0.16);
                border-bottom: none;
                background: #111620;
                color: #a0aec0;
                border-top-left-radius: 7px;
                border-top-right-radius: 7px;
            }

            QTabBar::tab:hover {
                color: #ffffff;
                background: #181e2a;
            }

            QTabBar::tab:selected {
                background: #10151f;
                color: #ffffff;
                font-weight: 700;
            }

            QTabWidget#mainTabs::pane {
                border: none;
                background: transparent;
            }

            QTabBar#mainTabBar::tab {
                min-width: 104px;
                min-height: 46px;
                padding: 12px 10px;
                margin: 0 8px 8px 0;
                border: 1px solid #303744;
                border-radius: 8px;
                background: #181b20;
                color: #aeb7c5;
            }

            QTabBar#mainTabBar::tab:selected {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #6366f1, stop:1 #8b5cf6
                );
                border-color: rgba(139, 92, 246, 0.6);
                color: #ffffff;
            }

            QGroupBox {
                font-weight: 700;
                margin-top: 10px;
                padding: 12px 10px 10px 10px;
                border: 1px solid rgba(255, 255, 255, 0.06);
                border-radius: 12px;
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(17,22,32,0.95),
                    stop:1 rgba(24,30,42,0.90)
                );
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 6px;
                color: #e8ecf4;
            }
        """)

    def pick_checkpoint(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Checkpoint seç",
            os.getcwd(),
            "PyTorch Checkpoint (*.pt *.pth);;All Files (*)",
        )

        if path:
            self.checkpoint_edit.setText(path)

    def pick_output_dir(self):
        path = QFileDialog.getExistingDirectory(
            self,
            "Output klasörü seç",
            os.getcwd(),
        )

        if path:
            self.output_edit.setText(path)

    def pick_detector_dir(self):
        path = QFileDialog.getExistingDirectory(
            self,
            "Detector klasörü seç",
            os.getcwd(),
        )

        if path:
            self.detector_edit.setText(path)

    def pick_multiple_images(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Zaman serisi görüntülerini seç",
            os.getcwd(),
            "Images (*.png *.jpg *.jpeg *.bmp *.webp);;All Files (*)",
        )

        if paths:
            try:
                self.selected_images = [Path(p) for p in paths]
                self.refresh_input_list()
            except Exception as e:
                self.log_box.append(f"Fotoğraf yükleme hatası: {e}")
                QMessageBox.warning(self, "Fotoğraf yükleme hatası", str(e))

    def pick_image_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "Görüntü klasörü seç",
            os.getcwd(),
        )

        if folder:
            try:
                images = collect_images_from_folder(folder)
                self.selected_images = [Path(p) for p in images]
                self.refresh_input_list()

            except Exception as e:
                self.log_box.append(f"Klasör hatası: {e}")
                QMessageBox.warning(self, "Klasör hatası", str(e))

    def prepare_project_clean_images(self, image_paths):
        prepared = prepare_project_clean_dataset(
            clean_image_paths=image_paths,
            project_root=PROJECT_ROOT,
            clear_existing=True,
            log_callback=self.log_box.append,
        )

        return [Path(p) for p in prepared["copied_clean"]]

    def refresh_input_list(self):
        self.selected_images = sorted(
            self.selected_images,
            key=natural_sort_key,
        )

        self.input_list.clear()

        for path in self.selected_images:
            self.input_list.addItem(str(path))

        self.log_box.append(
            f"{len(self.selected_images)} adet görüntü seçildi ve dosya adına göre sıralandı."
        )

        if len(self.selected_images) >= 2:
            expected_outputs = len(self.selected_images) - 1
            self.log_box.append(
                f"Seçili koşulda üretilecek çıktı sayısı: {expected_outputs}"
            )

        if self.selected_images:
            self.curr_preview.clear_image()
            self.curr_preview.setText("Görüntüler listeye alındı\nÖnizleme için listeden bir dosya seç")

    def preview_selected_input(self):
        items = self.input_list.selectedItems()

        if not items:
            return

        self.show_image(items[0].text(), self.curr_preview)

    def validate_inputs(self):
        checkpoint = self.checkpoint_edit.text().strip()
        out_dir = self.output_edit.text().strip() or DEFAULT_OUTPUT_DIR

        missing = []

        if not Path(checkpoint).exists():
            missing.append(f"Checkpoint bulunamadı: {checkpoint}")

        if len(self.selected_images) < 2:
            missing.append("Zaman serisi üretimi için en az 2 görüntü seçmelisin.")

        for path in self.selected_images:
            if not Path(path).exists():
                missing.append(f"Görüntü bulunamadı: {path}")

        if missing:
            self.append_log("Pipeline/uretim baslatilamadi. Eksik girdiler:")
            for item in missing:
                self.append_log(f"- {item}")
            QMessageBox.warning(
                self,
                "Eksik dosya",
                "\n".join(missing),
            )
            return None

        return checkpoint, self.selected_images, out_dir

    def validate_detector_dir(self):
        detector_dir = self.detector_edit.text().strip() or DEFAULT_DETECTOR_DIR
        detector_path = Path(detector_dir)

        missing = []

        if not detector_path.exists():
            missing.append(f"Detector klasörü bulunamadı: {detector_dir}")

        for name in [
            "upscale_generated_images.py",
            "yolo_robustness_evaluation.py",
            "yolo_visual_failure_analysis.py",
            "semantic_segmentation_robustness.py",
            "segmentation_visual_comparison.py",
            "EDSR_x4.pb",
            "yolov8n.pt",
        ]:
            if not (detector_path / name).exists():
                missing.append(f"Detector dosyası bulunamadı: {detector_path / name}")

        if missing:
            self.append_log("Detector adimi baslatilamadi. Eksik dosyalar:")
            for item in missing:
                self.append_log(f"- {item}")
            QMessageBox.warning(
                self,
                "Detector eksik",
                "\n".join(missing),
            )
            return None

        return detector_path

    def selected_fault_conditions(self):
        return [
            ("blur", self.blur_severity_combo.currentText()),
            ("occlusion", self.occlusion_severity_combo.currentText()),
            ("brightness", self.brightness_severity_combo.currentText()),
        ]

    def selected_image_size(self):
        return int(self.image_size_combo.currentText().split(" ", 1)[0])

    def dependency_available(self, module_name):
        return importlib.util.find_spec(module_name) is not None

    def build_pipeline_status(self, mode="full"):
        detector_dir = Path(self.detector_edit.text().strip() or DEFAULT_DETECTOR_DIR)
        checkpoint = Path(self.checkpoint_edit.text().strip() or DEFAULT_CHECKPOINT)
        edsr_model = detector_dir / "EDSR_x4.pb"
        yolo_model = detector_dir / "yolov8n.pt"

        lines = []
        issues = []

        def add_file_status(label, path, required=True):
            if path.exists():
                size_mb = path.stat().st_size / (1024 * 1024)
                lines.append(f"OK  {label}: {path} ({size_mb:.1f} MB)")
            else:
                message = f"EKSIK  {label}: {path}"
                lines.append(message)
                if required:
                    issues.append(message)

        def add_package_status(label, module_name, required=True):
            if self.dependency_available(module_name):
                lines.append(f"OK  {label}: {module_name}")
            else:
                message = f"EKSIK  {label}: pip install {module_name}"
                lines.append(message)
                if required:
                    issues.append(message)

        lines.append("Pipeline on kontrol")
        lines.append("")

        if mode in {"full", "generate"}:
            add_file_status("RCGAN checkpoint", checkpoint)

        if mode in {"full", "upscale"}:
            add_file_status("EDSR modeli", edsr_model)
            try:
                import cv2

                if hasattr(cv2, "dnn_superres"):
                    lines.append("OK  OpenCV dnn_superres: opencv-contrib-python aktif")
                else:
                    message = "EKSIK  OpenCV dnn_superres: opencv-contrib-python gerekli"
                    lines.append(message)
                    issues.append(message)
            except Exception as exc:
                message = f"EKSIK  OpenCV import edilemedi: {exc}"
                lines.append(message)
                issues.append(message)

        if mode in {"full", "yolo"}:
            add_file_status("YOLO modeli", yolo_model)
            add_package_status("YOLO paketi", "ultralytics")

        if mode in {"full", "segmentation"}:
            add_package_status("SegFormer paketi", "transformers")
            add_package_status("Safetensors", "safetensors", required=False)
            add_package_status("Accelerate", "accelerate", required=False)

            try:
                from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor  # noqa: F401

                lines.append("OK  SegFormer siniflari import ediliyor")
                lines.append("OK  SegFormer model adi: nvidia/segformer-b0-finetuned-cityscapes-1024-1024")
            except Exception as exc:
                message = f"EKSIK  SegFormer import hatasi: {exc}"
                lines.append(message)
                issues.append(message)

        lines.append("")

        if issues:
            lines.append("Durum: EKSIK VAR, pipeline baslamadan once duzelt.")
        else:
            lines.append("Durum: HAZIR")

        return lines, issues

    def refresh_pipeline_status(self):
        lines, _issues = self.build_pipeline_status(mode="full")
        self.pipeline_status_box.setPlainText("\n".join(lines))

    def start_generation_single(self):
        self.start_generation(
            conditions=[(
                self.fault_combo.currentText(),
                self.severity_combo.currentText(),
            )],
        )

    def start_generation_selected_faults(self):
        self.start_generation(conditions=self.selected_fault_conditions())

    def start_generation(self, conditions):
        values = self.validate_inputs()

        if values is None:
            return

        checkpoint, image_paths, out_dir = values

        self.show_log_tab()
        self.set_buttons_enabled(False)
        self.output_list.clear()
        self.out_preview.clear_image()

        self.log_box.append("\n--- Yeni zaman serisi üretimi ---")

        total_outputs = (len(image_paths) - 1) * len(conditions)

        self.log_box.append(f"Seçilen clean frame sayısı: {len(image_paths)}")
        self.log_box.append(
            "Koşullar: " + ", ".join(f"{fault}_{severity}" for fault, severity in conditions)
        )
        self.log_box.append(f"Toplam üretilecek çıktı sayısı: {total_outputs}")
        self.log_box.append(f"Üretim çözünürlüğü: {self.selected_image_size()} px")

        QApplication.processEvents()

        self.worker = InferenceWorker(
            checkpoint=checkpoint,
            image_paths=image_paths,
            output_dir=out_dir,
            fault=self.fault_combo.currentText(),
            severity=self.severity_combo.currentText(),
            all_conditions=False,
            conditions=conditions,
            image_size=self.selected_image_size(),
        )

        self.worker.log.connect(self.log_box.append)
        self.worker.finished_ok.connect(self.on_finished)
        self.worker.failed.connect(self.on_failed)

        self.worker.start()

    def start_full_pipeline_single(self):
        self.append_log("\n--- Tam Pipeline butonuna basildi: Tek Kosul ---")
        self.start_full_pipeline(
            conditions=[(
                self.fault_combo.currentText(),
                self.severity_combo.currentText(),
            )],
        )

    def start_full_pipeline_selected_faults(self):
        self.append_log("\n--- Tam Pipeline butonuna basildi: 3 Hata ---")
        self.start_full_pipeline(conditions=self.selected_fault_conditions())

    def start_full_pipeline(self, conditions):
        self.append_log("Pipeline girdileri kontrol ediliyor...")
        values = self.validate_inputs()
        detector_dir = self.validate_detector_dir()

        if values is None or detector_dir is None:
            self.append_log("Pipeline durduruldu: gerekli girdi/model dosyalarindan biri eksik.")
            return

        checkpoint, image_paths, out_dir = values
        self.append_log("Temel dosya kontrolleri tamam. Pipeline worker baslatiliyor...")

        self.output_list.clear()
        self.out_preview.clear_image()
        self.log_box.append("\n--- Pipeline kuyruğa alındı ---")

        self.show_log_tab()
        QApplication.processEvents()

        self.pipeline_worker = PipelineWorker(
            mode="full",
            detector_dir=detector_dir,
            clean_image_paths=image_paths,
            checkpoint=checkpoint,
            output_dir=out_dir,
            fault=self.fault_combo.currentText(),
            severity=self.severity_combo.currentText(),
            all_conditions=False,
            conditions=conditions,
            image_size=self.selected_image_size(),
        )

        self.start_pipeline_worker(self.pipeline_worker)

    def start_prepare_detector(self):
        self.append_log("\n--- Veri Setini Hazirla butonuna basildi ---")
        detector_dir = self.validate_detector_dir()

        if detector_dir is None:
            return

        if len(self.selected_images) < 2:
            QMessageBox.warning(
                self,
                "Eksik dosya",
                "Detector veri seti için clean frame listesini seçmelisin.",
            )
            return

        if not self.output_paths:
            QMessageBox.warning(
                self,
                "Eksik çıktı",
                "Önce GAN çıktısı üretmelisin.",
            )
            return

        self.pipeline_worker = PipelineWorker(
            mode="prepare",
            detector_dir=detector_dir,
            clean_image_paths=self.selected_images,
            generated_image_paths=self.output_paths,
        )

        self.start_pipeline_worker(self.pipeline_worker)

    def start_upscale(self):
        self.append_log("\n--- Upscale butonuna basildi ---")
        detector_dir = self.validate_detector_dir()

        if detector_dir is None:
            return

        lines, issues = self.build_pipeline_status(mode="upscale")
        self.pipeline_status_box.setPlainText("\n".join(lines))

        if issues:
            QMessageBox.warning(self, "Upscale eksikleri", "\n".join(issues))
            return

        self.show_log_tab()
        QApplication.processEvents()

        self.pipeline_worker = PipelineWorker(
            mode="upscale",
            detector_dir=detector_dir,
            clean_image_paths=self.selected_images,
        )

        self.start_pipeline_worker(self.pipeline_worker)

    def start_yolo(self):
        self.append_log("\n--- YOLO butonuna basildi ---")
        detector_dir = self.validate_detector_dir()

        if detector_dir is None:
            return

        lines, issues = self.build_pipeline_status(mode="yolo")
        self.pipeline_status_box.setPlainText("\n".join(lines))

        if issues:
            QMessageBox.warning(self, "YOLO eksikleri", "\n".join(issues))
            return

        self.pipeline_worker = PipelineWorker(
            mode="yolo",
            detector_dir=detector_dir,
            clean_image_paths=self.selected_images,
        )

        self.start_pipeline_worker(self.pipeline_worker)

    def start_segmentation(self):
        self.append_log("\n--- Segmentasyon butonuna basildi ---")
        detector_dir = self.validate_detector_dir()

        if detector_dir is None:
            return

        lines, issues = self.build_pipeline_status(mode="segmentation")
        self.pipeline_status_box.setPlainText("\n".join(lines))

        if issues:
            QMessageBox.warning(self, "Segmentasyon eksikleri", "\n".join(issues))
            return

        self.pipeline_worker = PipelineWorker(
            mode="segmentation",
            detector_dir=detector_dir,
            clean_image_paths=self.selected_images,
        )

        self.start_pipeline_worker(self.pipeline_worker)

    def start_pipeline_worker(self, worker):
        self.show_log_tab()
        self.set_buttons_enabled(False)
        debug_log(f"[Goruntu] Pipeline worker start requested: {worker.mode}")
        self.log_box.append(f"\n--- Pipeline adimi baslatildi: {worker.mode} ---")
        QApplication.processEvents()
        worker.log.connect(self.log_box.append)
        worker.finished_ok.connect(self.on_pipeline_finished)
        worker.failed.connect(self.on_failed)
        worker.start()

    def on_pipeline_finished(self, result):
        self.set_buttons_enabled(True)

        generated_paths = result.get("generated_paths")

        if generated_paths:
            self.output_paths = generated_paths
            self.output_list.clear()

            for path in generated_paths:
                self.output_list.addItem(path)

            self.output_list.setCurrentRow(0)
            self.show_image(generated_paths[0], self.out_preview)

        self.log_box.append("Pipeline adımı tamamlandı.")

        self.refresh_pipeline_outputs()

        self.switch_main_page(4)

    def on_finished(self, paths):
        self.set_buttons_enabled(True)

        self.output_paths = paths
        self.output_list.clear()

        for path in paths:
            self.output_list.addItem(path)

        if paths:
            self.output_list.setCurrentRow(0)
            self.show_image(paths[0], self.out_preview)

        self.log_box.append("İşlem tamamlandı.")

        self.refresh_pipeline_outputs()

    def on_failed(self, error_text):
        self.set_buttons_enabled(True)

        self.show_log_tab()
        self.log_box.append(error_text)

        QMessageBox.critical(
            self,
            "Hata",
            error_text[-3000:],
        )

    def set_buttons_enabled(self, enabled):
        for btn in self._action_buttons:
            btn.setEnabled(enabled)

    def show_image(self, path, label):
        try:
            preview_path = build_safe_preview(path)
        except Exception as exc:
            if isinstance(label, ImagePreview):
                label.clear()
                label.setText(f"Görüntü açılamadı\n{exc}")
                label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            else:
                label.setText(f"Görüntü açılamadı\n{exc}")
            return

        if isinstance(label, ImagePreview):
            label.set_image(preview_path)
            return

        pixmap = QPixmap(str(preview_path))

        if pixmap.isNull():
            label.setText("Görüntü açılamadı")
            return

        label.setPixmap(
            pixmap.scaled(
                max(1, label.width() - 12),
                max(1, label.height() - 12),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

        label.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def preview_selected_output(self):
        items = self.output_list.selectedItems()

        if not items:
            return

        self.show_image(items[0].text(), self.out_preview)

    def set_artifact_group(self, group_name):
        self.active_artifact_group = group_name
        self.outputs_filter_btn.setChecked(group_name == "outputs")
        self.results_filter_btn.setChecked(group_name == "results")
        self.refresh_pipeline_outputs()

    def refresh_pipeline_outputs(self):
        output_roots = [
            DEFAULT_OUTPUTS_ROOT / "gan_generated",
            DEFAULT_OUTPUTS_ROOT / "gan_upscaled",
            DEFAULT_OUTPUTS_ROOT / "segmentation_outputs",
            DEFAULT_OUTPUTS_ROOT / "segmentation_comparisons",
            DEFAULT_OUTPUTS_ROOT / "yolo_comparisons",
        ]
        result_roots = [
            DEFAULT_RESULTS_ROOT / "yolo",
            DEFAULT_RESULTS_ROOT / "segmentation",
        ]

        if self.active_artifact_group == "results":
            roots = result_roots
            group_label = "Results"
        else:
            roots = output_roots
            group_label = "Outputs"
            self.active_artifact_group = "outputs"

        artifact_paths = []

        for root in roots:
            if not root.exists():
                continue

            for path in root.rglob("*"):
                if not path.is_file():
                    continue

                suffix = path.suffix.lower()

                if suffix in IMAGE_SUFFIXES or suffix in TEXT_PREVIEW_SUFFIXES:
                    artifact_paths.append(path)

        def artifact_sort_key(path):
            try:
                parent = str(path.parent.relative_to(PROJECT_ROOT))
            except ValueError:
                parent = str(path.parent)

            return parent, path.name

        artifact_paths = sorted(artifact_paths, key=artifact_sort_key)

        self.artifact_list.clear()
        self.artifact_preview.clear_image()
        self.artifact_text_preview.clear()
        self.artifact_table_preview.clear()
        self.artifact_table_preview.setVisible(False)
        self.artifact_preview.setVisible(True)

        for path in artifact_paths:
            try:
                display_path = path.relative_to(PROJECT_ROOT)
            except ValueError:
                display_path = path

            self.artifact_list.addItem(str(display_path))

        if artifact_paths:
            self.artifact_list.setCurrentRow(0)

        self.log_box.append(f"{group_label} listesi yenilendi: {len(artifact_paths)} dosya.")

    def selected_artifact_path(self):
        items = self.artifact_list.selectedItems()

        if not items:
            return None

        path = Path(items[0].text())

        if not path.is_absolute():
            path = PROJECT_ROOT / path

        return path

    def preview_selected_artifact(self):
        path = self.selected_artifact_path()

        if path is None:
            return

        if not path.exists():
            self.artifact_preview.clear_image()
            self.artifact_text_preview.setPlainText("Dosya bulunamadi.")
            return

        suffix = path.suffix.lower()
        self.artifact_text_preview.clear()

        if suffix in IMAGE_SUFFIXES:
            self.artifact_table_preview.setVisible(False)
            self.artifact_preview.setVisible(True)
            self.show_image(path, self.artifact_preview)
            self.artifact_text_preview.setPlainText(str(path))
            return

        self.artifact_preview.clear_image()

        if suffix in TEXT_PREVIEW_SUFFIXES:
            if suffix == ".csv":
                self.artifact_preview.setVisible(False)
                self.artifact_table_preview.setVisible(True)
                self.populate_csv_table(path)
                self.artifact_text_preview.setPlainText(self.format_csv_preview(path))
                return

            self.artifact_table_preview.setVisible(False)
            self.artifact_preview.setVisible(True)
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                text = f"Dosya okunamadi:\n{exc}"

            lines = text.splitlines()

            if len(lines) > 160:
                text = "\n".join(lines[:160]) + "\n\n... sadece ilk 160 satir gosteriliyor."

            self.artifact_text_preview.setPlainText(text)
            return

        self.artifact_text_preview.setPlainText(str(path))

    def populate_csv_table(self, path):
        try:
            import pandas as pd

            df = pd.read_csv(path)
        except Exception:
            self.artifact_table_preview.clear()
            self.artifact_table_preview.setRowCount(0)
            self.artifact_table_preview.setColumnCount(0)
            return

        preview = df.head(60).copy()

        for column in preview.columns:
            if pd.api.types.is_float_dtype(preview[column]):
                preview[column] = preview[column].round(4)

        self.artifact_table_preview.clear()
        self.artifact_table_preview.setRowCount(len(preview))
        self.artifact_table_preview.setColumnCount(len(preview.columns))
        self.artifact_table_preview.setVisible(True)
        self.artifact_table_preview.setHorizontalHeaderLabels([str(column) for column in preview.columns])

        for row_index, (_idx, row) in enumerate(preview.iterrows()):
            for col_index, value in enumerate(row):
                item = QTableWidgetItem("" if pd.isna(value) else str(value))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.artifact_table_preview.setItem(row_index, col_index, item)

        self.artifact_table_preview.resizeRowsToContents()

    def format_csv_preview(self, path):
        try:
            import pandas as pd

            df = pd.read_csv(path)
        except Exception as exc:
            return f"CSV okunamadi:\n{exc}"

        lines = []
        lines.append("Sutun Aciklamalari")
        for column in df.columns:
            description = CSV_COLUMN_DESCRIPTIONS.get(column, "Bu CSV icindeki ek metrik/alan.")
            lines.append(f"- {column}: {description}")

        return "\n".join(lines)

    def open_selected_artifact(self):
        path = self.selected_artifact_path()

        if path is None or not path.exists():
            return

        if sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        elif sys.platform.startswith("win"):
            os.startfile(path)
        else:
            subprocess.Popen(["xdg-open", str(path)])

    def open_selected_artifact_folder(self):
        path = self.selected_artifact_path()

        if path is None:
            return

        folder = path.parent if path.is_file() else path
        self.open_folder(folder)

    def clear_outputs(self):
        self.output_paths = []
        self.selected_images = []

        self.output_list.clear()
        self.input_list.clear()
        self.artifact_list.clear()

        self.out_preview.clear_image()
        self.curr_preview.clear_image()
        self.artifact_preview.clear_image()
        self.artifact_text_preview.clear()
        self.artifact_table_preview.clear()
        self.artifact_table_preview.setVisible(False)
        self.artifact_preview.setVisible(True)

        self.log_box.append("Listeler temizlendi.")

    def open_output_folder(self):
        out_dir = self.output_edit.text().strip() or DEFAULT_OUTPUT_DIR
        self.open_folder(out_dir)

    def open_folder(self, folder):
        folder = str(folder)
        Path(folder).mkdir(parents=True, exist_ok=True)

        if sys.platform == "darwin":
            subprocess.Popen(["open", folder])
        elif sys.platform.startswith("win"):
            os.startfile(folder)
        else:
            subprocess.Popen(["xdg-open", folder])


def main():
    debug_log("[Görüntü] Qt arayüz başlatılıyor...")
    os.environ.setdefault("QT_MAC_WANTS_LAYER", "1")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    app = QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)
    app.setApplicationDisplayName(APP_TITLE)
    app.setOrganizationName("SentetikVeriPlatformu")
    app.setQuitOnLastWindowClosed(True)

    window = RCGANQtApp()
    window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
    force_window_visible(window)
    QTimer.singleShot(350, lambda: force_window_visible(window))
    debug_log("[Görüntü] Pencere gösterildi.")

    exit_code = app.exec()
    debug_log(f"[Görüntü] Uygulama kapandı. Exit code: {exit_code}")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
