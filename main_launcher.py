import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from PySide6.QtCore import QProcess, QProcessEnvironment, Qt
from PySide6.QtWidgets import (
    QApplication,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_APP_DIR = PROJECT_ROOT / "rcgan_qt_gui_app_v1"
IMAGE_APP_SCRIPT = IMAGE_APP_DIR / "qt_gui_app_updated.py"
IMAGE_APP_LOG = IMAGE_APP_DIR / "image_runtime.log"
IMAGE_APP_VENV = IMAGE_APP_DIR / "qtvenv"

DATA_APP_DIR = PROJECT_ROOT / "akilli_veri_arttirimi"
DATA_APP_SCRIPT = DATA_APP_DIR / "main.py"
DATA_APP_VENV = DATA_APP_DIR / "otonom_env"


class MainLauncher(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Sentetik Veri Platformu")
        self.resize(1040, 680)

        self.processes = []

        self._build_ui()
        self._apply_styles()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(28, 28, 28, 22)
        root.setSpacing(14)

        title = QLabel("Sentetik Veri Platformu")
        title.setObjectName("title")
        root.addWidget(title)

        subtitle = QLabel("Çalışmak istediğin pipeline'ı seç.")
        subtitle.setObjectName("subtitle")
        root.addWidget(subtitle)

        status = QLabel("● Aktif modüller hazır")
        status.setObjectName("statusIndicator")
        root.addWidget(status)

        cards = QHBoxLayout()
        cards.setSpacing(16)

        image_card = self._make_card(
            title="Görüntü Robustness Pipeline",
            body=(
                "Clean kamera frame'lerinden RCGAN ile bozulmuş görüntü üretir, "
                "EDSR ile upscale eder, YOLO ve SegFormer ile dayanıklılık analizi yapar."
            ),
            primary_text="Görüntü Modelini Aç",
            primary_action=self.open_image_app,
            secondary_text="Görüntü Klasörünü Aç",
            secondary_action=lambda: self.open_folder(IMAGE_APP_DIR),
        )

        data_card = self._make_card(
            title="Akıllı Veri Artırımı",
            body=(
                "CSV/tabular/yörünge verisini damıtır, veri tipine göre RCGAN, "
                "CTGAN veya SMOTE ile sentetik veri üretir ve utility/fidelity raporlar."
            ),
            primary_text="Veri Artırımı Modelini Aç",
            primary_action=self.open_data_app,
            secondary_text="Veri Artırımı Klasörünü Aç",
            secondary_action=lambda: self.open_folder(DATA_APP_DIR),
        )

        cards.addWidget(image_card, 1)
        cards.addWidget(data_card, 1)
        root.addLayout(cards, 2)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setPlaceholderText("Başlatılan uygulamaların durumları burada görünecek...")
        root.addWidget(self.log_box, 1)

        footer = QLabel(
            "Not: İki uygulama ayrı süreç olarak açılır. Ana ekranı kapatmadan ikisini de çalıştırabilirsin."
        )
        footer.setObjectName("hint")
        footer.setWordWrap(True)
        root.addWidget(footer)

    def _make_card(
        self,
        title,
        body,
        primary_text,
        primary_action,
        secondary_text,
        secondary_action,
    ):
        group = QGroupBox(title)
        layout = QVBoxLayout(group)
        layout.setSpacing(12)

        text = QLabel(body)
        text.setWordWrap(True)
        text.setObjectName("cardBody")
        layout.addWidget(text)
        layout.addStretch(1)

        primary = QPushButton(primary_text)
        primary.setObjectName("primaryButton")
        primary.clicked.connect(primary_action)
        layout.addWidget(primary)

        secondary = QPushButton(secondary_text)
        secondary.clicked.connect(secondary_action)
        layout.addWidget(secondary)

        return group

    def _apply_styles(self):
        self.setStyleSheet("""
            /* ── Base ── */
            QWidget {
                font-family: 'Inter', 'SF Pro Display', 'Segoe UI', sans-serif;
                font-size: 14px;
                color: #e8ecf4;
                background: #06080f;
            }

            /* ── Title ── */
            QLabel#title {
                font-size: 30px;
                font-weight: 800;
                color: #ffffff;
                background: transparent;
                padding-bottom: 0px;
            }

            /* ── Subtitle ── */
            QLabel#subtitle {
                font-size: 15px;
                color: #a0aec0;
                background: transparent;
                padding-bottom: 2px;
            }

            /* ── Status Indicator ── */
            QLabel#statusIndicator {
                font-size: 13px;
                color: #10b981;
                background: transparent;
                padding-bottom: 4px;
            }

            /* ── Card Body Text ── */
            QLabel#cardBody {
                color: #a0aec0;
                font-size: 13px;
                line-height: 1.45;
                background: transparent;
            }

            /* ── Hint / Footer ── */
            QLabel#hint {
                color: #5a6578;
                font-size: 13px;
                padding: 10px 14px;
                background: #0c1017;
                border: 1px solid rgba(255, 255, 255, 0.06);
                border-radius: 8px;
            }

            /* ── Glassmorphism Cards ── */
            QGroupBox {
                font-size: 16px;
                font-weight: 700;
                margin-top: 14px;
                padding: 20px 16px 16px 16px;
                border: 1px solid rgba(255, 255, 255, 0.06);
                border-radius: 14px;
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(17,22,32,0.95),
                    stop:1 rgba(24,30,42,0.90)
                );
            }
            QGroupBox:hover {
                border: 1px solid rgba(99, 102, 241, 0.25);
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 14px;
                padding: 0 8px;
                color: #e8ecf4;
            }

            /* ── Secondary Buttons ── */
            QPushButton {
                padding: 11px 16px;
                border-radius: 10px;
                border: 1px solid rgba(255, 255, 255, 0.06);
                background: #111620;
                color: #a0aec0;
                font-weight: 500;
            }
            QPushButton:hover {
                background: #181e2a;
                border-color: rgba(99, 102, 241, 0.3);
                color: #e8ecf4;
            }
            QPushButton:pressed {
                background: #0c1017;
            }

            /* ── Primary (Gradient) Buttons ── */
            QPushButton#primaryButton {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #6366f1, stop:1 #8b5cf6
                );
                color: #ffffff;
                border: 1px solid rgba(99, 102, 241, 0.4);
                font-weight: 700;
            }
            QPushButton#primaryButton:hover {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #818cf8, stop:1 #a78bfa
                );
                border-color: rgba(139, 92, 246, 0.6);
            }
            QPushButton#primaryButton:pressed {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4f46e5, stop:1 #7c3aed
                );
            }

            /* ── Terminal Log Area ── */
            QTextEdit {
                font-family: 'JetBrains Mono', 'Fira Code', 'Menlo', monospace;
                font-size: 12px;
                color: #22d3ee;
                background: #06080f;
                border: 1px solid rgba(255, 255, 255, 0.06);
                border-radius: 10px;
                padding: 8px;
                selection-background-color: rgba(99, 102, 241, 0.3);
                selection-color: #e8ecf4;
            }

            /* ── Scrollbar (thin, rounded, cyan) ── */
            QScrollBar:vertical {
                width: 6px;
                background: transparent;
                margin: 4px 0;
            }
            QScrollBar::handle:vertical {
                background: rgba(34, 211, 238, 0.35);
                border-radius: 3px;
                min-height: 30px;
            }
            QScrollBar::handle:vertical:hover {
                background: rgba(34, 211, 238, 0.55);
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical,
            QScrollBar::add-page:vertical,
            QScrollBar::sub-page:vertical {
                height: 0px;
                background: transparent;
            }
            QScrollBar:horizontal {
                height: 6px;
                background: transparent;
                margin: 0 4px;
            }
            QScrollBar::handle:horizontal {
                background: rgba(34, 211, 238, 0.35);
                border-radius: 3px;
                min-width: 30px;
            }
            QScrollBar::handle:horizontal:hover {
                background: rgba(34, 211, 238, 0.55);
            }
            QScrollBar::add-line:horizontal,
            QScrollBar::sub-line:horizontal,
            QScrollBar::add-page:horizontal,
            QScrollBar::sub-page:horizontal {
                width: 0px;
                background: transparent;
            }

            /* ── Message Boxes ── */
            QMessageBox {
                background: #111620;
            }
            QMessageBox QLabel {
                color: #e8ecf4;
                background: transparent;
            }
            QMessageBox QPushButton {
                min-width: 80px;
            }
        """)

    def open_image_app(self):
        # Görüntü arayüzü torch/torchvision'u lazy yükler. Venv içinde ağır import
        # kontrolü UI thread'ini kilitleyebildiği için arayüzü launcher'ın çalışan
        # Python'u ile açıyoruz; model bağımlılıkları üretim anında raporlanır.
        python = sys.executable
        missing = self.missing_modules(
            python=python,
            modules=["PySide6", "PIL"],
        )

        if missing:
            message = (
                "Görüntü robustness uygulaması için Python ortamı hazır değil.\n\n"
                f"Kullanılan Python:\n{python}\n\n"
                f"Eksik modüller: {', '.join(missing)}\n\n"
                "Kurmak için terminalde:\n"
                "pip install -r requirements.txt"
            )
            self.log(message)
            QMessageBox.warning(self, "Ortam hazır değil", message)
            return

        self.stop_stale_processes(IMAGE_APP_SCRIPT)
        self.launch_app(
            label="Görüntü Robustness Pipeline",
            script=IMAGE_APP_SCRIPT,
            cwd=IMAGE_APP_DIR,
            python=python,
            log_path=IMAGE_APP_LOG,
        )

    def open_data_app(self):
        # Bu app tarafında bazı paketler ilk importta birkaç saniye bekletebiliyor.
        # Preflight import kontrolü yanlış "eksik modül" alarmı verdiği için direkt
        # süreci başlatıyoruz; gerçek hata olursa log alanına süreç çıktısı düşer.
        python = self.responsive_python(DATA_APP_VENV)

        self.launch_app(
            label="Akıllı Veri Artırımı",
            script=DATA_APP_SCRIPT,
            cwd=DATA_APP_DIR,
            python=python,
        )

    def preferred_python(self, venv_dir):
        venv_dir = Path(venv_dir)
        candidate = (
            venv_dir / "Scripts" / "python.exe"
            if sys.platform.startswith("win")
            else venv_dir / "bin" / "python"
        )
        return str(candidate) if candidate.exists() else sys.executable

    def responsive_python(self, venv_dir):
        candidate = self.preferred_python(venv_dir)
        if candidate == sys.executable:
            return candidate
        try:
            result = subprocess.run(
                [candidate, "-c", "import sys; print(sys.executable)"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=3,
                check=False,
            )
            if result.returncode == 0:
                return candidate
        except (OSError, subprocess.TimeoutExpired):
            pass
        self.log(f"Venv Python yanıt vermiyor, sistem Python kullanılacak: {candidate}")
        return sys.executable

    def missing_modules(self, python, modules):
        missing = []

        for module in modules:
            try:
                result = subprocess.run(
                    [python, "-c", f"import {module}"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    timeout=4,
                )
            except subprocess.TimeoutExpired:
                missing.append(f"{module} (kontrol zaman aşımı)")
                continue

            if result.returncode != 0:
                missing.append(module)

        return missing

    def stop_stale_processes(self, script):
        script = str(Path(script).resolve())
        current_pid = os.getpid()
        try:
            result = subprocess.run(
                ["pgrep", "-f", script],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                check=False,
            )
        except OSError:
            return

        stale_pids = []
        for raw_pid in result.stdout.splitlines():
            try:
                pid = int(raw_pid.strip())
            except ValueError:
                continue
            if pid and pid != current_pid:
                stale_pids.append(pid)

        if not stale_pids:
            return

        self.log(f"Eski görüntü süreci temizleniyor: {', '.join(map(str, stale_pids))}")
        for pid in stale_pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass

        time.sleep(0.7)

        for pid in stale_pids:
            try:
                os.kill(pid, 0)
            except OSError:
                continue
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass

    def launch_app(self, label, script, cwd, python, log_path=None):
        if not script.exists():
            QMessageBox.warning(self, "Dosya bulunamadı", f"Uygulama dosyası yok:\n{script}")
            return

        process = QProcess(self)
        process.setProgram(python)
        process.setArguments([str(script)])
        process.setWorkingDirectory(str(cwd))
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("LANG", "en_US.UTF-8")
        environment.insert("LC_ALL", "en_US.UTF-8")
        environment.insert("PYTHONUNBUFFERED", "1")
        environment.insert("QT_MAC_WANTS_LAYER", "1")
        environment.insert("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        if log_path:
            environment.insert("SENTETIK_IMAGE_LOG", str(log_path))
        process.setProcessEnvironment(environment)

        process.readyReadStandardOutput.connect(
            lambda proc=process, name=label: self.read_process_output(name, proc)
        )
        process.errorOccurred.connect(
            lambda error, name=label: self.log(f"{name} süreç hatası: {error}")
        )
        process.finished.connect(
            lambda code, status, name=label: self.log(
                f"{name} kapandı. Exit code: {code}, status: {status}"
            )
        )

        process.start()

        if not process.waitForStarted(3000):
            QMessageBox.critical(self, "Başlatılamadı", f"{label} başlatılamadı.")
            return

        self.processes.append(process)
        self.log(f"{label} başlatıldı. PID: {process.processId()}")
        if log_path:
            self.log(f"{label} log dosyası: {log_path}")

    def read_process_output(self, label, process):
        text = bytes(process.readAllStandardOutput()).decode("utf-8", errors="replace")

        if text.strip():
            self.log(f"[{label}] {text.rstrip()}")

    def open_folder(self, folder):
        folder = Path(folder)

        if not folder.exists():
            QMessageBox.warning(self, "Klasör bulunamadı", str(folder))
            return

        if sys.platform == "darwin":
            subprocess.Popen(["open", str(folder)])
        elif sys.platform.startswith("win"):
            os.startfile(folder)
        else:
            subprocess.Popen(["xdg-open", str(folder)])

    def log(self, message):
        self.log_box.append(str(message))


def main():
    app = QApplication(sys.argv)
    window = MainLauncher()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
