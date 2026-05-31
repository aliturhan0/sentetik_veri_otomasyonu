import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import requests
import webview


os.environ.setdefault("LC_ALL", "en_US.UTF-8")
os.environ.setdefault("LANG", "en_US.UTF-8")

APP_DIR = Path(__file__).resolve().parent
BACKEND_DIR = APP_DIR / "backend"
INDEX_PATH = BACKEND_DIR / "index.html"
OUTPUT_DIR = APP_DIR / "outputs"
HOST = os.getenv("SENTETIK_HOST", "127.0.0.1")
PORT = int(os.getenv("SENTETIK_PORT", "8000"))
BASE_URL = f"http://{HOST}:{PORT}"


def api_ready():
    try:
        response = requests.get(f"{BASE_URL}/api/system_status", timeout=0.8)
        return response.status_code == 200
    except requests.RequestException:
        return False


def port_is_listening(host, port):
    try:
        with socket.create_connection((host, port), timeout=0.25):
            return True
    except OSError:
        return False


def load_desktop_html():
    html = INDEX_PATH.read_text(encoding="utf-8")
    style = (BACKEND_DIR / "style.css").read_text(encoding="utf-8")
    script = (BACKEND_DIR / "script.js").read_text(encoding="utf-8")
    html = html.replace('<link rel="stylesheet" href="style.css">', f"<style>\n{style}\n</style>")
    html = html.replace(
        '<script src="script.js?v=utility-protocol-v1"></script>',
        f"<script>\n{script}\n</script>",
    )
    return html


class DesktopApi:
    def save_distilled_csv(self):
        return self._save_output("distilled_data.csv", "distilled_clean_data.csv")

    def save_generated_csv(self):
        return self._save_output("live_synthetic_output.csv", "sentetik_veri.csv")

    def _save_output(self, source_name, default_name):
        source = OUTPUT_DIR / source_name
        if not source.exists():
            return {"ok": False, "message": f"Çıktı dosyası henüz yok: {source.name}"}

        result = webview.windows[0].create_file_dialog(
            webview.SAVE_DIALOG,
            save_filename=default_name,
        )
        if not result:
            return {"ok": False, "cancelled": True}

        target = Path(result)
        shutil.copy2(source, target)
        return {"ok": True, "path": str(target)}


class DataApp:
    def __init__(self):
        self.backend_process = None
        self.backend_log = None

    def start_backend_if_needed(self):
        if api_ready():
            print("[DataUI] Backend zaten hazır.", flush=True)
            return

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.backend_log = open(OUTPUT_DIR / "backend_runtime.log", "a", encoding="utf-8")

        env = os.environ.copy()
        env["SENTETIK_HOST"] = HOST
        env["SENTETIK_PORT"] = str(PORT)
        env["PYTHONUNBUFFERED"] = "1"
        env["LC_ALL"] = "en_US.UTF-8"
        env["LANG"] = "en_US.UTF-8"

        try:
            self.backend_process = subprocess.Popen(
                [sys.executable, str(BACKEND_DIR / "server.py")],
                cwd=str(APP_DIR),
                env=env,
                stdout=self.backend_log,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            print(f"[DataUI] Backend ayrı süreçte başlatıldı. PID: {self.backend_process.pid}", flush=True)
            
            # Wait for backend to be ready
            import time
            max_retries = 20
            for i in range(max_retries):
                if api_ready():
                    print("[DataUI] Backend başarıyla ayağa kalktı.", flush=True)
                    break
                time.sleep(0.5)
            else:
                print("[DataUI] UYARI: Backend belirtilen sürede yanıt vermedi, arayüz bağlantı hatası verebilir.", flush=True)
                
        except Exception as e:
            print(f"[DataUI] KRİTİK HATA: Backend başlatılamadı! Detay: {e}", flush=True)
            import traceback
            traceback.print_exc()

    def stop_backend(self):
        if self.backend_process and self.backend_process.poll() is None:
            self.backend_process.terminate()
            try:
                self.backend_process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.backend_process.kill()
        if self.backend_log:
            self.backend_log.close()


def main():
    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"Eski arayüz bulunamadı: {INDEX_PATH}")

    print("=" * 60, flush=True)
    print("Akıllı Sentetik Veri Artırımı hızlı masaüstü arayüzle başlatılıyor...", flush=True)
    print("=" * 60, flush=True)

    app = DataApp()
    app.start_backend_if_needed()

    window = webview.create_window(
        "Sentetik Veri Üretim Hattı",
        html=load_desktop_html(),
        js_api=DesktopApi(),
        width=1440,
        height=920,
        min_size=(1100, 720),
    )
    window.events.closed += app.stop_backend
    webview.start(gui="cocoa", debug=False)


if __name__ == "__main__":
    main()
