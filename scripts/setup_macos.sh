#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAIN_ENV="$ROOT_DIR/env"
IMAGE_ENV_LINK="$ROOT_DIR/.venv311"
DATA_APP_DIR="$ROOT_DIR/akilli_veri_arttirimi"
DATA_ENV="$DATA_APP_DIR/otonom_env"

log() {
  printf "\n[%s] %s\n" "$(date '+%H:%M:%S')" "$*"
}

fail() {
  printf "\nHATA: %s\n" "$*" >&2
  exit 1
}

python_is_supported() {
  "$1" - <<'PY'
import sys
raise SystemExit(0 if (3, 10) <= sys.version_info[:2] <= (3, 12) else 1)
PY
}

python_version() {
  "$1" - <<'PY'
import sys
print(".".join(map(str, sys.version_info[:3])))
PY
}

pick_python() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    command -v "$PYTHON_BIN" >/dev/null 2>&1 || fail "PYTHON_BIN bulunamadi: $PYTHON_BIN"
    python_is_supported "$PYTHON_BIN" || fail "PYTHON_BIN Python 3.10, 3.11 veya 3.12 olmali. Bulunan: $("$PYTHON_BIN" --version)"
    printf "%s\n" "$PYTHON_BIN"
    return
  fi

  local candidates=(python3.12 python3.11 python3.10 python)
  local candidate
  for candidate in "${candidates[@]}"; do
    if command -v "$candidate" >/dev/null 2>&1 && python_is_supported "$candidate"; then
      command -v "$candidate"
      return
    fi
  done

  fail "Uygun Python bulunamadi. Python 3.10, 3.11 veya 3.12 kurup tekrar calistirin. Python 3.14 bu proje icin onerilmez."
}

create_or_reuse_venv() {
  local python_bin="$1"
  local env_dir="$2"

  if [[ -x "$env_dir/bin/python" ]]; then
    log "Mevcut ortam kullaniliyor: $env_dir ($("$env_dir/bin/python" --version))"
    return
  fi

  log "Sanal ortam olusturuluyor: $env_dir"
  "$python_bin" -m venv "$env_dir"
}

install_requirements() {
  local env_dir="$1"
  local requirements_file="$2"

  log "pip guncelleniyor: $env_dir"
  "$env_dir/bin/python" -m pip install --upgrade pip

  log "Paketler kuruluyor: $requirements_file"
  "$env_dir/bin/python" -m pip install -r "$requirements_file"
}

verify_main_env() {
  log "Ana ortam kontrol ediliyor"
  "$MAIN_ENV/bin/python" - <<'PY'
from six.moves import _thread  # noqa: F401
from ultralytics import YOLO
import cv2
import pandas
import PySide6
import torch
import transformers

print("Python OK")
print("ultralytics:", __import__("ultralytics").__version__)
has_superres = hasattr(cv2, "dnn_superres") and hasattr(cv2.dnn_superres, "DnnSuperResImpl_create")
print("opencv dnn_superres:", has_superres)
print("pandas:", pandas.__version__)
print("PySide6:", PySide6.__version__)
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
PY
}

verify_data_env() {
  log "Akilli Veri Artirimi ortami kontrol ediliyor"
  "$DATA_ENV/bin/python" - <<'PY'
import ctgan
import fastapi
import numpy
import pandas
import sklearn
import tensorflow
import torch
import uvicorn
import webview

print("Python OK")
print("ctgan:", getattr(ctgan, "__version__", "installed"))
print("fastapi:", fastapi.__version__)
print("pandas:", pandas.__version__)
print("tensorflow:", tensorflow.__version__)
print("torch:", torch.__version__)
PY
}

if [[ "$(uname -s)" != "Darwin" ]]; then
  fail "Bu script macOS icin hazirlandi. Windows/Linux kurulumu icin README adimlarini kullanin."
fi

cd "$ROOT_DIR"

PYTHON_BIN_SELECTED="$(pick_python)"
log "Secilen Python: $PYTHON_BIN_SELECTED ($(python_version "$PYTHON_BIN_SELECTED"))"

create_or_reuse_venv "$PYTHON_BIN_SELECTED" "$MAIN_ENV"
install_requirements "$MAIN_ENV" "$ROOT_DIR/requirements.txt"

if [[ -e "$IMAGE_ENV_LINK" && ! -L "$IMAGE_ENV_LINK" && "$IMAGE_ENV_LINK" != "$MAIN_ENV" ]]; then
  log ".venv311 zaten var, dokunulmayacak: $IMAGE_ENV_LINK"
elif [[ ! -e "$IMAGE_ENV_LINK" ]]; then
  log "Launcher uyumlulugu icin .venv311 -> env baglantisi olusturuluyor"
  ln -s "env" "$IMAGE_ENV_LINK"
fi

create_or_reuse_venv "$PYTHON_BIN_SELECTED" "$DATA_ENV"
install_requirements "$DATA_ENV" "$DATA_APP_DIR/requirements.txt"

verify_main_env
verify_data_env

cat <<EOF

Kurulum tamamlandi.

Ana launcher icin:
  cd "$ROOT_DIR"
  source env/bin/activate
  python main_launcher.py

Akilli Veri Artirimi ortami:
  $DATA_ENV/bin/python

Not: Python 3.14 yerine Python 3.10, 3.11 veya 3.12 kullanildi.
EOF
