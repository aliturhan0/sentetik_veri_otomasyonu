"""
Sentetik Veri Üretim Hattı — Adaptive Akıllı Veri Artırım Platformu
Bilgi Damıtma + RCGAN/CTGAN/SMOTE Sentetik Üretim
"""
import os, io, csv, random, json, traceback, subprocess, threading
import importlib.util

CTGAN = None
CTGAN_IMPORT_ERROR = None


def is_ctgan_available(load=False):
    """CTGAN'i uygulama açılışında değil, gerektiği anda yükle."""
    global CTGAN, CTGAN_IMPORT_ERROR

    if CTGAN is not None:
        return True

    if not load:
        return importlib.util.find_spec("ctgan") is not None

    try:
        from ctgan import CTGAN as CTGANClass
        CTGAN = CTGANClass
        CTGAN_IMPORT_ERROR = None
        print('[✅] CTGAN modülü yüklendi', flush=True)
        return True
    except ImportError as exc:
        CTGAN_IMPORT_ERROR = str(exc)
        print('[⚠️] CTGAN yok, SMOTE fallback kullanılacak', flush=True)
        return False

import torch, torch.nn as nn
from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd, numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
from scipy.spatial.distance import cdist
from collections import Counter


def _env_int(name, default, min_value=None, max_value=None):
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default

    if min_value is not None:
        value = max(min_value, value)
    if max_value is not None:
        value = min(max_value, value)
    return value


def _parse_cors_origins():
    raw = os.getenv(
        "SENTETIK_CORS_ORIGINS",
        "http://127.0.0.1:8000,http://localhost:8000,null",
    )
    origins = [item.strip() for item in raw.split(",") if item.strip()]
    return origins or ["http://127.0.0.1:8000", "http://localhost:8000"]


RANDOM_SEED = _env_int("SENTETIK_RANDOM_SEED", 42)
API_HOST = os.getenv("SENTETIK_HOST", "127.0.0.1")
API_PORT = _env_int("SENTETIK_PORT", 8000, min_value=1, max_value=65535)
MAX_UPLOAD_MB = _env_int("SENTETIK_MAX_UPLOAD_MB", 250, min_value=1)
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
CTGAN_MAX_ROWS = _env_int("SENTETIK_CTGAN_MAX_ROWS", 10000, min_value=100)
CTGAN_MIN_ROWS = _env_int("SENTETIK_CTGAN_MIN_ROWS", 100, min_value=20)
CTGAN_EPOCHS_SMALL = _env_int("SENTETIK_CTGAN_EPOCHS_SMALL", 150, min_value=1)
CTGAN_EPOCHS_MEDIUM = _env_int("SENTETIK_CTGAN_EPOCHS_MEDIUM", 100, min_value=1)
CTGAN_EPOCHS_LARGE = _env_int("SENTETIK_CTGAN_EPOCHS_LARGE", 50, min_value=1)
SYNTHETIC_TARGET_F1_IMPROVEMENT = _env_int("SENTETIK_TARGET_F1_IMPROVEMENT", 15, min_value=1)
SYNTHETIC_TARGET_MINORITY_RECALL = float(os.getenv("SENTETIK_TARGET_MINORITY_RECALL", "0.80"))
SYNTHETIC_VALIDATION_MIN_KEEP_RATIO = float(os.getenv("SENTETIK_VALIDATION_MIN_KEEP_RATIO", "0.30"))
RCGAN_ANOMALY_SEVERITY = os.getenv("SENTETIK_RCGAN_ANOMALY_SEVERITY", "medium").lower().strip()
if RCGAN_ANOMALY_SEVERITY not in {"low", "medium", "high"}:
    RCGAN_ANOMALY_SEVERITY = "medium"

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


def reset_random_state(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

app = FastAPI(title="Sentetik Veri Üretim Hattı")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_cors_origins(),
    allow_origin_regex=r"^(?:https?://(?:127\.0\.0\.1|localhost)(?::\d+)?|null)$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
LATEST_ANALYSIS_PATH = os.path.join(OUTPUT_DIR, "latest_analysis.json")
SEED_PATH = os.path.join(PROJECT_DIR, "waymo_seed_MASSIVE.csv")
SEED_LEGACY_PATH = os.path.join(PROJECT_DIR, "waymo_seed.csv")
BASELINE_MODEL_PATH = os.path.join(OUTPUT_DIR, "waymo_rcgan_GODMODE_A100_STABLE.pth")
V2_MODEL_PATH = os.path.join(PROJECT_DIR, "waymo_rcgan_GODMODE_V2_PHYSICS_AWARE.pth")
V2_OUTPUT_MODEL_PATH = os.path.join(OUTPUT_DIR, "waymo_rcgan_GODMODE_V2_PHYSICS_AWARE.pth")
MODEL_SELECTION = os.getenv("SENTETIK_RCGAN_MODEL", "v2").lower().strip()
DEFAULT_MODEL_PATH = V2_MODEL_PATH if MODEL_SELECTION == "v2" else BASELINE_MODEL_PATH
MODEL_PATH = os.getenv("SENTETIK_RCGAN_MODEL_PATH", DEFAULT_MODEL_PATH)
LOCAL_MIRROR_PROJECT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(PROJECT_DIR)),
    "githubdeneme",
    "akilli_veri_arttirimi",
)
MODEL_VERSION = os.getenv(
    "SENTETIK_RCGAN_MODEL_VERSION",
    "v2" if os.path.basename(MODEL_PATH) == os.path.basename(V2_MODEL_PATH)
    else "baseline" if os.path.basename(MODEL_PATH) == os.path.basename(BASELINE_MODEL_PATH)
    else "custom",
)

_seed_row_cache = {}
_seed_count_pending = set()
_seed_count_lock = threading.Lock()

def safe_write_csv(df, path, **kwargs):
    """CSV çıktısını önce geçici dosyaya yazıp sonra atomik olarak değiştirir."""
    tmp_path = f"{path}.tmp"
    try:
        df.to_csv(tmp_path, **kwargs)
        os.replace(tmp_path, path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass


def safe_write_json(data, path):
    tmp_path = f"{path}.tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as output:
            json.dump(data, output, ensure_ascii=False)
        os.replace(tmp_path, path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass

def is_git_lfs_pointer(path):
    """Detect Git LFS pointer files that have not been pulled yet."""
    try:
        if not path or not os.path.isfile(path) or os.path.getsize(path) > 1024:
            return False

        with open(path, "rb") as f:
            head = f.read(128)

        return head.startswith(b"version https://git-lfs.github.com/spec/v1")
    except OSError:
        return False


def is_dataless_file(path):
    """macOS iCloud placeholder dosyalarını okumadan belirle."""
    if not path or not os.path.exists(path):
        return False
    try:
        result = subprocess.run(
            ["/bin/ls", "-lO", path],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
            check=False,
        )
        return "dataless" in result.stdout
    except (OSError, subprocess.TimeoutExpired):
        return False


def get_local_asset_path(path):
    """Placeholder varlıklarda aynı yapıya sahip yerel çalışma kopyasını kullan."""
    candidates = [path]
    try:
        relative = os.path.relpath(path, PROJECT_DIR)
        if not relative.startswith(".."):
            candidates.append(os.path.join(LOCAL_MIRROR_PROJECT_DIR, relative))
    except ValueError:
        pass

    for candidate in candidates:
        try:
            if not os.path.isfile(candidate) or is_dataless_file(candidate):
                continue
            if os.path.getsize(candidate) <= 0 or is_git_lfs_pointer(candidate):
                continue
            return candidate
        except OSError:
            continue
    return None


def get_seed_path():
    """Return the usable seed CSV, preferring the real repo file over legacy symlinks."""
    candidates = [SEED_PATH, SEED_LEGACY_PATH]
    for path in candidates:
        resolved = get_local_asset_path(path)
        if resolved:
            return resolved
    return None


def get_v2_model_path():
    for path in (V2_MODEL_PATH, V2_OUTPUT_MODEL_PATH):
        resolved = get_local_asset_path(path)
        if resolved:
            return resolved
    return None

def decode_csv_upload(filename, raw):
    if not filename.lower().endswith(".csv"):
        raise ValueError(
            f"Sadece CSV dosyası yüklenebilir. Seçilen dosya: {filename}"
        )

    if not raw or len(raw) == 0:
        raise ValueError("CSV dosyası boş. Lütfen veri içeren bir CSV dosyası seçin.")

    if len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError(
            f"Dosya çok büyük ({len(raw) / (1024 * 1024):.1f} MB). "
            f"Bu kurulumda üst sınır {MAX_UPLOAD_MB} MB."
        )

    for encoding in ("utf-8-sig", "utf-8"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            text = None

    if text is None:
        raise ValueError(
            "Dosya UTF-8 CSV olarak okunamadı. Lütfen .csv dosyası yükleyin; "
            "DMG/ZIP/Excel ikili dosyaları desteklenmez."
        )

    text = repair_literal_newlines(text)

    if "," not in text and "\n" not in text:
        raise ValueError("Dosya CSV gibi görünmüyor. Lütfen geçerli bir .csv dosyası seçin.")

    return text


def read_csv_upload_sync(file):
    return decode_csv_upload(file.filename or "", file.file.read())


async def read_csv_upload(file):
    return decode_csv_upload(file.filename or "", await file.read())


def repair_literal_newlines(text):
    """
    Bazı büyük CSV exportlarında ilk satır `\\n` metniyle birleşmiş geliyor.
    Pandas bunu gerçek satır sonu saymadığı için header + ilk veri satırını
    tek satır okuyor ve Waymo şeması 100 kolon yerine ~200 kolona kayıyor.
    """
    if "\\n" in text[:8192]:
        return text.replace("\\n", "\n")
    return text


def read_csv_text_robust(raw_text):
    """Bozuk Waymo/CSV satırlarını kullanıcıyı durdurmadan okunabilir hale getirir."""
    raw_text = repair_literal_newlines(raw_text)
    try:
        return normalize_input_dataframe(pd.read_csv(io.StringIO(raw_text)))
    except pd.errors.ParserError as first_error:
        print(f"[CSV] Standart okuma başarısız, satır onarımı deneniyor: {first_error}")

    reader = csv.reader(io.StringIO(raw_text))
    rows = [row for row in reader if row and any(str(cell).strip() for cell in row)]
    if not rows:
        raise ValueError("CSV dosyası boş veya okunabilir satır içermiyor.")

    header = [str(col).strip() for col in rows[0]]
    if len(header) > 150:
        repaired_header = []
        for item in header:
            value = str(item).strip()
            try:
                float(value)
                break
            except ValueError:
                if value:
                    repaired_header.append(value)
        header = repaired_header or header[:100]

    expected = len(header)
    has_label = expected > 0 and header[-1].strip().lower() == "label"
    feature_count = expected - 1 if has_label else expected
    repaired_rows = []
    split_rows = 0
    trimmed_rows = 0
    padded_rows = 0

    def append_fixed(values):
        nonlocal trimmed_rows, padded_rows
        if len(values) > expected:
            values = values[:expected]
            trimmed_rows += 1
        elif len(values) < expected:
            values = values + (["normal"] if has_label and len(values) == feature_count else [])
            if len(values) < expected:
                values = values + [""] * (expected - len(values))
            padded_rows += 1
        repaired_rows.append(values)

    for row in rows[1:]:
        values = [str(cell).strip() for cell in row]
        if len(values) == expected:
            repaired_rows.append(values)
            continue

        if len(values) > expected:
            if expected > 0 and len(values) % expected == 0:
                for start in range(0, len(values), expected):
                    append_fixed(values[start:start + expected])
                    split_rows += 1
                continue
            if has_label and feature_count > 0 and len(values) % feature_count == 0:
                for start in range(0, len(values), feature_count):
                    append_fixed(values[start:start + feature_count] + ["normal"])
                    split_rows += 1
                continue
        append_fixed(values)

    df = normalize_input_dataframe(pd.DataFrame(repaired_rows, columns=header))
    print(
        "[CSV] Onarım tamamlandı: "
        f"{len(repaired_rows)} satır, {expected} sütun "
        f"(split={split_rows}, trim={trimmed_rows}, pad={padded_rows})"
    )
    return df


def normalize_input_dataframe(df):
    """CSV'den gelen boş/numeric görünümlü alanları pipeline için güvenli tipe çevirir."""
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    df = df.replace(r"^\s*$", np.nan, regex=True)

    label_candidates = {
        "label", "class", "target", "category", "type", "sınıf", "etiket"
    }
    waymo_cols = set(waymo_feature_columns()) if "waymo_feature_columns" in globals() else set()

    converted = 0
    for col in df.columns:
        col_name = str(col).strip()
        lower_name = col_name.lower()
        if lower_name in label_candidates:
            df[col] = df[col].fillna("normal").astype(str)
            continue

        numeric = pd.to_numeric(df[col], errors="coerce")
        non_empty = int(df[col].notna().sum())
        numeric_count = int(numeric.notna().sum())
        numeric_ratio = numeric_count / max(non_empty, 1)

        if col_name in waymo_cols or numeric_ratio >= 0.80:
            df[col] = numeric.fillna(0).astype(np.float32)
            converted += 1

    if converted:
        print(f"[CSV] Numeric normalize: {converted} sütun float32 yapıldı, boş numeric değerler 0 ile dolduruldu.")
    return df


def repaired_header_names(path):
    try:
        with open(path, "rb") as f:
            first_line = f.readline(1024 * 1024).decode("utf-8-sig", errors="replace")
    except OSError:
        return None

    if "\\n" not in first_line:
        return None

    header_text = first_line.split("\\n", 1)[0]
    names = [col.strip() for col in header_text.split(",") if col.strip()]
    if len(names) < 2:
        return None
    return names


def read_seed_dataframe(seed_path, total_rows, sample_size=10000):
    header_names = repaired_header_names(seed_path)

    if total_rows > sample_size:
        prob = sample_size / total_rows

        if header_names:
            skiprows = lambda i: i == 0 or (i > 0 and random.random() > prob)
            df = pd.read_csv(seed_path, names=header_names, header=None, skiprows=skiprows)
        else:
            skiprows = lambda i: i > 0 and random.random() > prob
            df = pd.read_csv(seed_path, skiprows=skiprows)

        if len(df) > sample_size:
            df = df.sample(sample_size, random_state=RANDOM_SEED).reset_index(drop=True)
        return df

    if header_names:
        return pd.read_csv(seed_path, names=header_names, header=None, skiprows=1)

    return pd.read_csv(seed_path)

def count_csv_rows(path):
    if not path:
        return 0
    try:
        stat = os.stat(path)
        cache_key = (path, stat.st_mtime_ns, stat.st_size)
        if cache_key not in _seed_row_cache:
            _seed_row_cache.clear()
            with open(path, "rb") as f:
                _seed_row_cache[cache_key] = max(sum(1 for _ in f) - 1, 0)
        return _seed_row_cache[cache_key]
    except OSError:
        return 0

def count_csv_rows_cached_or_fast(path, max_sync_mb=50):
    """Status ekranı için dev CSV'yi bloklamadan satır sayısı döndürür."""
    if not path:
        return 0
    try:
        stat = os.stat(path)
        cache_key = (path, stat.st_mtime_ns, stat.st_size)
        if cache_key in _seed_row_cache:
            return _seed_row_cache[cache_key]
        if stat.st_size > max_sync_mb * 1024 * 1024:
            with _seed_count_lock:
                if path not in _seed_count_pending:
                    _seed_count_pending.add(path)
                    threading.Thread(target=_count_seed_rows_background, args=(path,), daemon=True).start()
            return None
        return count_csv_rows(path)
    except OSError:
        return 0


def _count_seed_rows_background(path):
    try:
        count_csv_rows(path)
    finally:
        with _seed_count_lock:
            _seed_count_pending.discard(path)


def safe_torch_load(path, map_location="cpu"):
    """Prefer PyTorch's safer weights-only loader, then fallback for old checkpoints."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except Exception as safe_error:
        print(f"[⚠️] weights_only=True yükleme başarısız, legacy checkpoint deneniyor: {safe_error}")
        return torch.load(path, map_location=map_location, weights_only=False)

# ═══════════════ RCGAN GODMODE ═══════════════
class RCGAN_Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_embed = nn.Embedding(6, 16)
        self.lstm = nn.LSTM(80, 512, 3, batch_first=True, bidirectional=True)
        self.out = nn.Sequential(nn.Linear(1024, 256), nn.LeakyReLU(0.2), nn.Linear(256, 5))
    def forward(self, z, labels):
        emb = self.label_embed(labels).unsqueeze(1).repeat(1, 20, 1)
        x, _ = self.lstm(torch.cat([z, emb], dim=2))
        return self.out(x)

rcgan = None
RCGAN_LOAD_ERROR = None
RCGAN_LAZY_LOAD = os.getenv("SENTETIK_EAGER_RCGAN", "0") != "1"


def load_rcgan_model():
    """RCGAN modelini isteğe bağlı yükler; uygulama açılışını bloklamaz."""
    global rcgan, RCGAN_LOAD_ERROR, MODEL_PATH, MODEL_VERSION

    if rcgan is not None:
        return rcgan

    try:
        model_to_load = get_local_asset_path(MODEL_PATH)
        if not model_to_load and MODEL_PATH != BASELINE_MODEL_PATH:
            print(f"[⚠️] Seçilen RCGAN modeli bulunamadı: {MODEL_PATH}. Baseline modele dönülüyor.")
            MODEL_PATH = BASELINE_MODEL_PATH
            MODEL_VERSION = "baseline-fallback"
            model_to_load = get_local_asset_path(MODEL_PATH)

        if not model_to_load:
            RCGAN_LOAD_ERROR = (
                f"RCGAN model dosyası yerel olarak okunamıyor: {MODEL_PATH}. "
                "Dosyayı iCloud'dan indir veya yerel model kopyası ekle."
            )
            print(f"[⚠️] {RCGAN_LOAD_ERROR}")
            return None

        model = RCGAN_Generator()
        checkpoint = safe_torch_load(model_to_load, map_location='cpu')
        state_dict = checkpoint['G'] if isinstance(checkpoint, dict) and 'G' in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        model.eval()
        rcgan = model
        RCGAN_LOAD_ERROR = None
        print(f"[✅] RCGAN GODMODE yüklendi: {os.path.basename(MODEL_PATH)} ({MODEL_VERSION})")
        return rcgan
    except Exception as e:
        RCGAN_LOAD_ERROR = str(e)
        print(f"[⚠️] RCGAN hata: {e}")
        rcgan = None
        return None


if RCGAN_LAZY_LOAD:
    print(f"[ℹ️] RCGAN lazy modda: uygulama hemen açılacak, model ihtiyaçta yüklenecek ({os.path.basename(MODEL_PATH)}).")
else:
    load_rcgan_model()

# ═══════════════ BİLGİ DAMITMA MOTORU ═══════════════
def distill_dataset(df, allow_label_repair=False):
    """
    4 Katmanlı Bilgi Damıtma:
    1. Duplikasyon temizliği
    2. Gürültülü etiket tespiti + düzeltme
    3. İstatistiksel outlier filtreleme
    4. Sıfır-varyans & entropi bazlı sütun temizliği
    """
    report = {"original_rows": len(df), "original_cols": len(df.columns), "steps": []}
    
    # Label sütununu bul
    label_col = _find_label_col(df)
    numeric_cols = [c for c in df.columns if df[c].dtype in ['float64','float32','int64','int32'] and c != label_col]
    
    if not numeric_cols:
        report["steps"].append({"name": "UYARI", "detail": "Numerik sütun bulunamadı."})
        return df, report, label_col, numeric_cols
    
    # ── KATMAN 1: Duplikasyon ──
    before = len(df)
    df = df.drop_duplicates(subset=numeric_cols, keep='first')
    dup_removed = before - len(df)
    report["steps"].append({
        "name": "Duplikasyon Temizliği",
        "removed": dup_removed,
        "detail": f"{dup_removed} birebir kopya satır silindi"
    })
    
    # Yakın-duplikatlar: cosine benzerliği tabular sensör verilerinde aşırı agresif
    # davranabiliyor. Sadece ölçüm hassasiyeti kaynaklı kopyaları temizle.
    near_dup = 0
    if len(df) < 10000 and len(numeric_cols) > 0:
        rounded_numeric = df[numeric_cols].round(4)
        keep = ~rounded_numeric.duplicated(keep='first')
        near_dup = int((~keep).sum())
        if near_dup > 0:
            df = df.iloc[keep].reset_index(drop=True)
    
    if near_dup > 0:
        report["steps"].append({
            "name": "Yakın-Duplikat Temizliği",
            "removed": int(near_dup),
            "detail": f"{near_dup} yakın-kopya satır silindi (>%99.99 benzerlik)"
        })
    
    # ── KATMAN 2: Gürültülü Etiket Tespiti ──
    label_fixes = 0
    if allow_label_repair and label_col and len(df) > 50 and len(numeric_cols) >= 3:
        X = np.nan_to_num(df[numeric_cols].values.astype(np.float32))
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        
        le = LabelEncoder()
        y = le.fit_transform(df[label_col].astype(str))
        
        if len(np.unique(y)) >= 2:
            k = min(7, len(df) - 1)
            nn_model = NearestNeighbors(n_neighbors=k).fit(X_s)
            _, indices = nn_model.kneighbors(X_s)
            
            noisy_mask = np.zeros(len(df), dtype=bool)
            for i in range(len(df)):
                neighbor_labels = y[indices[i][1:]]  # kendisi hariç
                agreement = np.mean(neighbor_labels == y[i])
                if agreement < 0.15:  # %85+ komşu farklı etiket → gürültülü
                    # Çoğunluk etiketi ile düzelt
                    most_common = Counter(neighbor_labels).most_common(1)[0][0]
                    y[i] = most_common
                    label_fixes += 1
            
            if label_fixes > 0:
                df[label_col] = le.inverse_transform(y)
        
        report["steps"].append({
            "name": "Etiket Düzeltme",
            "fixed": label_fixes,
            "detail": f"{label_fixes} gürültülü etiket komşu çoğunluğuna göre düzeltildi"
        })
    elif label_col:
        report["steps"].append({
            "name": "Etiket Koruması",
            "fixed": 0,
            "detail": "Utility ölçümü için gerçek sınıf etiketleri değiştirilmedi"
        })
    
    # ── KATMAN 3: Outlier Filtreleme ──
    outlier_removed = 0
    if len(numeric_cols) >= 1 and len(df) > 10:
        # Anomali sınıflarını koru, ancak 10'dan fazla örneği olan her sınıfın KENDİ İÇİNDEKİ outlier'ları temizle
        if label_col:
            outlier_idx_list = []
            for cls_name, count in df[label_col].value_counts().items():
                if count >= 10:
                    cls_mask = df[label_col] == cls_name
                    for col in numeric_cols:
                        q1 = df.loc[cls_mask, col].quantile(0.15)
                        q3 = df.loc[cls_mask, col].quantile(0.85)
                        iqr = max(q3 - q1, 1e-5)  # Eğer tüm değerler aynıysa sıfıra bölme/aşırı filtrelemeyi engelle
                        lower = q1 - 2.5 * iqr
                        upper = q3 + 2.5 * iqr
                        bad = df.loc[cls_mask][(df.loc[cls_mask, col] < lower) | (df.loc[cls_mask, col] > upper)].index
                        outlier_idx_list.extend(bad)
            
            outlier_idx = list(set(outlier_idx_list))
            outlier_removed = len(outlier_idx)
            if outlier_removed > 0:
                df = df.drop(outlier_idx).reset_index(drop=True)
        else:
            outlier_idx_list = []
            for col in numeric_cols:
                q1 = df[col].quantile(0.15)
                q3 = df[col].quantile(0.85)
                iqr = q3 - q1
                lower = q1 - 2.5 * iqr
                upper = q3 + 2.5 * iqr
                bad = df[(df[col] < lower) | (df[col] > upper)].index
                outlier_idx_list.extend(bad)
            
            outlier_idx = list(set(outlier_idx_list))
            outlier_removed = len(outlier_idx)
            if outlier_removed > 0:
                df = df.drop(outlier_idx).reset_index(drop=True)
        
        report["steps"].append({
            "name": "Outlier Filtreleme",
            "removed": outlier_removed,
            "detail": f"{outlier_removed} fiziksel/istatistiksel outlier silindi (Strict IQR)"
        })
    
    # ── KATMAN 4: Sütun Temizliği ──
    cols_removed = []
    waymo_protected = set(f'{c}({i+1})' for c in ['x','y','speed','vx','vy'] for i in range(20))
    
    for col in numeric_cols[:]:
        if col in waymo_protected:
            continue
            
        # Sıfır varyans
        if df[col].std() < 1e-10:
            cols_removed.append(col)
            numeric_cols.remove(col)
        # %95+ NaN
        elif df[col].isna().mean() > 0.95:
            cols_removed.append(col)
            numeric_cols.remove(col)
    
    if cols_removed:
        df = df.drop(columns=cols_removed)
        report["steps"].append({
            "name": "Sütun Temizliği",
            "removed": len(cols_removed),
            "detail": f"{len(cols_removed)} bilgisiz sütun silindi: {', '.join(cols_removed[:5])}"
        })
    
    # NaN doldur
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    
    report["clean_rows"] = len(df)
    report["clean_cols"] = len(df.columns)
    report["reduction_pct"] = round((1 - len(df)/report["original_rows"]) * 100, 1) if report["original_rows"] > 0 else 0
    
    return df, report, label_col, numeric_cols

def _find_label_col(df):
    for col in df.columns:
        if col.lower() in ['label','class','target','category','type','sınıf','etiket','y']:
            return col
    for col in df.columns:
        if df[col].dtype == 'object' and 1 < df[col].nunique() < 30:
            return col
    for col in df.columns:
        if df[col].dtype in ['int64','int32'] and 1 < df[col].nunique() < 30:
            return col
    return None

# ═══════════════ ADAPTIVE GENERATION ═══════════════
def analyze_classes(df, label_col):
    if not label_col: return {}
    dist = df[label_col].value_counts().to_dict()
    return {str(k): int(v) for k, v in dist.items()}

def _is_integer_like(series):
    try:
        values = pd.to_numeric(series.dropna(), errors="coerce").dropna().values
        if len(values) == 0:
            return False
        return bool(np.all(np.isclose(values, np.round(values), atol=1e-6)))
    except Exception:
        return False

def profile_schema(df, label_col=None):
    """
    Veri setini üretim açısından profiller:
      - ID/timestamp gibi üretimi bozan kolonları ayırır
      - kategorik/sürekli/binary kolon rollerini çıkarır
      - CTGAN/SMOTE için güvenli numeric kolon listesini üretir
    """
    n_rows = max(len(df), 1)
    protected = set(waymo_feature_columns()) if "waymo_feature_columns" in globals() else set()
    id_tokens = ("id", "uuid", "guid", "hash", "index", "idx", "file", "filename", "name")
    time_tokens = ("time", "timestamp", "date", "datetime", "frame", "created", "updated")
    columns = []
    usable_numeric = []
    categorical = []
    excluded = []

    for col in df.columns:
        s = df[col]
        low = str(col).lower().strip()
        missing_pct = float(s.isna().mean() * 100.0)
        unique_count = int(s.nunique(dropna=True))
        unique_ratio = float(unique_count / n_rows)
        is_numeric = bool(pd.api.types.is_numeric_dtype(s))
        integer_like = _is_integer_like(s) if is_numeric else False
        role = "unknown"
        warnings = []

        if col == label_col:
            role = "label"
        elif col in protected:
            role = "waymo_feature"
            usable_numeric.append(col)
        elif missing_pct >= 95:
            role = "high_missing"
            excluded.append(col)
            warnings.append("Kolonun %95+ değeri boş; üretimden çıkarılır.")
        elif any(token == low or low.endswith(f"_{token}") or low.startswith(f"{token}_") for token in id_tokens):
            role = "id_like"
            excluded.append(col)
            warnings.append("ID/anahtar benzeri kolon; sentetik üretimde ezberleme riski taşır.")
        elif is_numeric and unique_ratio > 0.98 and integer_like:
            role = "id_like"
            excluded.append(col)
            warnings.append("Neredeyse her satırda benzersiz integer; ID gibi davranıyor.")
        elif any(token in low for token in time_tokens):
            role = "temporal"
            excluded.append(col)
            warnings.append("Zaman/index kolonu; dağılım üretimi yerine sıralama bağlamında kullanılmalı.")
        elif unique_count <= 2:
            role = "binary"
            categorical.append(col)
            if is_numeric:
                usable_numeric.append(col)
        elif is_numeric and unique_count <= min(30, max(10, int(n_rows * 0.05))):
            role = "ordinal_or_categorical_numeric"
            categorical.append(col)
            usable_numeric.append(col)
        elif is_numeric:
            role = "continuous_numeric"
            usable_numeric.append(col)
        elif unique_count <= min(50, max(20, int(n_rows * 0.10))):
            role = "categorical"
            categorical.append(col)
        else:
            role = "text_or_high_cardinality"
            excluded.append(col)
            warnings.append("Yüksek kardinaliteli/metinsel kolon; mevcut üretim motorları için güvenli değil.")

        columns.append({
            "name": str(col),
            "dtype": str(s.dtype),
            "role": role,
            "missing_pct": round(missing_pct, 2),
            "unique_count": unique_count,
            "unique_ratio": round(unique_ratio, 4),
            "numeric": is_numeric,
            "integer_like": integer_like,
            "warnings": warnings,
        })

    class_profile = {}
    if label_col and label_col in df.columns:
        counts = df[label_col].astype(str).value_counts()
        if len(counts) > 0:
            minority_class = str(counts.idxmin())
            majority_class = str(counts.idxmax())
            imbalance_ratio = float(counts.max() / max(counts.min(), 1))
            class_profile = {
                "minority_class": minority_class,
                "majority_class": majority_class,
                "imbalance_ratio": round(imbalance_ratio, 3),
                "is_imbalanced": bool(imbalance_ratio >= 1.5),
                "class_distribution": {str(k): int(v) for k, v in counts.to_dict().items()},
            }

    return {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": columns,
        "usable_numeric_columns": [c for c in usable_numeric if c in df.columns and c != label_col],
        "categorical_columns": [c for c in categorical if c in df.columns and c != label_col],
        "excluded_columns": [c for c in excluded if c in df.columns and c != label_col],
        "class_profile": class_profile,
    }

def _schema_role_map(schema_profile):
    if not schema_profile:
        return {}
    return {item["name"]: item["role"] for item in schema_profile.get("columns", [])}

def _safe_numeric_columns(df, numeric_cols, schema_profile=None):
    if not schema_profile:
        return [c for c in numeric_cols if c in df.columns]
    usable = [c for c in schema_profile.get("usable_numeric_columns", []) if c in numeric_cols and c in df.columns]
    return usable if usable else [c for c in numeric_cols if c in df.columns]

def _clip_to_original_domain(df_orig, df_gen, numeric_cols):
    if df_gen.empty:
        return df_gen
    for col in numeric_cols:
        if col in df_orig.columns and col in df_gen.columns:
            c_min = pd.to_numeric(df_orig[col], errors="coerce").min()
            c_max = pd.to_numeric(df_orig[col], errors="coerce").max()
            if pd.isna(c_min) or pd.isna(c_max):
                continue
            margin = (c_max - c_min) * 0.05
            c_min_bound = max(0, c_min - margin) if c_min >= 0 else c_min - margin
            df_gen[col] = pd.to_numeric(df_gen[col], errors="coerce").clip(
                lower=c_min_bound,
                upper=c_max + margin,
            )
    return df_gen

def _filter_waymo_physics(df_gen):
    if df_gen.empty or not is_waymo_frame(df_gen):
        return df_gen, {"removed": 0, "reason": "Waymo kolonları yok."}

    speed = df_gen[[f"speed({i+1})" for i in range(20)]].to_numpy(dtype=float)
    vx = df_gen[[f"vx({i+1})" for i in range(20)]].to_numpy(dtype=float)
    vy = df_gen[[f"vy({i+1})" for i in range(20)]].to_numpy(dtype=float)
    x = df_gen[[f"x({i+1})" for i in range(20)]].to_numpy(dtype=float)
    y = df_gen[[f"y({i+1})" for i in range(20)]].to_numpy(dtype=float)

    finite = np.isfinite(speed).all(axis=1) & np.isfinite(vx).all(axis=1) & np.isfinite(vy).all(axis=1)
    speed_ok = (speed >= -1e-6).all(axis=1) & (speed <= 60).all(axis=1)
    accel = np.diff(speed, axis=1) / 0.1
    accel_ok = (np.abs(accel) <= 18).mean(axis=1) >= 0.90
    step = np.sqrt(np.diff(x, axis=1) ** 2 + np.diff(y, axis=1) ** 2)
    step_ok = (step <= 10).mean(axis=1) >= 0.90
    vel_mag = np.sqrt(vx ** 2 + vy ** 2)
    coherence_error = np.abs(vel_mag - speed) / (np.abs(speed) + 1.0)
    coherence_ok = np.nanmean(coherence_error, axis=1) <= 1.25
    keep = finite & speed_ok & accel_ok & step_ok & coherence_ok

    filtered = df_gen.loc[keep].reset_index(drop=True)
    return filtered, {"removed": int((~keep).sum()), "reason": "Fiziksel validator: hız/ivme/sıçrama/speed-vx-vy kontrolü."}

def validate_synthetic_rows(df_orig, df_gen, label_col, numeric_cols, method, is_waymo):
    """Üretilen satırları domain, manifold ve label güvenliği açısından filtreler."""
    report = {
        "input_rows": int(len(df_gen)),
        "output_rows": int(len(df_gen)),
        "removed_rows": 0,
        "steps": [],
    }
    if df_gen.empty:
        return df_gen, report

    original_gen = df_gen.copy()
    before = len(df_gen)

    if label_col and label_col in df_orig.columns and label_col in df_gen.columns:
        allowed = set(df_orig[label_col].astype(str).unique())
        if method == "rcgan":
            allowed.update({"normal", "spike", "drift", "dropout", "freeze", "noise"})
            report["steps"].append({
                "name": "RCGAN Anomali Etiketleri",
                "detail": "Modelin uretebildigi tanimli yörünge anomalileri gecerli etiket olarak kabul edildi.",
            })
        mask = df_gen[label_col].astype(str).isin(allowed)
        removed = int((~mask).sum())
        if removed > 0:
            df_gen = df_gen.loc[mask].reset_index(drop=True)
            report["steps"].append({"name": "Label Güvenliği", "removed": removed})

    common_numeric = [c for c in numeric_cols if c in df_orig.columns and c in df_gen.columns]
    if common_numeric:
        finite_mask = np.isfinite(np.nan_to_num(df_gen[common_numeric].to_numpy(dtype=float), nan=np.nan)).all(axis=1)
        removed = int((~finite_mask).sum())
        if removed > 0:
            df_gen = df_gen.loc[finite_mask].reset_index(drop=True)
            report["steps"].append({"name": "Finite Kontrol", "removed": removed})

    if method != "rcgan" and len(df_orig) >= 10 and len(df_gen) >= 10 and len(common_numeric) >= 2:
        try:
            Xo = np.nan_to_num(df_orig[common_numeric].to_numpy(dtype=np.float32))
            Xg = np.nan_to_num(df_gen[common_numeric].to_numpy(dtype=np.float32))
            scaler = StandardScaler().fit(Xo)
            Xo_s = scaler.transform(Xo)
            Xg_s = scaler.transform(Xg)
            nn = NearestNeighbors(n_neighbors=min(5, len(Xo_s))).fit(Xo_s)
            orig_dist, _ = nn.kneighbors(Xo_s)
            gen_dist, _ = nn.kneighbors(Xg_s)
            orig_ref = orig_dist[:, -1]
            threshold = np.nanpercentile(orig_ref, 99) + (np.nanpercentile(orig_ref, 75) - np.nanpercentile(orig_ref, 25)) * 2.0
            keep = gen_dist[:, 0] <= max(float(threshold), 1e-6)
            removed = int((~keep).sum())
            if removed > 0:
                df_gen = df_gen.loc[keep].reset_index(drop=True)
                report["steps"].append({"name": "Manifold Uzaklık Filtresi", "removed": removed})
        except Exception as e:
            report["steps"].append({"name": "Manifold Uzaklık Filtresi", "warning": str(e)})

    if is_waymo or method == "rcgan":
        df_phys, phys_report = _filter_waymo_physics(df_gen)
        if phys_report.get("removed", 0) > 0:
            df_gen = df_phys
            report["steps"].append({"name": "Fiziksel Validator", **phys_report})

    min_keep = max(1, int(before * SYNTHETIC_VALIDATION_MIN_KEEP_RATIO))
    if len(df_gen) < min_keep and len(original_gen) >= min_keep:
        report["steps"].append({
            "name": "Validator Geri Alma",
            "detail": f"Filtre çok agresif oldu; en az {min_keep} satır korunacak şekilde orijinal üretim kullanıldı."
        })
        df_gen = original_gen.sample(min(before, len(original_gen)), random_state=RANDOM_SEED).reset_index(drop=True)

    df_gen = df_gen.drop_duplicates().reset_index(drop=True)
    report["output_rows"] = int(len(df_gen))
    report["removed_rows"] = int(before - len(df_gen))
    return df_gen, report

# ═══════════════ AKILLI YÖRÜNGE DÖNÜŞTÜRÜCÜ ═══════════════
def try_convert_to_waymo(df, label_col):
    """
    Herhangi bir koordinat/pozisyon/hareket verisini Waymo formatına dönüştürür.
    Desteklenen veri türleri:
      - GPS/Konum: lat/lon, pos_x/pos_y, x/y, latitude/longitude
      - IMU/Sensör: accel_x/accel_y, gyro_x/gyro_y
      - Hız: speed, velocity, vel_x/vel_y
    Dönüşüm: Sıralı satırları 20'şerli pencereler halinde keser,
    eksik kanalları (speed, vx, vy) pozisyon farkından hesaplar.
    """
    cols_lower = {c: c.lower().strip().replace(" ", "_").replace("-", "_") for c in df.columns}
    
    # ── 1. Pozisyon sütunlarını bul ──
    x_col, y_col = None, None
    speed_col, vx_col, vy_col = None, None, None
    
    # Pozisyon aday isimleri (öncelik sırasıyla)
    # ⚠️ SADECE gerçek konum/pozisyon sütunları! İvme (accel), jiroskop (gyro) vs. KONUM DEĞİL!
    x_candidates = [
        'x', 'pos_x', 'position_x', 'coord_x', 'coordinate_x', 'center_x', 'centroid_x',
        'ego_x', 'vehicle_x', 'agent_x', 'object_x', 'track_x', 'map_x', 'local_x',
        'global_x', 'longitude', 'lon', 'lng', 'easting'
    ]
    y_candidates = [
        'y', 'pos_y', 'position_y', 'coord_y', 'coordinate_y', 'center_y', 'centroid_y',
        'ego_y', 'vehicle_y', 'agent_y', 'object_y', 'track_y', 'map_y', 'local_y',
        'global_y', 'latitude', 'lat', 'northing'
    ]
    speed_candidates = ['speed', 'speed_mps', 'ego_speed', 'vehicle_speed', 'velocity', 'vel', 'spd']
    vx_candidates = ['vx', 'vel_x', 'velocity_x', 'v_x', 'ego_vx', 'vehicle_vx', 'agent_vx']
    vy_candidates = ['vy', 'vel_y', 'velocity_y', 'v_y', 'ego_vy', 'vehicle_vy', 'agent_vy']
    motion_context = [
        'time', 'timestamp', 'frame', 'frame_id', 't', 'dt', 'id', 'track_id', 'vehicle_id',
        'agent_id', 'object_id', 'label', 'class', 'target', 'type', 'heading', 'yaw',
        'yaw_rate', 'orientation', 'steering', 'steering_angle', 'lane_id', 'lane',
        'lane_offset', 'throttle', 'brake', 'accel', 'acceleration', 'acceleration_x',
        'acceleration_y', 'accel_x', 'accel_y', 'ax', 'ay', 'longitudinal_accel',
        'lateral_accel'
    ]
    routing_context = [c for c in motion_context if c not in ['label', 'class', 'target', 'type', 'id']]
    
    # Yörünge ile ilgili tüm sütun isimleri
    trajectory_keywords = set(x_candidates + y_candidates + speed_candidates + vx_candidates + vy_candidates + motion_context)
    
    for orig, low in cols_lower.items():
        if not x_col and low in x_candidates: x_col = orig
        if not y_col and low in y_candidates: y_col = orig
        if not speed_col and low in speed_candidates: speed_col = orig
        if not vx_col and low in vx_candidates: vx_col = orig
        if not vy_col and low in vy_candidates: vy_col = orig
    
    # En az 2 KONUM sütunu bulamazsak dönüştürme yapılamaz → CTGAN'a düşsün
    if not x_col or not y_col:
        return None

    # Generic x/y kolonlarını ancak hareket bağlamı varsa RCGAN'a al.
    has_motion_signal = any([speed_col, vx_col, vy_col]) or any(cols_lower[c] in routing_context for c in df.columns)
    if cols_lower[x_col] in ['x', 'y'] and cols_lower[y_col] in ['x', 'y'] and not has_motion_signal:
        print("[ℹ️] Generic x/y bulundu ama hız/zaman/araç bağlamı yok; CTGAN'a yönlendiriliyor.")
        return None
    
    # ── Zengin Veri Seti Kontrolü ──
    # Eğer dosyada konum dışı sütunlar çoğunluktaysa (IMU, sensör, vs.),
    # RCGAN'a dönüştürmek veri kaybına yol açar. CTGAN tüm sütunları korur.
    non_trajectory_cols = [c for c in df.columns if cols_lower[c] not in trajectory_keywords]
    trajectory_cols_found = [c for c in df.columns if cols_lower[c] in trajectory_keywords]
    
    rich_limit = max(8, len(trajectory_cols_found) * 2)
    if len(non_trajectory_cols) > rich_limit:
        # Dosyada çok fazla yörünge-dışı sütun var → zengin veri seti
        # CTGAN hepsini öğrensin, RCGAN sadece 5 kanal bilir, geri kalanı kaybolur
        print(f"[ℹ️] Zengin veri seti tespit edildi ({len(non_trajectory_cols)} ekstra sütun: {non_trajectory_cols[:5]}...)")
        print(f"[ℹ️] Tüm sütunları korumak için CTGAN'a yönlendiriliyor.")
        return None
    
    print(f"[🔄] Saf yörünge verisi: x={x_col}, y={y_col}, speed={speed_col}")
    
    # ── 2. Verileri çıkar ──
    x_data = pd.to_numeric(df[x_col], errors='coerce').fillna(0).values
    y_data = pd.to_numeric(df[y_col], errors='coerce').fillna(0).values
    
    # Speed yoksa vx/vy veya pozisyon farkından hesapla
    if speed_col:
        speed_data = pd.to_numeric(df[speed_col], errors='coerce').fillna(0).values
    elif vx_col and vy_col:
        vx_tmp = pd.to_numeric(df[vx_col], errors='coerce').fillna(0).values
        vy_tmp = pd.to_numeric(df[vy_col], errors='coerce').fillna(0).values
        speed_data = np.sqrt(vx_tmp**2 + vy_tmp**2)
    else:
        dx = np.diff(x_data, prepend=x_data[0])
        dy = np.diff(y_data, prepend=y_data[0])
        speed_data = np.sqrt(dx**2 + dy**2)

    movement_span = (np.nanmax(x_data) - np.nanmin(x_data)) + (np.nanmax(y_data) - np.nanmin(y_data))
    if movement_span < 1e-6:
        print("[ℹ️] Konum kolonları hareket içermiyor; CTGAN'a yönlendiriliyor.")
        return None
    
    # vx/vy yoksa pozisyon farkından hesapla
    if vx_col:
        vx_data = pd.to_numeric(df[vx_col], errors='coerce').fillna(0).values
    else:
        vx_data = np.diff(x_data, prepend=x_data[0])
    
    if vy_col:
        vy_data = pd.to_numeric(df[vy_col], errors='coerce').fillna(0).values
    else:
        vy_data = np.diff(y_data, prepend=y_data[0])
    
    # Label varsa al
    if label_col and label_col in df.columns:
        labels = df[label_col].values
    else:
        labels = None
    
    # ── 3. 20'şerli Pencere Oluştur ──
    WINDOW = 20
    STRIDE = 10  # %50 örtüşme → daha fazla yörünge
    n = len(x_data)
    
    if n < WINDOW:
        return None
    
    rows = []
    for start in range(0, n - WINDOW + 1, STRIDE):
        end = start + WINDOW
        
        # Penceredeki verileri al
        xw = x_data[start:end]
        yw = y_data[start:end]
        sw = speed_data[start:end]
        vxw = vx_data[start:end]
        vyw = vy_data[start:end]
        
        # Relative koordinatlara dönüştür (başlangıç noktası = 0,0)
        xw = xw - xw[0]
        yw = yw - yw[0]
        
        row = {}
        for i in range(WINDOW):
            row[f'x({i+1})'] = float(xw[i])
            row[f'y({i+1})'] = float(yw[i])
            row[f'speed({i+1})'] = float(abs(sw[i]))
            row[f'vx({i+1})'] = float(vxw[i])
            row[f'vy({i+1})'] = float(vyw[i])
        
        # Label: penceredeki en sık label
        if labels is not None:
            window_labels = labels[start:end]
            try:
                most_common = pd.Series(window_labels).mode()[0]
                row['label'] = most_common
            except:
                row['label'] = 'normal'
        else:
            row['label'] = 'normal'
        
        rows.append(row)
    
    if not rows:
        return None
    
    df_waymo = pd.DataFrame(rows)
    print(f"[🔄] {n} satır → {len(df_waymo)} yörünge penceresi (20 adım, stride {STRIDE})")
    return df_waymo


def generate_adaptive(df, label_col, numeric_cols, n_samples):
    """Waymo → RCGAN, Yörünge verisi → Dönüştür+RCGAN, diğerleri → CTGAN, fallback → SMOTE"""
    is_waymo = is_waymo_frame(df)
    if is_waymo and rcgan is None:
        load_rcgan_model()
    schema_profile = profile_schema(df, label_col)
    safe_numeric_cols = _safe_numeric_columns(df, numeric_cols, schema_profile)
    generation_report = {
        "schema_profile": schema_profile,
        "selected_numeric_columns": [str(c) for c in safe_numeric_cols],
        "validator": {},
        "strategy_notes": [],
        "requested_rows": int(n_samples),
    }
    
    # ── Akıllı Yörünge Dönüştürücü ──
    # Waymo formatında değilse ama koordinat/pozisyon verisi varsa, otomatik dönüştür
    df_converted = None
    if not is_waymo and rcgan is None:
        # Ham x/y/speed benzeri trajektör verileri RCGAN'a çevrilebiliyorsa modeli tembel yükle.
        try:
            load_rcgan_model()
        except Exception as exc:
            print(f"[⚠️] RCGAN lazy yükleme başarısız, genel üreticiye geçilecek: {exc}")

    if not is_waymo and rcgan is not None:
        df_converted = try_convert_to_waymo(df, label_col)
        if df_converted is not None:
            is_waymo = True
            df = df_converted
            numeric_cols = waymo_feature_columns()
            safe_numeric_cols = numeric_cols
            schema_profile = profile_schema(df, "label" if "label" in df.columns else label_col)
            generation_report["schema_profile"] = schema_profile
            generation_report["selected_numeric_columns"] = [str(c) for c in safe_numeric_cols]
            generation_report["strategy_notes"].append("Ham koordinat verisi 20 adımlı Waymo trajektör şemasına dönüştürüldü.")
            print(f"[🔄] Yörünge verisi Waymo formatına dönüştürüldü: {len(df)} yörünge")
    
    df_gen = pd.DataFrame()
    method = ""
    # Waymo verisi ise ve RCGAN modeli yüklüyse DAİMA RCGAN kullan.
    if is_waymo and rcgan is not None:
        candidate_count = max(n_samples, int(np.ceil(n_samples / 0.60)))
        df_gen, method = generate_waymo(df, candidate_count, requested_rows=n_samples), 'rcgan'
        generation_report["strategy_notes"].append("RCGAN + prosedürel anomaly üretimi seçildi.")
        if df_gen.attrs.get("rcgan_postprocess"):
            generation_report["rcgan_postprocess"] = df_gen.attrs.get("rcgan_postprocess")
    
    # CTGAN: Genel veri setleri için on-the-fly GAN eğitimi
    elif is_ctgan_available(load=False) and len(df) >= CTGAN_MIN_ROWS and len(safe_numeric_cols) >= 2:
        try:
            candidate_count = max(n_samples, int(np.ceil(n_samples / 0.60)))
            df_gen, method = generate_ctgan(df, label_col, safe_numeric_cols, candidate_count, schema_profile), 'ctgan'
            generation_report["candidate_rows"] = int(candidate_count)
            generation_report["strategy_notes"].append("CTGAN schema-aware modda eğitildi; ID/timestamp/high-missing kolonlar üretimden çıkarıldı.")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f'[⚠️] CTGAN başarısız, SMOTE fallback: {e}')
            candidate_count = max(n_samples, int(np.ceil(n_samples / 0.60)))
            df_gen, method = generate_smart(df, label_col, safe_numeric_cols, candidate_count, schema_profile), 'smote'
            generation_report["candidate_rows"] = int(candidate_count)
            generation_report["strategy_notes"].append(f"CTGAN başarısız oldu, Borderline-SMOTE fallback kullanıldı: {e}")
    else:
        candidate_count = max(n_samples, int(np.ceil(n_samples / 0.60)))
        df_gen, method = generate_smart(df, label_col, safe_numeric_cols, candidate_count, schema_profile), 'smote'
        generation_report["candidate_rows"] = int(candidate_count)
        generation_report["strategy_notes"].append("Veri CTGAN için küçük/uygunsuz; Borderline-SMOTE + Gaussian fallback seçildi.")

    # ── Domain Shift Koruması (Fiziksel Sınırlandırma) ──
    # Üretilen veriyi orijinal temiz verinin fiziksel sınırlarına (min-max) hapset
    if not df_gen.empty and method != 'rcgan':
        df_gen = _clip_to_original_domain(df, df_gen, safe_numeric_cols)
        generation_report["strategy_notes"].append("Sentetik numeric değerler orijinal domain aralığına %5 toleransla sınırlandı.")

    if not df_gen.empty:
        df_gen, validator_report = validate_synthetic_rows(
            df, df_gen, label_col, safe_numeric_cols if method != "rcgan" else waymo_feature_columns(), method, is_waymo
        )
        if len(df_gen) > n_samples:
            excess = int(len(df_gen) - n_samples)
            df_gen = df_gen.sample(n_samples, random_state=RANDOM_SEED).reset_index(drop=True)
            validator_report["steps"].append({
                "name": "Hedef Örnek Sayısı",
                "removed": excess,
                "detail": "Kalite kontrolünden geçen adaylar hedef çıktı sayısına indirildi.",
            })
            validator_report["output_rows"] = int(len(df_gen))
            validator_report["removed_rows"] = int(validator_report["input_rows"] - len(df_gen))
        if method == "rcgan":
            post_report = generation_report.get("rcgan_postprocess")
            if post_report is not None:
                post_report["output_rows_after_validation"] = int(len(df_gen))
                post_report["anomaly_distribution_final"] = {
                    str(k): int(v) for k, v in df_gen["label"].value_counts().to_dict().items()
                }
        generation_report["validator"] = validator_report
        safe_write_csv(df_gen, os.path.join(OUTPUT_DIR, 'live_synthetic_output.csv'), index=False)
        
    return df_gen, method, is_waymo, generation_report

def generate_ctgan(df, label_col, numeric_cols, n_samples, schema_profile=None):
    """On-the-fly CTGAN eğitimi — her veri setine adapte olur"""
    if not is_ctgan_available(load=True):
        raise ImportError(CTGAN_IMPORT_ERROR or "CTGAN modülü yüklenemedi.")
    reset_random_state()
    
    # Eğitim verisini hazırla
    train_cols = list(numeric_cols)
    if label_col and label_col in df.columns:
        train_cols.append(label_col)
    df_train = df[train_cols].copy()
    
    # ── Akıllı Örnekleme ──
    # CTGAN dağılımı öğrenmek için 10K satır fazlasıyla yeterli.
    # 500K satır sokmak saatlerce sürer, gereksiz.
    if len(df_train) > CTGAN_MAX_ROWS:
        # Sınıf dengeli (stratified) örnekleme yap
        try:
            from sklearn.model_selection import train_test_split
            df_train, _ = train_test_split(
                df_train, train_size=CTGAN_MAX_ROWS,
                stratify=df_train[label_col], random_state=RANDOM_SEED
            )
        except:
            df_train = df_train.sample(CTGAN_MAX_ROWS, random_state=RANDOM_SEED)
        print(f'[CTGAN] {len(df)} satırdan {len(df_train)} örneklem alındı (hız optimizasyonu)')
    
    print(f'[CTGAN] {len(df_train)} satır üzerinde eğitim başlıyor...')
    
    # Kategorik sütunları belirle
    discrete_cols = [label_col] if label_col and label_col in df_train.columns else []
    categorical_from_schema = set(schema_profile.get("categorical_columns", [])) if schema_profile else set()
    for col in numeric_cols:
        if col in categorical_from_schema or df_train[col].nunique() < 10:
            discrete_cols.append(col)
    discrete_cols = list(dict.fromkeys([c for c in discrete_cols if c in df_train.columns]))
    
    # Epoch sayısı veri boyutuna göre ayarla (hız vs kalite)
    n_rows = len(df_train)
    if n_rows < 500:
        epochs = CTGAN_EPOCHS_SMALL
    elif n_rows < 2000:
        epochs = CTGAN_EPOCHS_MEDIUM
    else:
        epochs = CTGAN_EPOCHS_LARGE  # Büyük veri → az epoch, hızlı
    
    # CTGAN batch_size must be a multiple of pac (default 10)
    ideal_batch = min(500, len(df_train))
    batch_size = max(10, (ideal_batch // 10) * 10)
    
    # CTGAN eğit
    model = CTGAN(
        epochs=epochs,
        batch_size=batch_size,
        generator_dim=(128, 128),
        discriminator_dim=(128, 128),
        verbose=False
    )
    model.fit(df_train, discrete_columns=discrete_cols)
    
    # Sentetik veri üret
    reset_random_state()
    df_gen = model.sample(n_samples)
    print(f'[CTGAN] {n_samples} sentetik örnek üretildi')
    
    # Kaydet
    safe_write_csv(df_gen, os.path.join(OUTPUT_DIR, 'live_synthetic_output.csv'), index=False)
    return df_gen

def generate_smart(df, label_col, numeric_cols, n_samples, schema_profile=None):
    if not label_col or not numeric_cols:
        return pd.DataFrame()
    
    dist = df[label_col].value_counts().to_dict()
    if not dist: return pd.DataFrame()
    
    classes = list(dist.keys())
    counts = np.array([max(int(dist[c]), 1) for c in classes], dtype=np.float64)
    median_count = float(np.median(counts))
    inv_weights = 1.0 / np.sqrt(counts)
    inv_weights = inv_weights / inv_weights.sum()
    quotas = np.maximum(1, np.floor(inv_weights * n_samples).astype(int))
    while quotas.sum() < n_samples:
        quotas[int(np.argmin(counts))] += 1
    while quotas.sum() > n_samples and quotas.max() > 1:
        quotas[int(np.argmax(quotas))] -= 1
    
    X = np.nan_to_num(df[numeric_cols].values.astype(np.float32))
    y_all = df[label_col].astype(str).values
    X_global_s = None
    danger_by_class = {}
    if len(classes) >= 2 and len(df) > 8 and len(numeric_cols) >= 2:
        try:
            X_global_s = StandardScaler().fit_transform(X)
            k_global = min(8, len(df) - 1)
            nn_global = NearestNeighbors(n_neighbors=k_global + 1).fit(X_global_s)
            _, global_nbrs = nn_global.kneighbors(X_global_s)
            for cls in classes:
                cls_str = str(cls)
                cls_indices = np.where(y_all == cls_str)[0]
                danger = []
                for idx in cls_indices:
                    nbr_labels = y_all[global_nbrs[idx][1:]]
                    other_ratio = float(np.mean(nbr_labels != cls_str))
                    if 0.25 <= other_ratio <= 0.85:
                        danger.append(idx)
                danger_by_class[cls] = danger
        except Exception:
            danger_by_class = {}
    rows = []
    
    for class_index, cls in enumerate(classes):
        mask = df[label_col] == cls
        X_cls = X[mask]
        target_count = int(quotas[class_index])

        if len(X_cls) == 0:
            continue
        if len(X_cls) == 1:
            cls_std = np.ones(X_cls.shape[1], dtype=np.float32) * 1e-4
            for _ in range(target_count):
                pt = X_cls[0] + np.random.normal(0, cls_std)
                row = {numeric_cols[j]: float(pt[j]) for j in range(len(numeric_cols))}
                row[label_col] = cls
                rows.append(row)
            continue
        
        cls_std = np.std(X_cls, axis=0) + 1e-8
        k = min(5, len(X_cls)-1)
        nn_m = NearestNeighbors(n_neighbors=k+1).fit(X_cls)
        
        minority_focus = len(X_cls) <= median_count
        n_smote = int(target_count * (0.70 if minority_focus else 0.50))
        n_noise = target_count - n_smote
        original_indices = np.where(mask.values if hasattr(mask, "values") else mask)[0]
        danger_global_indices = danger_by_class.get(cls, [])
        danger_local_indices = []
        if danger_global_indices:
            index_to_local = {int(global_idx): local_idx for local_idx, global_idx in enumerate(original_indices)}
            danger_local_indices = [index_to_local[i] for i in danger_global_indices if i in index_to_local]
        base_candidates = danger_local_indices if danger_local_indices else list(range(len(X_cls)))
        
        for _ in range(n_smote):
            idx = random.choice(base_candidates)
            _, nbrs = nn_m.kneighbors([X_cls[idx]])
            nbr = X_cls[random.choice(nbrs[0][1:])]
            alpha = random.uniform(0.1, 0.9)
            pt = X_cls[idx] + alpha * (nbr - X_cls[idx])
            row = {numeric_cols[j]: float(pt[j]) for j in range(len(numeric_cols))}
            row[label_col] = cls; rows.append(row)
        
        for _ in range(n_noise):
            base = X_cls[random.choice(base_candidates)].copy()
            noise_scale = 0.10 if minority_focus else 0.07
            pt = base + np.random.normal(0, cls_std * noise_scale)
            row = {numeric_cols[j]: float(pt[j]) for j in range(len(numeric_cols))}
            row[label_col] = cls; rows.append(row)
    
    if not rows: return pd.DataFrame()
    df_gen = pd.DataFrame(rows)
    if len(df_gen) > n_samples:
        df_gen = df_gen.sample(n_samples, random_state=RANDOM_SEED).reset_index(drop=True)
    safe_write_csv(df_gen, os.path.join(OUTPUT_DIR, "live_synthetic_output.csv"), index=False)
    return df_gen

def _smooth_vector(values, kernel=None):
    arr = np.asarray(values, dtype=float).copy()
    if len(arr) < 3:
        return arr
    kernel = np.asarray(kernel if kernel is not None else [0.2, 0.6, 0.2], dtype=float)
    padded = np.pad(arr, (1, 1), mode="edge")
    smoothed = np.convolve(padded, kernel / kernel.sum(), mode="valid")
    smoothed[0] = arr[0]
    smoothed[-1] = arr[-1]
    return smoothed

def _repair_waymo_sequence(seq, severity_factor=1.0):
    """RCGAN sonrası fiziksel tutarlılığı artırır; anomaly karakterini tamamen silmez."""
    arr = np.asarray(seq, dtype=float).copy()
    if arr.shape != (20, 5):
        return arr

    arr[:, 0] = _smooth_vector(arr[:, 0], [0.15, 0.70, 0.15])
    arr[:, 1] = _smooth_vector(arr[:, 1], [0.15, 0.70, 0.15])
    arr[:, 2] = _smooth_vector(np.clip(arr[:, 2], 0, 60), [0.20, 0.60, 0.20])

    dx = np.diff(arr[:, 0], prepend=arr[0, 0])
    dy = np.diff(arr[:, 1], prepend=arr[0, 1])
    derived_vx = np.clip(dx / 0.1, -20, 20)
    derived_vy = np.clip(dy / 0.1, -20, 20)

    # Mevcut model çıktısını yok etmeden hız vektörünü konum farkıyla uyumlu hale getir.
    blend = 0.45 if severity_factor >= 1.0 else 0.60
    arr[:, 3] = np.clip((1.0 - blend) * arr[:, 3] + blend * derived_vx, -20, 20)
    arr[:, 4] = np.clip((1.0 - blend) * arr[:, 4] + blend * derived_vy, -20, 20)

    vel_mag = np.sqrt(arr[:, 3] ** 2 + arr[:, 4] ** 2)
    arr[:, 2] = np.clip((0.70 * arr[:, 2]) + (0.30 * vel_mag), 0, 60)

    accel = np.diff(arr[:, 2], prepend=arr[0, 2]) / 0.1
    too_high = np.abs(accel) > 18
    if too_high.any():
        for i in np.where(too_high)[0]:
            if i == 0:
                continue
            delta = np.clip(arr[i, 2] - arr[i - 1, 2], -1.8, 1.8)
            arr[i, 2] = np.clip(arr[i - 1, 2] + delta, 0, 60)

    return arr

def _enforce_waymo_diversity(df_gen, target_count):
    if df_gen.empty or not is_waymo_frame(df_gen):
        return df_gen, {"removed": 0, "detail": "Waymo diversity filtresi uygulanmadı."}

    feature_cols = []
    for family in ["x", "y", "speed"]:
        feature_cols.extend([f"{family}({i+1})" for i in range(20)])
    X = np.nan_to_num(df_gen[feature_cols].to_numpy(dtype=np.float32))
    std = X.std(axis=0) + 1e-6
    Xn = (X - X.mean(axis=0)) / std
    signatures = pd.DataFrame(np.round(Xn, 2)).astype(str).agg("|".join, axis=1)
    keep = ~signatures.duplicated(keep="first")
    filtered = df_gen.loc[keep].reset_index(drop=True)
    removed = int((~keep).sum())

    if len(filtered) < max(1, int(target_count * 0.60)):
        return df_gen.reset_index(drop=True), {
            "removed": 0,
            "detail": "Diversity filtresi çok agresif olacağı için geri alındı.",
        }

    return filtered, {
        "removed": removed,
        "detail": "Normalize edilmiş x/y/speed imzası ile aşırı benzer trajektörler elendi.",
    }

def generate_waymo(df, n_samples, requested_rows=None):
    LM = {'normal':0,'spike':1,'drift':2,'dropout':3,'freeze':4,'noise':5}
    LN = {v:k for k,v in LM.items()}
    severity_factor = {"low": 0.65, "medium": 1.0, "high": 1.35}.get(RCGAN_ANOMALY_SEVERITY, 1.0)
    post_report = {
        "severity": RCGAN_ANOMALY_SEVERITY,
        "severity_factor": severity_factor,
        "requested_rows": int(requested_rows if requested_rows is not None else n_samples),
        "candidate_rows": int(n_samples),
        "anomaly_distribution_before_filter": {},
        "anomaly_distribution_after_filter": {},
        "trajectory_repair": {
            "smoothing": "x/y/speed 3-nokta hareketli ortalama",
            "velocity_coherence": "vx/vy konum farkıyla harmanlandı; speed velocity magnitude ile dengelendi",
            "acceleration_limit_mps2": 18,
        },
        "diversity_filter": {},
    }
    pool = []
    dc = df[df['label']=='normal'] if 'label' in df.columns else df
    if len(dc)==0: dc=df
    
    # 1000 yörüngeyi hep baştan almak yerine seed kontrollü rastgele seç
    sample_df = dc.sample(min(1000, len(dc)), random_state=RANDOM_SEED) if len(dc) > 0 else dc
    for _, r in sample_df.iterrows():
        try:
            v = np.stack([
                pd.to_numeric(
                    pd.Series([r[f'{c}({i+1})'] for i in range(20)]),
                    errors="coerce",
                ).fillna(0).to_numpy(dtype=np.float32)
                for c in ['x','y','speed','vx','vy']
            ], axis=1)
            if np.all(v[:,0]==0) and np.all(v[:,1]==0): continue
            pool.append(v)
        except Exception as e:
            pass
    if not pool: return pd.DataFrame()
    
    sc = [f'speed({i+1})' for i in range(20)]
    xc = [f'x({i+1})' for i in range(20)]
    speed_values = df[sc].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    x_values = df[xc].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    sm = max(min(np.nanpercentile(speed_values,95),50),5)
    xr = min(max(np.nanpercentile(x_values,95)-np.nanpercentile(x_values,5),1),40)
    
    def spike(t):
        t=t.copy();i=random.randint(2,17)
        t[i,0]+=random.choice([-1,1])*random.uniform(xr*.05,xr*.15)*severity_factor
        t[i,2]=min(t[i,2]*random.uniform(1.0 + severity_factor, 1.0 + 3.0*severity_factor), sm*(1.2 + severity_factor))
        return t
    def drift(t):
        t=t.copy()
        drift_step = xr * 0.01 * severity_factor
        for i in range(10,20):
            t[i,0] += (i - 10) * drift_step
        return t
    def dropout(t):
        t=t.copy()
        duration = 3 if severity_factor < 1 else 5 if severity_factor < 1.2 else 7
        end = min(20, 10 + duration)
        for i in range(10,end): t[i,2:]=0;t[i,:2]=t[9,:2]
        return t
    def freeze(t):
        t=t.copy()
        duration = 3 if severity_factor < 1 else 5 if severity_factor < 1.2 else 7
        end = min(20, 10 + duration)
        t[10:end,:]=t[10,:]
        return t
    def noise(t):
        t=t.copy()
        t[:,0]+=np.random.normal(0,xr*.02*severity_factor,20)
        t[:,2]=np.clip(t[:,2]+np.random.normal(0,sm*.05*severity_factor,20),0,None)
        return t
    
    gens={1:spike,2:drift,3:dropout,4:freeze,5:noise}
    seqs,yl=[],[]
    with torch.no_grad():
        reset_random_state()
        ya=np.random.choice([1,2,3,4,5],size=n_samples)
        z=torch.randn(n_samples,20,64)
        
        batch_size = 500
        Xai_list = []
        ya_tensor = torch.tensor(ya, dtype=torch.long)
        for i in range(0, n_samples, batch_size):
            end = min(i + batch_size, n_samples)
            Xai_batch = rcgan(z[i:end], ya_tensor[i:end]).numpy()
            Xai_list.append(Xai_batch)
        Xai = np.concatenate(Xai_list, axis=0) if Xai_list else np.empty((0, 20, 1))
        
        for i in range(n_samples):
            c=ya[i]; b=random.choice(pool).copy()
            try: a=gens[c](b)
            except: a=b
            ax=np.convolve(Xai[i,:,0]-np.mean(Xai[i,:,0]),np.ones(3)/3,mode='same')
            if np.max(np.abs(ax))>0: ax/=np.max(np.abs(ax))
            if c in [1,5]: a[:,0]+=ax*min(max(np.std(ax)+.1,.2),2.5)*(xr*.05)*severity_factor
            a[:,2]=np.clip(a[:,2],0,60);a[:,3:]=np.clip(a[:,3:],-20,20)
            a=_repair_waymo_sequence(a, severity_factor)
            seqs.append(a);yl.append(c)
    Xt=np.transpose(np.array(seqs),(0,2,1))
    cols=[f'{f}({i+1})' for f in ['x','y','speed','vx','vy'] for i in range(20)]
    dg=pd.DataFrame(Xt.reshape(-1,100),columns=cols)
    dg['label']=[LN[l] for l in yl]
    post_report["anomaly_distribution_before_filter"] = {str(k): int(v) for k, v in dg["label"].value_counts().to_dict().items()}
    dg, diversity_report = _enforce_waymo_diversity(
        dg, int(requested_rows if requested_rows is not None else n_samples)
    )
    post_report["diversity_filter"] = diversity_report
    post_report["anomaly_distribution_after_filter"] = {str(k): int(v) for k, v in dg["label"].value_counts().to_dict().items()}
    post_report["output_rows_after_postprocess"] = int(len(dg))
    dg.attrs["rcgan_postprocess"] = post_report
    safe_write_csv(dg, os.path.join(OUTPUT_DIR,"live_synthetic_output.csv"),index=False)
    return dg


def waymo_feature_columns():
    return [f'{f}({i+1})' for f in ['x','y','speed','vx','vy'] for i in range(20)]


def is_waymo_frame(df):
    return all(c in df.columns for c in waymo_feature_columns())


def prepare_evaluation_frame(df_clean, df_gen, method, is_waymo, label_col, numeric_cols):
    """
    RCGAN rota dönüşümü yapıldıysa kalite metriklerini aynı şema üstünden hesapla.
    Böylece x/y gibi ham koordinat verisi Waymo formatına çevrilmiş olsa bile
    evaluate() orijinal ve sentetik tarafı farklı kolon sırasıyla kıyaslamaz.
    """
    if method == "rcgan" and is_waymo_frame(df_gen):
        eval_df = df_clean

        if not is_waymo_frame(eval_df):
            converted = try_convert_to_waymo(df_clean, label_col)
            if converted is not None and not converted.empty:
                eval_df = converted

        if is_waymo_frame(eval_df):
            eval_df = eval_df.copy()
            eval_label_col = "label" if "label" in eval_df.columns else label_col
            if not eval_label_col or eval_label_col not in eval_df.columns:
                eval_df["label"] = "normal"
                eval_label_col = "label"

            return eval_df, eval_label_col, waymo_feature_columns()

    return df_clean, label_col, numeric_cols


def make_utility_split(df, label_col):
    """Etiketli genel veride üreticiye hiç gösterilmeyen validation/test parçaları oluştur."""
    if not label_col or label_col not in df.columns or is_waymo_frame(df):
        return None
    counts = df[label_col].astype(str).value_counts()
    if len(counts) < 2 or int(counts.min()) < 20:
        return None
    try:
        train_val, test = train_test_split(
            df, test_size=0.20, random_state=RANDOM_SEED,
            stratify=df[label_col].astype(str),
        )
        train, validation = train_test_split(
            train_val, test_size=0.25, random_state=RANDOM_SEED,
            stratify=train_val[label_col].astype(str),
        )
        return (
            train.reset_index(drop=True),
            validation.reset_index(drop=True),
            test.reset_index(drop=True),
        )
    except ValueError:
        return None


def _fit_task_predict(train_df, eval_df, label_col, numeric_cols):
    from sklearn.metrics import f1_score, recall_score, precision_score, average_precision_score

    if len(train_df) > 25000:
        train_df = train_df.sample(25000, random_state=RANDOM_SEED).reset_index(drop=True)

    classes = sorted(train_df[label_col].astype(str).unique().tolist())
    encoder = LabelEncoder().fit(classes)
    X_train = np.nan_to_num(train_df[numeric_cols].to_numpy(dtype=np.float32))
    X_eval = np.nan_to_num(eval_df[numeric_cols].to_numpy(dtype=np.float32))
    y_train = encoder.transform(train_df[label_col].astype(str))
    y_eval = encoder.transform(eval_df[label_col].astype(str))
    scaler = StandardScaler().fit(X_train)
    model = HistGradientBoostingClassifier(
        max_iter=100,
        max_depth=5,
        learning_rate=0.08,
        l2_regularization=0.1,
        random_state=RANDOM_SEED,
    )
    class_counts = np.bincount(y_train, minlength=len(classes)).astype(np.float64)
    class_counts[class_counts == 0] = 1.0
    sample_weight = (len(y_train) / (len(classes) * class_counts))[y_train]
    model.fit(scaler.transform(X_train), y_train, sample_weight=sample_weight)
    pred = model.predict(scaler.transform(X_eval))
    minority_class = train_df[label_col].astype(str).value_counts().idxmin()
    minority_idx = int(encoder.transform([str(minority_class)])[0])
    true_minority = y_eval == minority_idx
    pred_minority = pred == minority_idx
    metrics = {
        "weighted_f1": float(f1_score(y_eval, pred, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y_eval, pred, average="macro", zero_division=0)),
        "minority_recall": float(recall_score(true_minority, pred_minority, zero_division=0)),
        "minority_precision": float(precision_score(true_minority, pred_minority, zero_division=0)),
        "minority_class": str(minority_class),
    }
    if len(classes) == 2 and hasattr(model, "predict_proba"):
        scores = model.predict_proba(scaler.transform(X_eval))[:, minority_idx]
        metrics["minority_pr_auc"] = float(average_precision_score(true_minority, scores))
    return metrics


def _sample_synthetic_recipe(df_gen, label_col, minority_class, count, mode, seed_offset=0):
    if count <= 0:
        return df_gen.iloc[0:0].copy()
    valid = df_gen.copy()
    if label_col not in valid.columns:
        return valid.sample(count, replace=count > len(valid), random_state=RANDOM_SEED + seed_offset)
    if mode == "minority":
        minority = valid[valid[label_col].astype(str) == str(minority_class)]
        if minority.empty:
            return minority
        return minority.sample(count, replace=count > len(minority), random_state=RANDOM_SEED + seed_offset)
    return valid.sample(count, replace=count > len(valid), random_state=RANDOM_SEED + seed_offset)


def _dominant_feature_diagnostic(train_df, test_df, label_col, numeric_cols):
    """Tek bir kolon hedefi neredeyse çözüyor mu; tavan skorun açıklamasını üret."""
    best = None
    for col in numeric_cols:
        try:
            model = DecisionTreeClassifier(max_depth=1, class_weight="balanced", random_state=RANDOM_SEED)
            model.fit(train_df[[col]].fillna(0), train_df[label_col].astype(str))
            pred = model.predict(test_df[[col]].fillna(0))
            score = float(f1_score(test_df[label_col].astype(str), pred, average="weighted", zero_division=0))
            if best is None or score > best["weighted_f1"]:
                best = {"feature": col, "weighted_f1": score}
        except Exception:
            continue
    if best and best["weighted_f1"] >= 0.95:
        best["weighted_f1"] = round(best["weighted_f1"], 4)
        best["detail"] = (
            f"{best['feature']} tek basina izole testte F1={best['weighted_f1']:.4f}; "
            "augmentation artisini olcmek icin gorev zaten tavana cok yakin."
        )
        return best
    return None


def evaluate_utility_protocol(train_df, validation_df, test_df, df_gen, label_col, numeric_cols):
    """Sentetik oranını validation ile seçer, nihai sonucu dokunulmamış testte ölçer."""
    common_numeric = [c for c in numeric_cols if c in train_df.columns and c in test_df.columns and c in df_gen.columns]
    classes = set(train_df[label_col].astype(str).unique())
    if len(classes) < 2 or not common_numeric:
        return None, df_gen
    gen_valid = df_gen[df_gen[label_col].astype(str).isin(classes)].reset_index(drop=True)
    if gen_valid.empty:
        return None, df_gen

    baseline_val = _fit_task_predict(train_df, validation_df, label_col, common_numeric)
    minority_class = baseline_val["minority_class"]
    minority_real_count = int((train_df[label_col].astype(str) == minority_class).sum())
    recipe_specs = [("seed_only", "none", 0)]
    for factor in (0.5, 1.0, 2.0, 4.0):
        recipe_specs.append((f"minority_{factor:g}x", "minority", max(1, int(minority_real_count * factor))))
    for ratio in (0.10, 0.25, 0.50, 1.0):
        recipe_specs.append((f"mixed_{ratio:g}x", "mixed", max(1, int(len(train_df) * ratio))))

    trials = []
    best = None
    for index, (name, mode, count) in enumerate(recipe_specs):
        synthetic = _sample_synthetic_recipe(gen_valid, label_col, minority_class, count, mode, index)
        candidate_train = train_df if synthetic.empty else pd.concat([train_df, synthetic], ignore_index=True)
        metrics = _fit_task_predict(candidate_train, validation_df, label_col, common_numeric)
        objective = (metrics["macro_f1"] * 0.55) + (metrics["minority_recall"] * 0.35) + (metrics["minority_precision"] * 0.10)
        trial = {
            "recipe": name, "synthetic_rows": int(len(synthetic)),
            "macro_f1": round(metrics["macro_f1"], 4),
            "weighted_f1": round(metrics["weighted_f1"], 4),
            "minority_recall": round(metrics["minority_recall"], 4),
            "minority_precision": round(metrics["minority_precision"], 4),
            "objective": round(objective, 4),
        }
        trials.append(trial)
        if best is None or objective > best["objective_raw"]:
            best = {"recipe": name, "synthetic": synthetic, "objective_raw": objective, "validation": metrics}

    selected = best["synthetic"] if best is not None else gen_valid.iloc[0:0].copy()
    full_real_train = pd.concat([train_df, validation_df], ignore_index=True)
    baseline_test = _fit_task_predict(full_real_train, test_df, label_col, common_numeric)
    dominant_feature = _dominant_feature_diagnostic(full_real_train, test_df, label_col, common_numeric)
    augmented_train = full_real_train if selected.empty else pd.concat([full_real_train, selected], ignore_index=True)
    augmented_test = _fit_task_predict(augmented_train, test_df, label_col, common_numeric)
    seed_f1 = baseline_test["weighted_f1"]
    augmented_f1 = augmented_test["weighted_f1"]
    improvement = round(((augmented_f1 - seed_f1) / max(seed_f1, 1e-8)) * 100, 1)
    macro_improvement = round(
        ((augmented_test["macro_f1"] - baseline_test["macro_f1"]) / max(baseline_test["macro_f1"], 1e-8)) * 100, 1
    )
    max_f1_improvement = round(((1.0 - seed_f1) / max(seed_f1, 1e-8)) * 100, 1)
    f1_target_met = improvement >= float(SYNTHETIC_TARGET_F1_IMPROVEMENT)
    f1_target_applicable = max_f1_improvement >= float(SYNTHETIC_TARGET_F1_IMPROVEMENT)
    recall_target_met = augmented_test["minority_recall"] >= float(SYNTHETIC_TARGET_MINORITY_RECALL)
    recall_already_met = baseline_test["minority_recall"] >= float(SYNTHETIC_TARGET_MINORITY_RECALL)
    accepted_for_utility = not selected.empty
    utility = {
        "evaluable": True,
        "protocol": "Leakage-free train/validation/test; generator sadece train verisini gordu.",
        "split_rows": {"train": int(len(train_df)), "validation": int(len(validation_df)), "test": int(len(test_df))},
        "selected_recipe": best["recipe"],
        "selected_synthetic_rows": int(len(selected)),
        "candidate_generated_rows": int(len(gen_valid)),
        "accepted_for_utility": accepted_for_utility,
        "dominant_feature_diagnostic": dominant_feature,
        "selection_trials": trials,
        "minority_class": minority_class,
        "minority_recall_seed": round(baseline_test["minority_recall"], 4),
        "minority_recall_augmented": round(augmented_test["minority_recall"], 4),
        "minority_precision_seed": round(baseline_test["minority_precision"], 4),
        "minority_precision_augmented": round(augmented_test["minority_precision"], 4),
        "minority_pr_auc_seed": round(baseline_test.get("minority_pr_auc", 0), 4),
        "minority_pr_auc_augmented": round(augmented_test.get("minority_pr_auc", 0), 4),
        "macro_f1_seed": round(baseline_test["macro_f1"], 4),
        "macro_f1_augmented": round(augmented_test["macro_f1"], 4),
        "macro_improvement": macro_improvement,
        "f1_target_met": f1_target_met,
        "f1_target_applicable": f1_target_applicable,
        "f1_max_possible_improvement": max_f1_improvement,
        "f1_target_status": (
            "met" if f1_target_met else "not_applicable_ceiling" if not f1_target_applicable else "not_met"
        ),
        "recall_target_met": recall_target_met,
        "recall_already_met": recall_already_met,
        "recall_target_status": "already_met" if recall_already_met else "met" if recall_target_met else "not_met",
        "f1_target": f">={SYNTHETIC_TARGET_F1_IMPROVEMENT}% improvement",
        "recall_target": f">={int(SYNTHETIC_TARGET_MINORITY_RECALL * 100)}% minority recall",
    }
    fidelity = _compute_fidelity_report(train_df, selected if not selected.empty else gen_valid, common_numeric, label_col, list(classes))
    if accepted_for_utility:
        analysis_note = (
            f"Utility protokolu: {best['recipe']} recetesi validation ile secildi; "
            "final skor izole test setinden geldi."
        )
    elif not f1_target_applicable:
        dominant_text = (
            f" Baskin sinyal: {dominant_feature['feature']} tek basina F1={dominant_feature['weighted_f1']:.4f}."
            if dominant_feature else ""
        )
        analysis_note = (
            f"Seed model izole testte F1={seed_f1:.4f} ile tavanda; +%{SYNTHETIC_TARGET_F1_IMPROVEMENT:g} "
            f"artis bu veri icin matematiksel olarak mumkun degil. Sentetik adaylar utility modeline eklenmedi.{dominant_text}"
        )
    else:
        analysis_note = (
            "Sentetik adaylar validation setinde seed modelden daha faydali olmadi; "
            "final utility karsilastirmasinda seed_only recetesi korundu."
        )
    scores = {
        "seed_f1": round(seed_f1, 4),
        "augmented_f1": round(augmented_f1, 4),
        "improvement": improvement,
        "analysis_note": analysis_note,
        "fidelity": fidelity,
        "utility": utility,
    }
    return scores, selected


def _score_from_corr(value):
    try:
        return round(max(0.0, min(1.0, float(value))) * 100, 1)
    except:
        return 0.0

def _distribution_shift_report(fidelity):
    details = fidelity.get("column_details", []) if isinstance(fidelity, dict) else []
    warnings = []
    smds = []
    mean_diffs = []
    for item in details:
        mean_diff = float(item.get("mean_diff_pct", 0) or 0)
        smd = float(item.get("standardized_mean_diff", 0) or 0)
        mean_diffs.append(mean_diff)
        smds.append(smd)

        if smd >= 0.5:
            severity = "kritik"
        elif smd >= 0.25:
            severity = "uyarı"
        else:
            continue

        warnings.append({
            "column": item.get("column"),
            "mean_diff_pct": round(mean_diff, 1),
            "standardized_mean_diff": round(smd, 3),
            "severity": severity,
            "detail": f"{item.get('column')} ortalaması {round(smd,3)} standart sapma kaydı."
        })
    avg_smd = float(np.mean(smds)) if smds else 0.0
    max_smd = float(np.max(smds)) if smds else 0.0
    avg_mean_shift = float(np.mean(mean_diffs)) if mean_diffs else 0.0
    max_mean_shift = float(np.max(mean_diffs)) if mean_diffs else 0.0
    # Sıfır merkezli sensörlerde yüzde fark yanıltır; skor Cohen's d/SMD üstünden verilir.
    # 0.1 küçük, 0.25 orta uyarı, 0.5+ ciddi dağılım kayması kabul edilir.
    score = round(max(0.0, 100.0 - min(avg_smd, 0.75) * (100.0 / 0.75)), 1)
    return {"score": score, "avg_standardized_mean_diff": round(avg_smd, 3),
            "max_standardized_mean_diff": round(max_smd, 3),
            "avg_mean_shift_pct": round(avg_mean_shift, 1),
            "max_mean_shift_pct": round(max_mean_shift, 1), "warnings": warnings}

def _physical_consistency_report(df_gen):
    waymo_cols = [f'{c}({i+1})' for c in ['x','y','speed','vx','vy'] for i in range(20)]
    if df_gen.empty or not all(c in df_gen.columns for c in waymo_cols):
        return {"applicable": False, "score": None, "detail": "Waymo/trajectory kolonları yok; fizik skoru uygulanmadı."}
    
    speed = df_gen[[f'speed({i+1})' for i in range(20)]].to_numpy(dtype=float)
    vx = df_gen[[f'vx({i+1})' for i in range(20)]].to_numpy(dtype=float)
    vy = df_gen[[f'vy({i+1})' for i in range(20)]].to_numpy(dtype=float)
    x = df_gen[[f'x({i+1})' for i in range(20)]].to_numpy(dtype=float)
    y = df_gen[[f'y({i+1})' for i in range(20)]].to_numpy(dtype=float)
    
    finite_ratio = float(np.isfinite(speed).mean() * np.isfinite(vx).mean() * np.isfinite(vy).mean())
    negative_speed_ratio = float((speed < -1e-6).mean())
    high_speed_ratio = float((speed > 60).mean())
    accel = np.diff(speed, axis=1) / 0.1
    high_accel_ratio = float((np.abs(accel) > 12).mean()) if accel.size else 0.0
    lateral_step = np.sqrt(np.diff(x, axis=1)**2 + np.diff(y, axis=1)**2)
    jump_ratio = float((lateral_step > 8).mean()) if lateral_step.size else 0.0
    vel_mag = np.sqrt(vx**2 + vy**2)
    coherence_error = np.abs(vel_mag - speed) / (np.abs(speed) + 1.0)
    coherence_penalty = float(np.clip(np.nanmean(coherence_error), 0, 1))
    
    score = 100.0
    score -= negative_speed_ratio * 100
    score -= high_speed_ratio * 80
    score -= high_accel_ratio * 70
    score -= jump_ratio * 70
    score -= coherence_penalty * 20
    score *= finite_ratio
    score = round(max(0.0, min(100.0, score)), 1)
    
    issues = []
    if negative_speed_ratio > 0: issues.append("Negatif hız tespit edildi.")
    if high_speed_ratio > 0: issues.append("60 m/s üzeri hız tespit edildi.")
    if high_accel_ratio > 0.02: issues.append("Yüksek ivme oranı arttı.")
    if jump_ratio > 0.02: issues.append("Yörüngede ani konum sıçraması var.")
    if coherence_penalty > 0.25: issues.append("speed ile vx/vy büyüklüğü arasında uyumsuzluk var.")
    
    return {
        "applicable": True,
        "score": score,
        "negative_speed_ratio": round(negative_speed_ratio, 4),
        "high_speed_ratio": round(high_speed_ratio, 4),
        "high_accel_ratio": round(high_accel_ratio, 4),
        "jump_ratio": round(jump_ratio, 4),
        "velocity_coherence_error": round(coherence_penalty, 4),
        "issues": issues,
        "formula": "100 - negatif hız, aşırı hız, |ivme|>12 m/s², ani konum sıçraması ve speed-vx/vy uyumsuzluğu cezaları"
    }

def _compute_fidelity_report(df_orig, df_gen, numeric_cols, label_col=None, valid_classes=None):
    """
    Fidelity sınıflandırmadan bağımsızdır: label tek sınıflı olsa bile
    orijinal-sentetik dağılım benzerliği ölçülebilir.
    """
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim

    common_cols = [c for c in numeric_cols if c in df_orig.columns and c in df_gen.columns]
    if not common_cols or df_orig.empty or df_gen.empty:
        return {
            "applicable": False,
            "cosine_similarity": None,
            "column_correlation": None,
            "column_details": [],
            "reason": "Orijinal ve sentetik veri arasında ortak sayısal kolon bulunamadı."
        }

    orig_frame = df_orig
    gen_frame = df_gen
    if label_col and valid_classes is not None and label_col in df_orig.columns and label_col in df_gen.columns:
        orig_frame = df_orig[df_orig[label_col].astype(str).isin([str(c) for c in valid_classes])]
        gen_frame = df_gen[df_gen[label_col].astype(str).isin([str(c) for c in valid_classes])]

    if orig_frame.empty:
        orig_frame = df_orig
    if gen_frame.empty:
        gen_frame = df_gen

    X = np.nan_to_num(orig_frame[common_cols].to_numpy(dtype=np.float32))
    Xg = np.nan_to_num(gen_frame[common_cols].to_numpy(dtype=np.float32))
    if X.size == 0 or Xg.size == 0:
        return {
            "applicable": False,
            "cosine_similarity": None,
            "column_correlation": None,
            "column_details": [],
            "reason": "Fidelity için yeterli sayısal değer yok."
        }

    orig_means = np.mean(X, axis=0)
    gen_means = np.mean(Xg, axis=0)
    orig_stds = np.std(X, axis=0) + 1e-8
    gen_stds = np.std(Xg, axis=0) + 1e-8

    try:
        cosine_val = float(cos_sim(orig_means.reshape(1, -1), gen_means.reshape(1, -1))[0][0])
    except Exception:
        cosine_val = 0.0

    if len(common_cols) > 1:
        mean_corr = float(np.corrcoef(orig_means, gen_means)[0, 1])
        std_corr = float(np.corrcoef(orig_stds, gen_stds)[0, 1])
    else:
        mean_corr = 1.0
        std_corr = 1.0

    if np.isnan(mean_corr):
        mean_corr = 0.0
    if np.isnan(std_corr):
        std_corr = 0.0

    col_details = []
    for j, col in enumerate(common_cols[:20]):
        pooled_std = float(np.sqrt((orig_stds[j] ** 2 + gen_stds[j] ** 2) / 2.0)) + 1e-8
        mean_diff_abs = abs(float(orig_means[j]) - float(gen_means[j]))
        col_details.append({
            "column": col,
            "orig_mean": round(float(orig_means[j]), 4),
            "gen_mean": round(float(gen_means[j]), 4),
            "orig_std": round(float(orig_stds[j]), 4),
            "gen_std": round(float(gen_stds[j]), 4),
            "mean_diff_pct": round(mean_diff_abs / (abs(float(orig_means[j])) + 1e-8) * 100, 1),
            "standardized_mean_diff": round(mean_diff_abs / pooled_std, 4)
        })

    return {
        "applicable": True,
        "cosine_similarity": round(max(0.0, min(1.0, cosine_val)), 4),
        "mean_correlation": round(max(0.0, min(1.0, mean_corr)), 4),
        "std_correlation": round(max(0.0, min(1.0, std_corr)), 4),
        "column_correlation": round(max(0.0, min(1.0, (mean_corr + std_corr) / 2.0)), 4),
        "column_details": col_details,
        "common_numeric_cols": len(common_cols),
        "compared_rows": {"seed": int(len(orig_frame)), "synthetic": int(len(gen_frame))}
    }

def _quality_context(df_orig, df_gen, method, is_waymo, label_col, numeric_cols):
    class_counts = {}
    if label_col and label_col in df_orig.columns:
        class_counts = {str(k): int(v) for k, v in df_orig[label_col].astype(str).value_counts().to_dict().items()}

    valid_classes = [k for k, v in class_counts.items() if v >= 4]
    if is_waymo or is_waymo_frame(df_gen):
        task_type = "trajectory"
        primary_goal = "Fiziksel tutarlılık + yörünge dağılım benzerliği"
    elif len(class_counts) >= 2:
        task_type = "classification"
        primary_goal = "Sınıflandırma faydası + dağılım benzerliği"
    elif numeric_cols:
        task_type = "tabular_unsupervised"
        primary_goal = "Dağılım benzerliği + sentetik veri gerçekçiliği"
    else:
        task_type = "unknown"
        primary_goal = "Veri tanıma için yeterli yapı bulunamadı"

    utility_reason = None
    if len(class_counts) == 0:
        utility_reason = "Label kolonu bulunamadığı için classification utility uygulanmadı."
    elif len(class_counts) == 1:
        utility_reason = "Label tek sınıflı; F1/Recall için en az iki sınıf gerekir."
    elif len(valid_classes) < 2:
        utility_reason = "En az iki sınıfta 4+ örnek olmadığı için stratified utility testi güvenilir değil."

    return {
        "task_type": task_type,
        "primary_goal": primary_goal,
        "label_col": label_col,
        "class_count": int(len(class_counts)),
        "class_distribution": class_counts,
        "valid_utility_classes": valid_classes,
        "numeric_feature_count": int(len(numeric_cols)),
        "seed_rows": int(len(df_orig)),
        "synthetic_rows": int(len(df_gen)),
        "utility_reason": utility_reason,
        "method": method,
    }

def _component(score, weight, applicable=True, reason=None, detail=None):
    data = {"score": score if applicable else None, "weight": weight if applicable else 0, "applicable": bool(applicable)}
    if reason:
        data["reason"] = reason
    if detail:
        data["detail"] = detail
    return data

def build_quality_report(df_orig, df_gen, method, is_waymo, scores, label_col, numeric_cols):
    context = _quality_context(df_orig, df_gen, method, is_waymo, label_col, numeric_cols)
    fidelity = scores.get("fidelity", {}) or {}
    utility = scores.get("utility", {}) or {}
    fidelity_applicable = fidelity.get("applicable", True) is not False and (
        bool(fidelity.get("column_details")) or fidelity.get("cosine_similarity") is not None
    )
    fidelity_score = round((_score_from_corr(fidelity.get("cosine_similarity", 0) or 0) +
                            _score_from_corr(fidelity.get("column_correlation", 0) or 0)) / 2, 1)
    shift = _distribution_shift_report(fidelity)
    physical = _physical_consistency_report(df_gen)
    
    utility_applicable = utility.get("evaluable", True) is not False and scores.get("seed_f1", 0) > 0
    if utility_applicable:
        aug_f1 = float(scores.get("augmented_f1", 0) or 0)
        improvement = max(0.0, float(scores.get("improvement", 0) or 0))
        recall_aug = float(utility.get("minority_recall_augmented", utility.get("minority_recall", 0)) or 0)
        improvement_credit = 1.0 if utility.get("f1_target_status") == "not_applicable_ceiling" else min(
            improvement / float(SYNTHETIC_TARGET_F1_IMPROVEMENT), 1.0
        )
        utility_score = round((aug_f1 * 55) + (improvement_credit * 25) + (recall_aug * 20), 1)
    else:
        utility_score = None
    
    components = {}
    if fidelity_applicable:
        components["fidelity"] = _component(
            fidelity_score,
            0.35 if method != "rcgan" else 0.30,
            detail="Orijinal ve sentetik sayısal kolonların ortalama/std benzerliği ölçüldü."
        )
    else:
        components["fidelity"] = _component(
            None, 0, False,
            fidelity.get("reason", "Ortak sayısal kolon bulunamadığı için fidelity uygulanmadı.")
        )
    
    distribution_applicable = bool(shift.get("warnings")) or bool(fidelity.get("column_details"))
    if distribution_applicable:
        components["distribution"] = _component(
            shift["score"],
            0.25 if method != "rcgan" else 0.10,
            detail="Kolon bazlı standardized mean difference ile dağılım kayması ölçüldü."
        )
    else:
        components["distribution"] = _component(
            None, 0, False,
            "Dağılım kayması için kolon detayı üretilemedi."
        )
    
    if utility_score is not None:
        components["utility"] = _component(
            utility_score,
            0.35 if method != "rcgan" else 0.25,
            detail="Aynı test ayrımında seed model ile augmented model karşılaştırıldı."
        )
    else:
        components["utility"] = _component(
            None, 0, False,
            context.get("utility_reason") or scores.get("analysis_note") or "Utility metriği bu veri yapısı için güvenilir değil."
        )
    if physical["applicable"]:
        components["physical"] = _component(
            physical["score"],
            0.35 if method == "rcgan" else 0.15,
            detail="Hız, ivme, konum sıçraması ve speed-vx/vy tutarlılığı ölçüldü."
        )
    else:
        components["physical"] = _component(None, 0, False, physical.get("detail"))
    
    total_weight = sum(v["weight"] for v in components.values())
    overall = round(sum((v["score"] or 0) * v["weight"] for v in components.values()) / max(total_weight, 1e-8), 1) if total_weight > 0 else 0
    
    if method == "rcgan":
        routing = "RCGAN seçildi: veri Waymo formatında veya otonom araç/yörünge kolonlarından 20 adımlı trajektöre dönüştürülebildi."
    elif method == "ctgan":
        routing = "CTGAN seçildi: veri genel tabular/sensör formatında ve çok sütunlu dağılım korunmalı."
    else:
        routing = "SMOTE+Gaussian seçildi: veri CTGAN için küçük/uygunsuz veya CTGAN başarısız oldu."

    active_components = [
        name for name, data in components.items()
        if data.get("applicable") and data.get("score") is not None
    ]
    skipped_components = [
        {"name": name, "reason": data.get("reason", "Uygulanamaz")}
        for name, data in components.items()
        if not data.get("applicable")
    ]
    recommendations = []
    if not utility_applicable:
        recommendations.append(context.get("utility_reason") or "Classification metriği için label yapısını kontrol et.")
        recommendations.append("F1/Recall istiyorsan label kolonunda en az iki sınıf ve her sınıfta 4+ örnek olmalı.")
    if physical["applicable"] and physical["score"] is not None and physical["score"] < 90:
        recommendations.append("Fizik skorunu yükseltmek için hız/ivme sınırlarını daha sıkı filtrele veya RCGAN sonrası yörünge smoothing uygula.")
    if shift.get("warnings"):
        recommendations.append("Dağılım kayması olan kolonlarda sentetik üretimi yeniden ağırlıklandır veya daha fazla seed örnek kullan.")
    if fidelity_applicable and fidelity_score < 80:
        recommendations.append("Fidelity düşükse CTGAN/RCGAN üretim sayısını ve eğitim epoch/seed çeşitliliğini artır.")
    if utility.get("f1_target_status") == "not_applicable_ceiling":
        recommendations.append(
            "Seed F1 tavan seviyede: sentetik faydayı göstermek için tek bir sensör eşiğiyle çözülmeyen daha zor bir test görevi kullan."
        )
    if not recommendations:
        recommendations.append("Aktif metriklerde kritik problem görünmüyor; daha güçlü kanıt için daha çeşitli seed veriyle tekrar ölç.")
    
    return {
        "overall_score": overall,
        "grade": "A" if overall >= 90 else "B" if overall >= 80 else "C" if overall >= 70 else "D",
        "method": method,
        "context": context,
        "routing_explanation": routing,
        "score_explanation": (
            f"Bu veri için genel skor sadece uygulanabilir metriklerden hesaplandı: "
            f"{', '.join(active_components) if active_components else 'uygulanabilir metrik yok'}."
        ),
        "active_components": active_components,
        "skipped_components": skipped_components,
        "recommendations": recommendations,
        "components": components,
        "distribution_shift": shift,
        "physical_consistency": physical,
        "scientific_basis": [
            "Fidelity: orijinal ve sentetik verinin ortalama vektör cosine benzerliği ile kolon ortalama/std korelasyonlarının birleşimi.",
            f"Utility: aynı test ayrımı üzerinde Seed F1, Augmented F1, F1 iyileşmesi (hedef %{SYNTHETIC_TARGET_F1_IMPROVEMENT}) ve azınlık recall değişimi (hedef %{int(SYNTHETIC_TARGET_MINORITY_RECALL * 100)}).",
            "Distribution shift: kolon bazlı orijinal-sentetik ortalama fark yüzdesi; %15 uyarı, %30 kritik eşik.",
            "Physical consistency: RCGAN yörüngelerinde hız pozitifliği, hız sınırı, ivme, konum sıçraması ve speed-vx/vy uyumu."
        ]
    }

# ═══════════════ EVALUATION (Fidelity + Utility) ═══════════════
def evaluate(df_orig, df_gen, label_col, numeric_cols):
    """
    Akademik standartlara uygun Fidelity + Utility değerlendirmesi.
    
    Fidelity (Dağılım Benzerliği):
      - Cosine Similarity: Orijinal vs sentetik özellik vektörleri
      - Sütun bazlı ortalama/std karşılaştırması
      - Dağılım örtüşme oranı
    
    Utility (Görev Faydası):
      - Seed F1 vs Augmented F1 (weighted)
      - Sınıf bazlı F1 skorları
      - Azınlık sınıfı recall (hedef: %80)
      - F1 iyileştirme yüzdesi (hedef: %15)
    """
    from sklearn.metrics import f1_score, recall_score, precision_recall_curve, auc, classification_report
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim
    
    X = np.nan_to_num(df_orig[numeric_cols].values.astype(np.float32))
    y_raw = df_orig[label_col].astype(str)
    
    # Az üyeli sınıfları filtrele (stratify için min 4 gerekli)
    counts = y_raw.value_counts()
    valid_classes = counts[counts >= 4].index.tolist()
    fidelity = _compute_fidelity_report(df_orig, df_gen, numeric_cols, label_col, valid_classes)
    mask = y_raw.isin(valid_classes)
    X = X[mask]; y_raw = y_raw[mask].reset_index(drop=True)

    if len(valid_classes) < 2 or len(y_raw) < 8:
        return {"seed_f1": 0, "augmented_f1": 0, "improvement": 0,
                "analysis_note": "Utility metrikleri için en az iki sınıfta yeterli örnek gerekir.",
                "fidelity": fidelity,
                "utility": {"evaluable": False, "class_count": int(len(valid_classes)),
                            "reason": "En az iki sınıfta 4+ örnek yok; F1/Recall güvenilir hesaplanamaz.",
                            "minority_recall": 0, "f1_target_met": False, "recall_target_met": False}}
    
    le = LabelEncoder()
    le.fit(valid_classes)
    y = le.transform(y_raw)
    
    if len(np.unique(y)) < 2:
        return {"seed_f1": 0, "augmented_f1": 0, "improvement": 0,
                "analysis_note": "Utility metrikleri için en az iki sınıf gerekir; bu veri tek sınıflı.",
                "fidelity": fidelity,
                "utility": {"evaluable": False, "class_count": int(len(np.unique(y))),
                            "reason": "Tek sınıflı veri classification utility için uygun değil.",
                            "minority_recall": 0, "f1_target_met": False, "recall_target_met": False}}
    
    # ══ SEED MODEL (Orijinal veriyle) ══
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=RANDOM_SEED, stratify=y)
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr)
    Xte_s = sc.transform(Xte)
    
    seed_counts = np.bincount(ytr, minlength=len(le.classes_)).astype(np.float64)
    seed_counts[seed_counts == 0] = 1.0
    seed_weights = (len(ytr) / (len(le.classes_) * seed_counts))[ytr]
    m1 = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=RANDOM_SEED).fit(
        Xtr_s, ytr, sample_weight=seed_weights
    )
    pred_seed = m1.predict(Xte_s)
    f1s = float(f1_score(yte, pred_seed, average='weighted', zero_division=0))
    
    # Sınıf bazlı F1 (seed)
    f1_per_class_seed = {}
    for i, cls_name in enumerate(le.classes_):
        cls_mask = yte == i
        if cls_mask.sum() > 0:
            cls_f1 = float(f1_score(yte == i, pred_seed == i, zero_division=0))
            f1_per_class_seed[str(cls_name)] = round(cls_f1, 4)
    
    # Azınlık sınıfı recall (seed) — en az üyesi olan sınıf
    minority_class = counts[counts.index.isin(valid_classes)].idxmin()
    minority_idx = le.transform([str(minority_class)])[0]
    minority_mask_te = yte == minority_idx
    if minority_mask_te.sum() > 0:
        minority_recall_seed = float(recall_score(minority_mask_te, pred_seed == minority_idx, zero_division=0))
    else:
        minority_recall_seed = 0.0
    
    # ══ FIDELITY (Dağılım Benzerliği) ══
    gc = [c for c in numeric_cols if c in df_gen.columns]
    
    if gc and label_col in df_gen.columns:
        gen_mask = df_gen[label_col].astype(str).isin(valid_classes)
        df_gen_filtered = df_gen[gen_mask]
        
        if len(df_gen_filtered) > 0:
            Xg = np.nan_to_num(df_gen_filtered[gc].values.astype(np.float32))
            
            # 1. Cosine Similarity: Orijinal ve sentetik ortalama vektörleri
            orig_mean = np.mean(X[:, :len(gc)], axis=0).reshape(1, -1)
            gen_mean = np.mean(Xg, axis=0).reshape(1, -1)
            cosine_val = float(cos_sim(orig_mean, gen_mean)[0][0])
            
            # 2. Sütun bazlı ortalama/std korelasyonu
            orig_means = np.mean(X[:, :len(gc)], axis=0)
            gen_means = np.mean(Xg, axis=0)
            orig_stds = np.std(X[:, :len(gc)], axis=0) + 1e-8
            gen_stds = np.std(Xg, axis=0) + 1e-8
            
            # Ortalama korelasyonu
            if len(orig_means) > 1:
                mean_corr = float(np.corrcoef(orig_means, gen_means)[0, 1])
                std_corr = float(np.corrcoef(orig_stds, gen_stds)[0, 1])
            else:
                mean_corr = 1.0
                std_corr = 1.0
            
            # 3. Sütun detayları (ilk 10)
            col_details = []
            for j, col in enumerate(gc[:10]):
                pooled_std = float(np.sqrt((orig_stds[j] ** 2 + gen_stds[j] ** 2) / 2.0)) + 1e-8
                mean_diff_abs = abs(float(orig_means[j]) - float(gen_means[j]))
                col_details.append({
                    "column": col,
                    "orig_mean": round(float(orig_means[j]), 4),
                    "gen_mean": round(float(gen_means[j]), 4),
                    "orig_std": round(float(orig_stds[j]), 4),
                    "gen_std": round(float(gen_stds[j]), 4),
                    "mean_diff_pct": round(mean_diff_abs / (abs(float(orig_means[j])) + 1e-8) * 100, 1),
                    "standardized_mean_diff": round(mean_diff_abs / pooled_std, 4)
                })
            
            # 4. Histogram Verileri (Waymo için özel: x(10), y(10), speed(10), vx(10))
            histograms = {}
            target_cols = [c for c in ['x(10)', 'y(10)', 'speed(10)', 'vx(10)'] if c in gc]
            for col in target_cols:
                col_idx = gc.index(col)
                orig_data = X[:, col_idx]
                gen_data = Xg[:, col_idx]
                
                # Sınırları belirle
                min_val = min(np.min(orig_data), np.min(gen_data))
                max_val = max(np.max(orig_data), np.max(gen_data))
                bins = np.linspace(min_val, max_val, 50)
                
                # Orijinal histogram (yoğunluk)
                orig_hist, _ = np.histogram(orig_data, bins=bins, density=True)
                # Sentetik histogram (yoğunluk)
                gen_hist, _ = np.histogram(gen_data, bins=bins, density=True)
                
                # NumPy tiplerini native listelere çevir
                histograms[col] = {
                    "bins": [(bins[i] + bins[i+1])/2 for i in range(len(bins)-1)],
                    "orig_hist": [float(h) if not np.isnan(h) else 0.0 for h in orig_hist],
                    "gen_hist": [float(h) if not np.isnan(h) else 0.0 for h in gen_hist]
                }
            
            fidelity = {
                "cosine_similarity": round(cosine_val, 4),
                "mean_correlation": round(mean_corr, 4) if not np.isnan(mean_corr) else 0,
                "std_correlation": round(std_corr, 4) if not np.isnan(std_corr) else 0,
                "column_correlation": round((mean_corr + std_corr) / 2, 4) if not (np.isnan(mean_corr) or np.isnan(std_corr)) else 0,
                "column_details": col_details,
                "histograms": histograms
            }
            
            # ══ AUGMENTED MODEL (Orijinal + Sentetik veriyle) ══
            yg = le.transform(df_gen_filtered[label_col].astype(str))
            Xa = np.vstack([Xtr, Xg])
            ya = np.concatenate([ytr, yg])
            
            aug_counts = np.bincount(ya, minlength=len(le.classes_)).astype(np.float64)
            aug_counts[aug_counts == 0] = 1.0
            aug_weights = (len(ya) / (len(le.classes_) * aug_counts))[ya]
            m2 = GradientBoostingClassifier(n_estimators=150, max_depth=5, random_state=RANDOM_SEED).fit(
                sc.transform(Xa), ya, sample_weight=aug_weights
            )
            pred_aug = m2.predict(Xte_s)
            f1a = float(f1_score(yte, pred_aug, average='weighted', zero_division=0))
            
            # Sınıf bazlı F1 (augmented)
            f1_per_class_aug = {}
            for i, cls_name in enumerate(le.classes_):
                cls_f1 = float(f1_score(yte == i, pred_aug == i, zero_division=0))
                f1_per_class_aug[str(cls_name)] = round(cls_f1, 4)
            
            # Azınlık recall (augmented)
            if minority_mask_te.sum() > 0:
                minority_recall_aug = float(recall_score(minority_mask_te, pred_aug == minority_idx, zero_division=0))
            else:
                minority_recall_aug = 0.0
            
            # ══ HEDEF KONTROL ══
            improvement = round(((f1a - f1s) / (f1s + 1e-6)) * 100, 1)
            f1_target_met = improvement >= float(SYNTHETIC_TARGET_F1_IMPROVEMENT)
            recall_target_met = minority_recall_aug >= float(SYNTHETIC_TARGET_MINORITY_RECALL)
            
            return {
                "seed_f1": round(f1s, 4),
                "augmented_f1": round(f1a, 4),
                "improvement": improvement,
                "fidelity": fidelity,
                "utility": {
                    "f1_per_class_seed": f1_per_class_seed,
                    "f1_per_class_augmented": f1_per_class_aug,
                    "minority_class": str(minority_class),
                    "minority_recall_seed": round(minority_recall_seed, 4),
                    "minority_recall_augmented": round(minority_recall_aug, 4),
                    "f1_target_met": f1_target_met,
                    "recall_target_met": recall_target_met,
                    "f1_target": f">={SYNTHETIC_TARGET_F1_IMPROVEMENT}% improvement",
                    "recall_target": f">={int(SYNTHETIC_TARGET_MINORITY_RECALL * 100)}% minority recall"
                }
            }
    
    return {"seed_f1": round(f1s, 4), "augmented_f1": round(f1s, 4), "improvement": 0,
            "fidelity": fidelity,
            "utility": {"minority_recall": round(minority_recall_seed, 4),
                        "f1_target_met": False, "recall_target_met": False}}

# ═══════════════ API ═══════════════
@app.get("/api/system_status")
async def system_status():
    seed_path = get_seed_path()
    se = seed_path is not None
    active_model_path = get_local_asset_path(MODEL_PATH)
    v2_model_path = get_v2_model_path()
    return {"model_loaded":rcgan is not None,"model_name":os.path.basename(active_model_path or MODEL_PATH),
        "model_path":active_model_path,
        "model_version":MODEL_VERSION,
        "model_lazy_load": RCGAN_LAZY_LOAD,
        "model_load_error": RCGAN_LOAD_ERROR,
        "baseline_model":os.path.basename(BASELINE_MODEL_PATH),
        "v2_model":os.path.basename(V2_MODEL_PATH),
        "v2_available":v2_model_path is not None,
        "v2_path":v2_model_path,
        "seed_available":se,"seed_rows":count_csv_rows_cached_or_fast(seed_path),
        "seed_rows_note":"Büyük seed dosyalarında satır sayısı full automation sırasında hesaplanır.",
        "seed_file":os.path.basename(seed_path) if seed_path else None,
        "model_lfs_missing": active_model_path is None,
        "seed_lfs_missing": not se,
        "model_source_placeholder": is_git_lfs_pointer(MODEL_PATH) or is_dataless_file(MODEL_PATH),
        "seed_source_placeholder": is_git_lfs_pointer(SEED_PATH) or is_dataless_file(SEED_PATH),
        "status":"ready","mode":"adaptive+distillation",
        "limits":{"max_upload_mb":MAX_UPLOAD_MB,"ctgan_max_rows":CTGAN_MAX_ROWS},
        "capabilities":{
            "schema_profiler": True,
            "schema_aware_ctgan": is_ctgan_available(load=False),
            "borderline_smote": True,
            "synthetic_validator": True,
            "rcgan_postprocess": True,
            "rcgan_anomaly_severity": RCGAN_ANOMALY_SEVERITY,
            "target_f1_improvement_pct": SYNTHETIC_TARGET_F1_IMPROVEMENT,
            "target_minority_recall": SYNTHETIC_TARGET_MINORITY_RECALL,
        }}

@app.post("/api/distill")
def distill_endpoint(file: UploadFile = File(...)):
    """Veri setini damıt ve rapor döndür"""
    try:
        df = read_csv_text_robust(read_csv_upload_sync(file))
        df_clean, report, label_col, numeric_cols = distill_dataset(df)
        
        class_dist = analyze_classes(df_clean, label_col) if label_col else {}
        schema_profile = profile_schema(df_clean, label_col)
        
        # Temiz veriyi kaydet
        clean_path = os.path.join(OUTPUT_DIR, "distilled_data.csv")
        safe_write_csv(df_clean, clean_path, index=False)
        
        res = {
            "status": "success",
            "report": report,
            "label_col": label_col,
            "n_features": len(numeric_cols),
            "schema_profile": schema_profile,
            "class_distribution": class_dist,
            "is_waymo": all(f'x({i+1})' in df.columns for i in range(20)),
        }
        import math
        def sanitize(obj):
            if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)): return 0.0
            if isinstance(obj, dict): return {k: sanitize(v) for k, v in obj.items()}
            if isinstance(obj, list): return [sanitize(v) for v in obj]
            return obj
        res = sanitize(res)
        safe_write_json(res, LATEST_ANALYSIS_PATH)
        return res
    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        print(err_msg)
        return JSONResponse(status_code=400, content={"detail": str(e)})

@app.post("/api/evaluate_pipeline")
def evaluate_pipeline(file: UploadFile = File(...), n_samples: int = Form(1000)):
    try:
        raw_text = read_csv_upload_sync(file)
        
        # ── CSV Onarım Motoru v4 ──
        lines = raw_text.splitlines()
        
        if len(lines) >= 2:
            header_fields = lines[0].split(',')
            data_fields = lines[1].split(',')
            n_hf = len(header_fields)
            
            if n_hf > 150:
                real_header = []
                for f in header_fields:
                    f = f.strip()
                    try:
                        float(f)
                        break
                    except ValueError:
                        if f:
                            real_header.append(f)
                
                last_field = header_fields[-1].strip().rstrip('\n').rstrip('\r')
                if last_field == 'label' and 'label' not in real_header:
                    real_header.append('label')
                
                n_cols = len(real_header)
                new_header = ','.join(real_header)
                
                new_lines = [new_header]
                for line in lines[1:]:
                    line = line.strip()
                    if not line:
                        continue
                    fields = line.split(',')
                    
                    if len(fields) > n_cols:
                        real_data = fields[:n_cols - 1]
                        label_val = ''
                        for fi in reversed(fields):
                            fi = fi.strip()
                            if fi:
                                try:
                                    float(fi)
                                except ValueError:
                                    label_val = fi
                                    break
                        real_data.append(label_val)
                        new_lines.append(','.join(real_data))
                    else:
                        new_lines.append(line)
                
                raw_text = '\n'.join(new_lines)
                print(f"[🔧] Bozuk CSV onarıldı: {n_hf} alan → {n_cols} sütun, {len(new_lines)-1} satır")
        
        df = read_csv_text_robust(raw_text)
        
        df.columns = [str(c).split('\n')[0].split('\r')[0].strip() for c in df.columns]
        
        import re
        fixed_cols = {}
        waymo_pattern = re.compile(r'^((?:x|y|speed|vx|vy)\(\d+\)).*$', re.DOTALL)
        for col in df.columns:
            m = waymo_pattern.match(col)
            if m and m.group(1) != col:
                fixed_cols[col] = m.group(1)
        if fixed_cols:
            df = df.rename(columns=fixed_cols)
        
        garbage_cols = []
        for col in df.columns:
            try:
                float(col)
                garbage_cols.append(col)
            except ValueError:
                pass
        if garbage_cols:
            df = df.drop(columns=garbage_cols)
        
        n_samples = max(10, min(n_samples, 10000))
        
        # 1. Damıt
        df_clean, report, label_col, numeric_cols = distill_dataset(df)
        
        if not label_col:
            df_clean['label'] = 'normal'
            label_col = 'label'
            print("[ℹ️] Yüklenen veride label sütunu bulunamadı, 'label' oluşturuldu.")
        
        safe_write_csv(df_clean, os.path.join(OUTPUT_DIR, "distilled_data.csv"), index=False)
        
        # 2. Üret: Etiketli genel veride generator test/validation setini asla görmez.
        utility_split = make_utility_split(df_clean, label_col)
        generation_source = utility_split[0] if utility_split else df_clean
        df_gen, method, is_waymo, generation_report = generate_adaptive(generation_source, label_col, numeric_cols, n_samples)
        if df_gen.empty:
            return JSONResponse(status_code=400, content={"detail":"Üretim başarısız."})
        
        # 3. Değerlendir
        if utility_split and not is_waymo:
            train_df, validation_df, test_df = utility_split
            scores, selected_gen = evaluate_utility_protocol(
                train_df, validation_df, test_df, df_gen, label_col, numeric_cols
            )
            if scores is not None:
                generation_report["utility_selection"] = scores["utility"]
                if not selected_gen.empty:
                    df_gen = selected_gen.reset_index(drop=True)
                    safe_write_csv(df_gen, os.path.join(OUTPUT_DIR, "live_synthetic_output.csv"), index=False)
                    generation_report["output_role"] = "utility_accepted"
                else:
                    generation_report["output_role"] = "candidate_only_not_selected_for_utility"
                df_eval, eval_label_col, eval_numeric_cols = train_df, label_col, numeric_cols
            else:
                df_eval, eval_label_col, eval_numeric_cols = df_clean, label_col, numeric_cols
        else:
            scores = None
            df_eval, eval_label_col, eval_numeric_cols = prepare_evaluation_frame(
                df_clean, df_gen, method, is_waymo, label_col, numeric_cols
            )
        eval_numeric_cols = [c for c in eval_numeric_cols if c in df_eval.columns and c in df_gen.columns]
        if not eval_numeric_cols:
            return JSONResponse(status_code=400, content={"detail":"Değerlendirme için ortak numerik kolon bulunamadı."})
        if scores is None:
            scores = evaluate(df_eval, df_gen, eval_label_col, eval_numeric_cols)
        quality_report = build_quality_report(
            df_eval, df_gen, method, is_waymo, scores, eval_label_col, eval_numeric_cols
        )
        
        safe_rows = max(report.get("clean_rows", 1), 1)
        res = {
            "status":"success", "method":method,
            "distillation": report,
            "dataset_info": {
                "rows":report.get("clean_rows", 0),"features":len(numeric_cols),
                "classes":len(analyze_classes(df_clean,label_col)),
                "label_col":label_col,"is_waymo":is_waymo,
                "class_distribution":analyze_classes(df_clean,label_col),
                "utility_protocol": "train_validation_test" if utility_split and not is_waymo else None,
            },
            "seed_count":report.get("clean_rows", 0),"gen_count":len(df_gen),
            "multiplication_factor":round((report.get("clean_rows", 0)+len(df_gen))/safe_rows,2),
            "generative_coverage":round(len(df_gen)/(safe_rows)*100,1),
            "generation_report": generation_report,
            "quality_report": quality_report,
            **scores,
        }
        
        import math
        def sanitize(obj):
            if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)): return 0.0
            if isinstance(obj, dict): return {k: sanitize(v) for k, v in obj.items()}
            if isinstance(obj, list): return [sanitize(v) for v in obj]
            return obj
            
        res = sanitize(res)
        safe_write_json(res, LATEST_ANALYSIS_PATH)
        return res
    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        print(err_msg)
        return JSONResponse(status_code=400, content={"detail": str(e)})


@app.post("/api/run_full_automation")
def run_full_automation(body: dict):
    try:
        n_samples = max(10, min(body.get("n_samples", 2000), 10000))
    except:
        n_samples = 2000
    
    seed_path = get_seed_path()
    if not seed_path:
        return JSONResponse(status_code=400, content={"detail":"Seed dosyası yok."})
    
    # 700K'lık devasa dosyadan rastgele 10.000 satır seçerek al
    # Tüm dosyayı RAM'e almamak için skiprows olasılık hesabı kullanıyoruz
    total_rows = count_csv_rows(seed_path)
    sample_size = 10000
    
    df = read_seed_dataframe(seed_path, total_rows, sample_size=sample_size)
        
    df_clean, report, label_col, numeric_cols = distill_dataset(df)
    
    # Waymo seed'inde genellikle label sütunu yoktur (hepsi normaldir). 
    # Eğer yoksa, pipeline'ın çökmemesi için biz ekliyoruz.
    if not label_col:
        df_clean['label'] = 'normal'
        label_col = 'label'
        print("[ℹ️] Label sütunu bulunamadı, 'label' adında 'normal' değerlerle oluşturuldu.")
    
    df_gen, method, is_waymo, generation_report = generate_adaptive(df_clean, label_col, numeric_cols, n_samples)
    if df_gen.empty:
        return JSONResponse(status_code=500, content={"detail":"Üretim başarısız."})
    
    df_eval, eval_label_col, eval_numeric_cols = prepare_evaluation_frame(
        df_clean, df_gen, method, is_waymo, label_col, numeric_cols
    )
    eval_numeric_cols = [c for c in eval_numeric_cols if c in df_eval.columns and c in df_gen.columns]
    if not eval_numeric_cols:
        return JSONResponse(status_code=400, content={"detail":"Değerlendirme için ortak numerik kolon bulunamadı."})
    scores = evaluate(df_eval, df_gen, eval_label_col, eval_numeric_cols)
    quality_report = build_quality_report(
        df_eval, df_gen, method, is_waymo, scores, eval_label_col, eval_numeric_cols
    )
    class_distribution = analyze_classes(df_clean, label_col)
    analysis_note = quality_report.get("context", {}).get("utility_reason")
    result = {"status":"success","mode":"full_automation","method":method,"distillation":report,
        "analysis_note":analysis_note,
        "dataset_info":{
            "rows":report.get("clean_rows", 0),
            "sampled_rows":len(df),
            "source_seed_rows":total_rows,
            "features":len(numeric_cols),
            "classes":len(class_distribution),
            "label_col":label_col,
            "is_waymo":is_waymo,
            "class_distribution":class_distribution,
        },
        "seed_count":report["clean_rows"],"gen_count":len(df_gen),
        "multiplication_factor":round((report["clean_rows"]+len(df_gen))/report["clean_rows"],2),
        "generative_coverage":round(len(df_gen)/(report["clean_rows"]+1)*100,1),
        "generation_report": generation_report,
        "quality_report": quality_report, **scores}
    safe_write_json(result, LATEST_ANALYSIS_PATH)
    return result


@app.get("/api/latest_analysis")
async def latest_analysis():
    if not os.path.exists(LATEST_ANALYSIS_PATH):
        return JSONResponse(status_code=404, content={"detail": "Henüz tamamlanmış analiz yok."})
    try:
        with open(LATEST_ANALYSIS_PATH, "r", encoding="utf-8") as analysis_file:
            return json.load(analysis_file)
    except (OSError, json.JSONDecodeError) as exc:
        return JSONResponse(status_code=500, content={"detail": f"Analiz kaydı okunamadı: {exc}"})

@app.post("/api/simulation_sample")
def simulation_sample(data: dict):
    anomaly_type = data.get("type", "spike")
    target_steps = 80  # 8 saniyelik simülasyon

    def _interp_to_target(arr, target_n):
        """Kısa veriyi (ör. 20 adım) hedef uzunluğa (80 adım) interpolasyonla uzat."""
        n = len(arr)
        if n >= target_n:
            return arr[:target_n]
        old_t = np.linspace(0, 1, n)
        new_t = np.linspace(0, 1, target_n)
        return np.interp(new_t, old_t, arr).tolist()

    # --- Gerçek RCGAN verisinden çekmeyi dene ---
    path = os.path.join(OUTPUT_DIR, "live_synthetic_output.csv")
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            if 'label' in df.columns and 'x(1)' in df.columns:
                sub = df[df['label'] == anomaly_type]
                if len(sub) > 0:
                    r = sub.sample(1).iloc[0]
                    cols_available = max(i for i in range(1, 81) if f'x({i})' in df.columns)
                    raw_x = [float(r[f'x({i+1})']) for i in range(cols_available)]
                    raw_y = [float(r[f'y({i+1})']) for i in range(cols_available)]
                    raw_speed = [float(r[f'speed({i+1})']) for i in range(cols_available)]
                    raw_vx = [float(r[f'vx({i+1})']) for i in range(cols_available)]
                    raw_vy = [float(r[f'vy({i+1})']) for i in range(cols_available)]

                    # Eğer veri kısaysa (ör. 20 adım), interpolasyonla 80 adıma uzat
                    return {
                        "source": "rcgan", "type": anomaly_type,
                        "x": _interp_to_target(raw_x, target_steps),
                        "y": _interp_to_target(raw_y, target_steps),
                        "speed": _interp_to_target(raw_speed, target_steps),
                        "vx": _interp_to_target(raw_vx, target_steps),
                        "vy": _interp_to_target(raw_vy, target_steps),
                    }
        except Exception:
            pass

    # --- Deterministik demo veri üretimi (veri yoksa) ---
    import hashlib
    seed_val = int(hashlib.md5(anomaly_type.encode()).hexdigest()[:8], 16) % (2**31)
    rng = np.random.RandomState(seed_val)

    dt = 0.1
    base_speed = 12.0
    base_heading = 0.05

    x, y, speed, vx, vy = [0.0], [0.0], [base_speed], [base_speed], [base_heading]

    for i in range(1, target_steps):
        t = i * dt
        s = base_speed + rng.normal(0, 0.15)
        dx = s * dt
        dy = base_heading * dt + rng.normal(0, 0.02)

        if anomaly_type == "spike":
            if i in [18, 19, 20, 42, 43, 44, 65, 66]:
                dy += rng.choice([-1, 1]) * (4.0 + rng.uniform(0, 2.5))
                s += rng.choice([-1, 1]) * (10.0 + rng.uniform(0, 5))
        elif anomaly_type == "drift":
            drift_factor = (t / (target_steps * dt)) ** 1.6
            dy += drift_factor * 3.5
            s += drift_factor * 4.0
        elif anomaly_type == "freeze":
            if 22 <= i <= 45:
                dx = 0.0
                dy = 0.0
                s = speed[-1]
        elif anomaly_type == "dropout":
            if rng.random() < 0.22:
                dx = rng.normal(0, 3.0)
                dy = rng.normal(0, 2.5)
                s = max(0.1, s + rng.normal(0, 7))
        elif anomaly_type == "noise":
            dy += rng.normal(0, 0.7)
            s += rng.normal(0, 1.8)

        s = max(0.0, s)
        x.append(x[-1] + dx)
        y.append(y[-1] + dy)
        speed.append(s)
        vx.append(dx / dt)
        vy.append(dy / dt)

    return {
        "source": "demo", "type": anomaly_type,
        "x": x, "y": y, "speed": speed, "vx": vx, "vy": vy
    }

@app.get("/api/download_generated")
async def download_generated():
    p=os.path.join(OUTPUT_DIR,"live_synthetic_output.csv")
    if os.path.exists(p): return FileResponse(path=p, filename="sentetik_veri.csv", media_type="application/octet-stream")
    return JSONResponse(status_code=404,content={"detail":"Henüz veri üretilmedi."})

@app.get("/api/download_distilled")
async def download_distilled():
    p=os.path.join(OUTPUT_DIR,"distilled_data.csv")
    if os.path.exists(p): return FileResponse(path=p, filename="distilled_clean_data.csv", media_type="application/octet-stream")
    return JSONResponse(status_code=404,content={"detail":"Henüz damıtma yapılmadı."})

app.mount("/", StaticFiles(directory=BASE_DIR, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*55)
    print("🛡️  Sentetik Veri Üretim Hattı — Bilgi Damıtma + Adaptif Üretim")
    print(f"   RCGAN: {'GODMODE ✅' if rcgan else 'Yok ⚠️'}")
    print(f"   Pipeline: Damıtma → Sentez → Değerlendirme")
    print("="*55 + "\n")
    uvicorn.run(app, host=API_HOST, port=API_PORT, access_log=False)
