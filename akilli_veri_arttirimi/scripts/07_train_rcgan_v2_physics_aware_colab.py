"""
RCGAN GODMODE V2 - Physics-Aware Conditional Training for Google Colab A100.

Bu script baseline modeli ASLA ezmez.

Baseline:
    /content/drive/MyDrive/waymo_rcgan_GODMODE_A100_STABLE.pth

Yeni V2 çıktı:
    /content/drive/MyDrive/waymo_rcgan_GODMODE_V2_PHYSICS_AWARE.pth

Amaç:
  - 40GB Waymo seed verisini kaynak olarak kullanmak
  - Normal trajectory manifold'unu korumak
  - Eğitim sırasında spike/drift/dropout/freeze/noise anomaly label'ları üretmek
  - Discriminator'a fake label'ları doğru vermek
  - Generator'a fiziksel tutarlılık kaybı eklemek
  - Eski modele kolay rollback imkanı bırakmak
"""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from google.colab import drive


# =============================================================================
# CONFIG
# =============================================================================
SEQ_LEN = 20
N_CHANNELS = 5
NOISE_DIM = 64
N_CLASSES = 6
FEATURES = ["x", "y", "speed", "vx", "vy"]

EPOCHS = int(os.getenv("RCGAN_V2_EPOCHS", "650"))
BATCH_SIZE = int(os.getenv("RCGAN_V2_BATCH_SIZE", "2048"))
LR_G = float(os.getenv("RCGAN_V2_LR_G", "0.0001"))
LR_D = float(os.getenv("RCGAN_V2_LR_D", "0.00008"))
CHUNK_SIZE = int(os.getenv("RCGAN_V2_CHUNK_SIZE", "250000"))
MAX_ROWS_PER_EPOCH = int(os.getenv("RCGAN_V2_MAX_ROWS_PER_EPOCH", "2500000"))
SAVE_EVERY = int(os.getenv("RCGAN_V2_SAVE_EVERY", "25"))
SEED = int(os.getenv("RCGAN_V2_SEED", "42"))

LAMBDA_PHYSICS = float(os.getenv("RCGAN_V2_LAMBDA_PHYSICS", "0.18"))
LAMBDA_SMOOTH = float(os.getenv("RCGAN_V2_LAMBDA_SMOOTH", "0.05"))
LAMBDA_DIVERSITY = float(os.getenv("RCGAN_V2_LAMBDA_DIVERSITY", "0.03"))

DRIVE_ROOT = Path("/content/drive/MyDrive")
FILE_PATH = DRIVE_ROOT / "waymo_seed_MASSIVE.csv"
BASELINE_PATH = DRIVE_ROOT / "waymo_rcgan_GODMODE_A100_STABLE.pth"
SAVE_PATH = DRIVE_ROOT / "waymo_rcgan_GODMODE_V2_PHYSICS_AWARE.pth"
BEST_PATH = DRIVE_ROOT / "waymo_rcgan_GODMODE_V2_PHYSICS_AWARE_BEST.pth"
MIN_PATH = DRIVE_ROOT / "waymo_normalization_min_v2.npy"
MAX_PATH = DRIVE_ROOT / "waymo_normalization_max_v2.npy"
LOG_PATH = DRIVE_ROOT / "waymo_rcgan_v2_training_log.jsonl"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# =============================================================================
# DRIVE + DEVICE
# =============================================================================
drive.mount("/content/drive")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    raise RuntimeError("Bu eğitim A100/T4 gibi CUDA GPU ile çalıştırılmalı.")

print(f"\n[RCGAN V2] Cihaz: {torch.cuda.get_device_name(0)}")
print(f"[RCGAN V2] Kaynak CSV: {FILE_PATH}")
print(f"[RCGAN V2] Baseline korunacak: {BASELINE_PATH}")
print(f"[RCGAN V2] Yeni model: {SAVE_PATH}\n")

if not FILE_PATH.exists():
    raise FileNotFoundError(f"Waymo seed CSV bulunamadı: {FILE_PATH}")

if BASELINE_PATH.exists():
    backup_path = DRIVE_ROOT / f"{BASELINE_PATH.stem}_BACKUP_DO_NOT_DELETE.pth"
    if not backup_path.exists():
        shutil.copy2(BASELINE_PATH, backup_path)
        print(f"[SAFEGUARD] Baseline backup oluşturuldu: {backup_path}")


# =============================================================================
# DATA HELPERS
# =============================================================================
def safe_torch_load(path: Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def waymo_columns() -> list[str]:
    return [f"{name}({i+1})" for name in FEATURES for i in range(SEQ_LEN)]


def load_waymo_chunk(chunk: pd.DataFrame) -> np.ndarray:
    cols = waymo_columns()
    missing = [c for c in cols if c not in chunk.columns]
    if missing:
        raise ValueError(f"CSV Waymo şemasında değil. Eksik kolon örneği: {missing[:5]}")

    arrays = []
    for name in FEATURES:
        fcols = [f"{name}({i+1})" for i in range(SEQ_LEN)]
        arrays.append(chunk[fcols].to_numpy(dtype=np.float32))
    x = np.stack(arrays, axis=2).astype(np.float32)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def compute_or_load_minmax() -> tuple[np.ndarray, np.ndarray]:
    if MIN_PATH.exists() and MAX_PATH.exists():
        print("[DATA] V2 normalization min/max bulundu, tekrar kullanılacak.")
        return np.load(MIN_PATH), np.load(MAX_PATH)

    print("[DATA] Min/max streaming hesaplanıyor...")
    x_min = np.full((1, 1, N_CHANNELS), np.inf, dtype=np.float32)
    x_max = np.full((1, 1, N_CHANNELS), -np.inf, dtype=np.float32)
    rows = 0

    for chunk in pd.read_csv(FILE_PATH, chunksize=CHUNK_SIZE):
        x = load_waymo_chunk(chunk)
        x_min = np.minimum(x_min, x.min(axis=(0, 1), keepdims=True))
        x_max = np.maximum(x_max, x.max(axis=(0, 1), keepdims=True))
        rows += len(x)
        print(f"[DATA] min/max pass rows={rows:,}", end="\r")

    np.save(MIN_PATH, x_min)
    np.save(MAX_PATH, x_max)
    print(f"\n[DATA] Min/max kaydedildi: {MIN_PATH.name}, {MAX_PATH.name}")
    return x_min, x_max


def normalize(x: np.ndarray, x_min: np.ndarray, x_max: np.ndarray) -> np.ndarray:
    return 2.0 * (x - x_min) / (x_max - x_min + 1e-8) - 1.0


def denormalize_torch(x: torch.Tensor, x_min_t: torch.Tensor, x_max_t: torch.Tensor) -> torch.Tensor:
    return (x + 1.0) * 0.5 * (x_max_t - x_min_t) + x_min_t


def apply_anomaly_raw(x: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Normal Waymo trajectory'lerden label kontrollü anomaly üretir."""
    out = x.copy()
    speed_scale = max(float(np.nanpercentile(x[:, :, 2], 95)), 5.0)
    x_range = max(float(np.nanpercentile(x[:, :, 0], 95) - np.nanpercentile(x[:, :, 0], 5)), 1.0)

    for idx, label in enumerate(labels):
        t = out[idx]
        if label == 0:
            continue
        if label == 1:  # spike
            pos = random.randint(2, 17)
            t[pos, 0] += random.choice([-1.0, 1.0]) * random.uniform(x_range * 0.05, x_range * 0.16)
            t[pos, 2] = min(t[pos, 2] * random.uniform(1.8, 3.8), speed_scale * 2.2)
        elif label == 2:  # drift
            drift = random.uniform(x_range * 0.006, x_range * 0.018)
            direction = random.choice([-1.0, 1.0])
            for step in range(8, SEQ_LEN):
                t[step, 0] += direction * (step - 8) * drift
        elif label == 3:  # dropout
            start = random.randint(7, 12)
            duration = random.randint(3, 7)
            end = min(SEQ_LEN, start + duration)
            t[start:end, 2:] = 0.0
            t[start:end, :2] = t[max(start - 1, 0), :2]
        elif label == 4:  # freeze
            start = random.randint(7, 12)
            duration = random.randint(3, 7)
            end = min(SEQ_LEN, start + duration)
            t[start:end, :] = t[start, :]
        elif label == 5:  # noise
            t[:, 0] += np.random.normal(0.0, x_range * 0.025, SEQ_LEN)
            t[:, 1] += np.random.normal(0.0, x_range * 0.015, SEQ_LEN)
            t[:, 2] = np.clip(t[:, 2] + np.random.normal(0.0, speed_scale * 0.055, SEQ_LEN), 0.0, None)

    out[:, :, 2] = np.clip(out[:, :, 2], 0.0, 60.0)
    out[:, :, 3:] = np.clip(out[:, :, 3:], -25.0, 25.0)
    return out


def iter_epoch_batches(x_min: np.ndarray, x_max: np.ndarray):
    seen = 0
    for chunk in pd.read_csv(FILE_PATH, chunksize=CHUNK_SIZE):
        x_raw = load_waymo_chunk(chunk)
        labels = np.random.randint(0, N_CLASSES, size=len(x_raw), dtype=np.int64)
        x_aug = apply_anomaly_raw(x_raw, labels)
        x_norm = normalize(x_aug, x_min, x_max).astype(np.float32)

        order = np.random.permutation(len(x_norm))
        for start in range(0, len(order) - BATCH_SIZE + 1, BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            yield (
                torch.from_numpy(x_norm[idx]),
                torch.from_numpy(labels[idx]),
            )
            seen += BATCH_SIZE
            if seen >= MAX_ROWS_PER_EPOCH:
                return


# =============================================================================
# MODELS - Generator backend ile state_dict uyumlu tutuldu.
# =============================================================================
class RCGAN_Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_embed = nn.Embedding(N_CLASSES, 16)
        self.lstm = nn.LSTM(80, 512, 3, batch_first=True, bidirectional=True, dropout=0.2)
        self.out = nn.Sequential(nn.Linear(1024, 256), nn.LeakyReLU(0.2), nn.Linear(256, N_CHANNELS))

    def forward(self, z, labels):
        emb = self.label_embed(labels).unsqueeze(1).repeat(1, SEQ_LEN, 1)
        x, _ = self.lstm(torch.cat([z, emb], dim=2))
        return self.out(x)


class RCGAN_Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_embed = nn.Embedding(N_CLASSES, 16)
        self.lstm = nn.LSTM(21, 512, 3, batch_first=True, bidirectional=True, dropout=0.2)
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.15),
            nn.Linear(256, 1),
        )

    def forward(self, seq, labels):
        emb = self.label_embed(labels).unsqueeze(1).repeat(1, SEQ_LEN, 1)
        _, (h_n, _) = self.lstm(torch.cat([seq, emb], dim=2))
        h = torch.cat((h_n[-2], h_n[-1]), dim=1)
        return self.classifier(h)


# =============================================================================
# LOSSES
# =============================================================================
def physics_loss(fake_norm: torch.Tensor, x_min_t: torch.Tensor, x_max_t: torch.Tensor) -> torch.Tensor:
    fake = denormalize_torch(fake_norm, x_min_t, x_max_t)
    x = fake[:, :, 0]
    y = fake[:, :, 1]
    speed = fake[:, :, 2]
    vx = fake[:, :, 3]
    vy = fake[:, :, 4]

    vel_mag = torch.sqrt(vx.pow(2) + vy.pow(2) + 1e-6)
    speed_non_negative = torch.relu(-speed).mean()
    velocity_coherence = torch.abs(vel_mag - torch.clamp(speed, min=0.0)).mean() / 10.0
    accel = (speed[:, 1:] - speed[:, :-1]) / 0.1
    accel_penalty = torch.relu(torch.abs(accel) - 18.0).mean() / 18.0
    step = torch.sqrt((x[:, 1:] - x[:, :-1]).pow(2) + (y[:, 1:] - y[:, :-1]).pow(2) + 1e-6)
    jump_penalty = torch.relu(step - 10.0).mean() / 10.0
    return speed_non_negative + velocity_coherence + accel_penalty + jump_penalty


def smoothness_loss(fake_norm: torch.Tensor) -> torch.Tensor:
    if fake_norm.size(1) < 3:
        return torch.tensor(0.0, device=fake_norm.device)
    second_diff = fake_norm[:, 2:] - 2 * fake_norm[:, 1:-1] + fake_norm[:, :-2]
    return second_diff.abs().mean()


def diversity_loss(fake_norm: torch.Tensor) -> torch.Tensor:
    flat = fake_norm.reshape(fake_norm.size(0), -1)
    return -torch.clamp(flat.std(dim=0).mean(), max=1.0)


# =============================================================================
# TRAINING
# =============================================================================
x_min, x_max = compute_or_load_minmax()
x_min_t = torch.tensor(x_min, dtype=torch.float32, device=device)
x_max_t = torch.tensor(x_max, dtype=torch.float32, device=device)

G = RCGAN_Generator().to(device)
D = RCGAN_Discriminator().to(device)
opt_G = optim.AdamW(G.parameters(), lr=LR_G, betas=(0.5, 0.999), weight_decay=1e-5)
opt_D = optim.AdamW(D.parameters(), lr=LR_D, betas=(0.5, 0.999), weight_decay=1e-5)
criterion = nn.BCEWithLogitsLoss()

start_epoch = 0
best_score = math.inf
if SAVE_PATH.exists():
    print("[RESUME] V2 checkpoint bulundu, devam ediliyor.")
    ckpt = safe_torch_load(SAVE_PATH, map_location=device)
    G.load_state_dict(ckpt["G"])
    D.load_state_dict(ckpt["D"])
    if "opt_G" in ckpt:
        opt_G.load_state_dict(ckpt["opt_G"])
    if "opt_D" in ckpt:
        opt_D.load_state_dict(ckpt["opt_D"])
    start_epoch = int(ckpt.get("epoch", 0))
    best_score = float(ckpt.get("best_score", best_score))
elif BASELINE_PATH.exists():
    print("[WARM START] Baseline Generator yükleniyor; D temiz başlayacak.")
    ckpt = safe_torch_load(BASELINE_PATH, map_location=device)
    G.load_state_dict(ckpt["G"] if isinstance(ckpt, dict) and "G" in ckpt else ckpt)
else:
    print("[COLD START] Baseline yok, sıfırdan başlanıyor.")


def save_checkpoint(path: Path, epoch: int, score: float):
    payload = {
        "G": G.state_dict(),
        "D": D.state_dict(),
        "opt_G": opt_G.state_dict(),
        "opt_D": opt_D.state_dict(),
        "epoch": epoch,
        "best_score": score,
        "metadata": {
            "name": "RCGAN GODMODE V2 PHYSICS AWARE",
            "source_csv": str(FILE_PATH),
            "baseline_path": str(BASELINE_PATH),
            "seq_len": SEQ_LEN,
            "channels": FEATURES,
            "n_classes": N_CLASSES,
            "physics_loss": True,
            "synthetic_anomaly_labels": ["normal", "spike", "drift", "dropout", "freeze", "noise"],
        },
    }
    torch.save(payload, path)


print(f"\n[TRAIN] Başlıyor: {start_epoch}/{EPOCHS} epoch, batch={BATCH_SIZE}, max_rows/epoch={MAX_ROWS_PER_EPOCH:,}\n")
train_start = time.time()

for epoch in range(start_epoch, EPOCHS):
    G.train()
    D.train()
    epoch_d = []
    epoch_g = []
    epoch_phys = []
    epoch_start = time.time()

    for real_seqs, real_labels in iter_epoch_batches(x_min, x_max):
        real_seqs = real_seqs.to(device, non_blocking=True)
        real_labels = real_labels.to(device, non_blocking=True)
        bs = real_seqs.size(0)

        # Discriminator: fake label bug fix - fake kendi label'i ile yargılanır.
        opt_D.zero_grad(set_to_none=True)
        real_out = D(real_seqs, real_labels)
        real_loss = criterion(real_out, torch.full_like(real_out, 0.9))

        fake_labels = torch.randint(0, N_CLASSES, (bs,), device=device)
        z = torch.randn(bs, SEQ_LEN, NOISE_DIM, device=device)
        fake_seqs = G(z, fake_labels)
        fake_out = D(fake_seqs.detach(), fake_labels)
        fake_loss = criterion(fake_out, torch.zeros_like(fake_out))

        d_loss = real_loss + fake_loss
        d_loss.backward()
        nn.utils.clip_grad_norm_(D.parameters(), 1.0)
        opt_D.step()

        # Generator
        opt_G.zero_grad(set_to_none=True)
        fake_out_for_g = D(fake_seqs, fake_labels)
        adv_loss = criterion(fake_out_for_g, torch.ones_like(fake_out_for_g))
        phys = physics_loss(fake_seqs, x_min_t, x_max_t)
        smooth = smoothness_loss(fake_seqs)
        div = diversity_loss(fake_seqs)
        g_loss = adv_loss + (LAMBDA_PHYSICS * phys) + (LAMBDA_SMOOTH * smooth) + (LAMBDA_DIVERSITY * div)
        g_loss.backward()
        nn.utils.clip_grad_norm_(G.parameters(), 1.0)
        opt_G.step()

        epoch_d.append(float(d_loss.detach().cpu()))
        epoch_g.append(float(g_loss.detach().cpu()))
        epoch_phys.append(float(phys.detach().cpu()))

    mean_d = float(np.mean(epoch_d)) if epoch_d else 0.0
    mean_g = float(np.mean(epoch_g)) if epoch_g else 0.0
    mean_phys = float(np.mean(epoch_phys)) if epoch_phys else 0.0
    proxy_score = mean_phys + abs(mean_d - 1.386) * 0.05 + max(mean_g, 0.0) * 0.01

    log_row = {
        "epoch": epoch + 1,
        "d_loss": round(mean_d, 6),
        "g_loss": round(mean_g, 6),
        "physics_loss": round(mean_phys, 6),
        "proxy_score": round(proxy_score, 6),
        "seconds": round(time.time() - epoch_start, 2),
    }
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(log_row) + "\n")

    if (epoch + 1) % 5 == 0:
        print(
            f"Epoch [{epoch+1:4d}/{EPOCHS}] "
            f"D={mean_d:.4f} G={mean_g:.4f} Phys={mean_phys:.4f} "
            f"proxy={proxy_score:.4f} time={time.time()-epoch_start:.1f}s"
        )

    if proxy_score < best_score:
        best_score = proxy_score
        save_checkpoint(BEST_PATH, epoch + 1, best_score)
        print(f"[BEST] Yeni best checkpoint: epoch={epoch+1}, score={best_score:.5f}")

    if (epoch + 1) % SAVE_EVERY == 0:
        save_checkpoint(SAVE_PATH, epoch + 1, best_score)
        print(f"[SAVE] V2 checkpoint kaydedildi: {SAVE_PATH}")

save_checkpoint(SAVE_PATH, EPOCHS, best_score)
print(f"\n[DONE] V2 eğitim tamamlandı. Final: {SAVE_PATH}")
print(f"[DONE] Best: {BEST_PATH}")
print(f"[DONE] Süre: {(time.time() - train_start) / 3600:.2f} saat")
