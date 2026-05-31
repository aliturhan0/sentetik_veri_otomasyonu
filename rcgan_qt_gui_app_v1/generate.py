from __future__ import annotations

import argparse
import os
import re
import subprocess
import time
from pathlib import Path

from PIL import Image


torch = None
transforms = None
save_image = None
make_grid = None
RecurrentGenerator = None
_GENERATOR_CACHE = {}


def load_torch_stack():
    """Torch/torchvision ağır importlarını arayüz açılışından üretim anına erteler."""
    global torch, transforms, save_image, make_grid, RecurrentGenerator

    if torch is not None:
        return

    import torch as torch_module
    from torchvision import transforms as transforms_module
    from torchvision.utils import make_grid as make_grid_fn
    from torchvision.utils import save_image as save_image_fn
    from model import RecurrentGenerator as RecurrentGeneratorClass

    torch = torch_module
    transforms = transforms_module
    save_image = save_image_fn
    make_grid = make_grid_fn
    RecurrentGenerator = RecurrentGeneratorClass


FAULT_MAP = {
    "blur": 0,
    "occlusion": 1,
    "brightness": 2,
}

SEVERITY_MAP = {
    "low": 0,
    "medium": 1,
    "high": 2,
}


CONDITIONS_ALL = [
    ("blur", "low"),
    ("blur", "medium"),
    ("blur", "high"),
    ("brightness", "low"),
    ("brightness", "medium"),
    ("brightness", "high"),
    ("occlusion", "low"),
    ("occlusion", "medium"),
    ("occlusion", "high"),
]


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_MIRROR_ROOT = PROJECT_ROOT.parent / "githubdeneme"


def is_dataless_file(path):
    if os.name != "posix":
        return False
    try:
        flags = subprocess.run(
            ["/bin/ls", "-lO", str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
            check=False,
        ).stdout
        return "dataless" in flags
    except subprocess.TimeoutExpired:
        return False


def local_mirror_file(path):
    try:
        relative = path.resolve().relative_to(PROJECT_ROOT)
    except ValueError:
        return None

    candidate = LOCAL_MIRROR_ROOT / relative
    if candidate.exists() and not is_dataless_file(candidate):
        return candidate
    return None


def ensure_local_file(path, timeout=30):
    """Yerel dosyayı döndür; placeholder varsa hazır lokal yedeği kullan."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dosya bulunamadı: {path}")

    if is_dataless_file(path):
        mirror = local_mirror_file(path)
        if mirror is not None:
            return mirror
        raise TimeoutError(
            f"Dosya yalnızca iCloud placeholder olarak mevcut: {path}. "
            "Dosyayı yerel olarak indirip tekrar dene."
        )

    return path


def pick_device() -> torch.device:
    load_torch_stack()
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def natural_sort_key(path):
    name = Path(path).name
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", name)
    ]


def collect_images_from_folder(folder_path):
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Klasör bulunamadı: {folder_path}")

    images = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    images = sorted(images, key=natural_sort_key)

    if len(images) < 2:
        raise ValueError("Zaman serisi üretimi için en az 2 görüntü gerekir.")

    return images


def make_condition(fault: str, severity: str, device: torch.device) -> torch.Tensor:
    if fault not in FAULT_MAP:
        raise ValueError(f"fault hatalı: {fault}. Seçenekler: {list(FAULT_MAP)}")

    if severity not in SEVERITY_MAP:
        raise ValueError(f"severity hatalı: {severity}. Seçenekler: {list(SEVERITY_MAP)}")

    cond = torch.zeros(1, 6, dtype=torch.float32, device=device)
    cond[0, FAULT_MAP[fault]] = 1.0
    cond[0, 3 + SEVERITY_MAP[severity]] = 1.0

    return cond


def make_condition_map(cond_vec: torch.Tensor, h: int, w: int) -> torch.Tensor:
    return cond_vec.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, h, w)


def denorm(x: torch.Tensor) -> torch.Tensor:
    return ((x + 1.0) / 2.0).clamp(0, 1)


def load_image(path: str, transform, device: torch.device) -> torch.Tensor:
    local_path = ensure_local_file(path, timeout=30)
    img = Image.open(local_path).convert("RGB")
    return transform(img).unsqueeze(0).to(device)


def safe_torch_load(checkpoint_path: str, device: torch.device):
    load_torch_stack()
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)
    except Exception as safe_error:
        print(f"[UYARI] weights_only=True yükleme başarısız, legacy checkpoint deneniyor: {safe_error}")
        return torch.load(checkpoint_path, map_location=device, weights_only=False)


def load_generator(checkpoint_path: str, device: torch.device) -> RecurrentGenerator:
    load_torch_stack()
    checkpoint_path = ensure_local_file(checkpoint_path, timeout=120)
    cache_key = (str(Path(checkpoint_path).resolve()), str(device))
    cached = _GENERATOR_CACHE.get(cache_key)
    if cached is not None:
        return cached

    model = RecurrentGenerator(
        cond_channels=6,
        hidden_dim=512,
        out_channels=3,
    ).to(device)

    checkpoint = safe_torch_load(checkpoint_path, device)

    if isinstance(checkpoint, dict) and "generator_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["generator_state_dict"])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        raise ValueError("Checkpoint formatı beklenen yapıda değil.")

    model.eval()
    _GENERATOR_CACHE[cache_key] = model
    return model


def parse_conditions(args):
    if args.all_conditions:
        return CONDITIONS_ALL

    return [(args.fault, args.severity)]


def generate_single_pair(
    checkpoint_path,
    prev_img,
    curr_img,
    output_dir,
    fault="occlusion",
    severity="high",
    all_conditions=False,
    conditions=None,
    image_size=256,
    log_callback=None,
):
    def log(message):
        if log_callback:
            log_callback(str(message))
        else:
            print(message)

    device = pick_device()
    load_torch_stack()
    log(f"Cihaz: {device}")
    t0 = time.perf_counter()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    generator = load_generator(checkpoint_path, device)
    log(f"Model hazır: {time.perf_counter() - t0:.1f} sn")

    prev_tensor = load_image(prev_img, transform, device)
    curr_tensor = load_image(curr_img, transform, device)

    _, _, h, w = curr_tensor.shape

    if conditions is None:
        conditions = CONDITIONS_ALL if all_conditions else [(fault, severity)]

    preview_results = [denorm(curr_tensor[0].detach().cpu())]
    output_paths = []

    with torch.no_grad():
        for fault_name, severity_name in conditions:
            step_t = time.perf_counter()
            cond = make_condition(fault_name, severity_name, device)
            cond_map = make_condition_map(cond, h, w)

            fake = generator(prev_tensor, curr_tensor, cond_map)
            fake_cpu = denorm(fake[0].detach().cpu())

            output_path = output_dir / f"generated_{fault_name}_{severity_name}.png"
            save_image(fake_cpu, output_path)

            preview_results.append(fake_cpu)
            output_paths.append(str(output_path))

            log(f"Kaydedildi: {output_path}")
            log(f"Üretim süresi: {time.perf_counter() - step_t:.1f} sn")

    if all_conditions:
        grid = make_grid(preview_results, nrow=5, padding=4)
        grid_path = output_dir / "generated_grid_all_conditions.png"
        save_image(grid, grid_path)

        output_paths.insert(0, str(grid_path))
        log(f"Grid kaydedildi: {grid_path}")

    return output_paths


def generate_sequence(
    checkpoint_path,
    image_paths,
    output_dir,
    fault="occlusion",
    severity="high",
    all_conditions=False,
    conditions=None,
    image_size=256,
    log_callback=None,
):
    def log(message):
        if log_callback:
            log_callback(str(message))
        else:
            print(message)

    image_paths = [Path(p) for p in image_paths]
    image_paths = sorted(image_paths, key=natural_sort_key)

    if len(image_paths) < 2:
        raise ValueError("Zaman serisi üretimi için en az 2 görüntü seçmelisin.")

    device = pick_device()
    load_torch_stack()
    log(f"Cihaz: {device}")
    t0 = time.perf_counter()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    log("Model yükleniyor...")
    generator = load_generator(checkpoint_path, device)
    log(f"Model hazır: {time.perf_counter() - t0:.1f} sn")

    if conditions is None:
        conditions = CONDITIONS_ALL if all_conditions else [(fault, severity)]

    output_paths = []

    log(f"Toplam clean görüntü sayısı: {len(image_paths)}")
    log("İlk görüntü atlanacak çünkü önceki frame yok.")
    log(f"Üretilecek çıktı sayısı: {(len(image_paths) - 1) * len(conditions)}")

    condition_cache = {}
    with torch.no_grad():
        for i in range(1, len(image_paths)):
            prev_path = image_paths[i - 1]
            curr_path = image_paths[i]

            log(f"\nFrame çifti işleniyor: {prev_path.name} -> {curr_path.name}")

            if i == 1:
                prev_tensor = load_image(str(prev_path), transform, device)
            curr_tensor = load_image(str(curr_path), transform, device)

            _, _, h, w = curr_tensor.shape

            for fault_name, severity_name in conditions:
                step_t = time.perf_counter()
                cache_key = (fault_name, severity_name, h, w)
                cond_map = condition_cache.get(cache_key)
                if cond_map is None:
                    cond = make_condition(fault_name, severity_name, device)
                    cond_map = make_condition_map(cond, h, w)
                    condition_cache[cache_key] = cond_map

                fake = generator(prev_tensor, curr_tensor, cond_map)
                fake_cpu = denorm(fake[0].detach().cpu())

                curr_stem = curr_path.stem
                output_name = f"{i:04d}_{curr_stem}_{fault_name}_{severity_name}.png"
                output_path = output_dir / output_name

                save_image(fake_cpu, output_path)
                output_paths.append(str(output_path))

                log(f"Kaydedildi: {output_path}")
                log(f"Üretim süresi: {time.perf_counter() - step_t:.1f} sn")

            prev_tensor = curr_tensor

    log("\nZaman serisi üretimi tamamlandı.")
    return output_paths


def main():
    parser = argparse.ArgumentParser(description="RCGAN ConvLSTM görüntü üretim scripti")

    parser.add_argument("--checkpoint", required=True, help="checkpoint_epoch_29.pt dosya yolu")
    parser.add_argument("--out", default="outputs", help="Çıktı klasörü")
    parser.add_argument("--fault", default="occlusion", choices=list(FAULT_MAP.keys()))
    parser.add_argument("--severity", default="high", choices=list(SEVERITY_MAP.keys()))
    parser.add_argument("--all_conditions", action="store_true", help="9 koşulun tamamını üretir")
    parser.add_argument("--image_size", type=int, default=256)

    parser.add_argument("--prev", help="Tekli üretim için önceki clean frame")
    parser.add_argument("--curr", help="Tekli üretim için mevcut clean frame")

    parser.add_argument("--input_folder", help="Zaman serisi için clean görüntü klasörü")

    args = parser.parse_args()

    if args.input_folder:
        images = collect_images_from_folder(args.input_folder)

        generate_sequence(
            checkpoint_path=args.checkpoint,
            image_paths=images,
            output_dir=args.out,
            fault=args.fault,
            severity=args.severity,
            all_conditions=args.all_conditions,
            image_size=args.image_size,
        )

    elif args.prev and args.curr:
        generate_single_pair(
            checkpoint_path=args.checkpoint,
            prev_img=args.prev,
            curr_img=args.curr,
            output_dir=args.out,
            fault=args.fault,
            severity=args.severity,
            all_conditions=args.all_conditions,
            image_size=args.image_size,
        )

    else:
        raise ValueError(
            "Ya --input_folder vermelisin ya da --prev ve --curr birlikte verilmelidir."
        )


if __name__ == "__main__":
    main()
