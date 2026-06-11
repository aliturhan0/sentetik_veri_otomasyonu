import os
import shutil


DATASET_ROOT = "robustness_dataset"
UPSCALED_ROOT = os.path.join(DATASET_ROOT, "generated_upscaled")
MODEL_PATH = "EDSR_x4.pb"
TARGET_WIDTH = 1600
TARGET_HEIGHT = 900
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def default_generated_dirs(dataset_root=DATASET_ROOT):
    return {
        "blur_high": os.path.join(dataset_root, "generated", "blur_high"),
        "brightness_high": os.path.join(dataset_root, "generated", "brightness_high"),
        "occlusion_high": os.path.join(dataset_root, "generated", "occlusion_high"),
    }


def log_message(message, log_callback=None):
    if log_callback:
        log_callback(str(message))
    else:
        print(message)


def progress_iter(items, log_callback=None):
    if not hasattr(items, "__len__") or len(items) == 0:
        return items

    if log_callback:
        return items

    try:
        from tqdm import tqdm

        return tqdm(items)
    except ImportError:
        return items


def list_images(folder):
    return sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ])


def reset_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)

    os.makedirs(path, exist_ok=True)


def opencv_safe_model_path(model_path):
    import sys
    model_path = os.fspath(model_path)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"EDSR model dosyasi bulunamadi: {model_path}")

    # macOS ve Linux'ta OpenCV dosya yollarini sorunsuz okur, cache'e gerek yoktur.
    if sys.platform != "win32":
        return model_path

    if not os.path.isabs(model_path):
        return model_path

    try:
        relative_path = os.path.relpath(model_path, os.getcwd())
    except ValueError:
        relative_path = None

    if relative_path and not relative_path.startswith("..") and not os.path.isabs(relative_path):
        return relative_path

    # Windows icin cache dizini olustur
    drive = os.path.splitdrive(model_path)[0] or "C:"
    cache_root = os.path.join(drive + os.sep, "sentetik_model_cache")
    try:
        os.makedirs(cache_root, exist_ok=True)
    except OSError:
        # Eger drive root'a yazma yetkisi yoksa temp dizinini kullan
        import tempfile
        cache_root = os.path.join(tempfile.gettempdir(), "sentetik_model_cache")
        os.makedirs(cache_root, exist_ok=True)

    cached_model_path = os.path.join(cache_root, os.path.basename(model_path))

    if (
        not os.path.exists(cached_model_path)
        or os.path.getsize(cached_model_path) != os.path.getsize(model_path)
        or os.path.getmtime(cached_model_path) < os.path.getmtime(model_path)
    ):
        shutil.copy2(model_path, cached_model_path)

    return cached_model_path


def load_super_resolution_model(model_path=MODEL_PATH, log_callback=None):
    import cv2

    log_message("\nLoading Super Resolution model...\n", log_callback)

    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    safe_model_path = opencv_safe_model_path(model_path)

    try:
        sr.readModel(safe_model_path)
    except cv2.error as exc:
        raise RuntimeError(
            "EDSR modeli OpenCV tarafindan okunamadi. "
            f"Beklenen dosya: {model_path}. "
            "Dosya mevcutsa Windows yolundaki Turkce karakterler veya bozuk model dosyasi buna sebep olabilir."
        ) from exc

    sr.setModel("edsr", 4)

    log_message("Super Resolution model loaded.\n", log_callback)

    return sr


def upscale_image_file(
    sr,
    input_path,
    output_path,
    target_size=(TARGET_WIDTH, TARGET_HEIGHT),
    skip_existing=True,
):
    import cv2
    import numpy as np

    if skip_existing and os.path.exists(output_path):
        try:
            if os.path.getmtime(output_path) >= os.path.getmtime(input_path):
                return True, "cached"
        except OSError:
            pass

    image_data = np.fromfile(input_path, dtype=np.uint8)
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

    if image is None:
        return False, "unreadable"

    upscaled = sr.upsample(image)

    final = cv2.resize(
        upscaled,
        target_size,
        interpolation=cv2.INTER_CUBIC,
    )

    ext = os.path.splitext(output_path)[1] or ".png"
    ok, encoded = cv2.imencode(ext, final)

    if not ok:
        return False, "encode_failed"

    output_dir = os.path.dirname(os.fspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    encoded.tofile(output_path)

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        return False, "write_failed"

    return True, "generated"


def run_upscale_generated_images(
    dataset_root=DATASET_ROOT,
    generated_dirs=None,
    upscaled_root=None,
    model_path=MODEL_PATH,
    target_width=TARGET_WIDTH,
    target_height=TARGET_HEIGHT,
    log_callback=None,
):
    if generated_dirs is None:
        generated_dirs = default_generated_dirs(dataset_root)

    if upscaled_root is None:
        upscaled_root = os.path.join(dataset_root, "generated_upscaled")

    sr = load_super_resolution_model(
        model_path=model_path,
        log_callback=log_callback,
    )

    target_size = (target_width, target_height)
    outputs = {}
    skipped = []
    cached = 0

    for condition, input_dir in generated_dirs.items():
        if not os.path.isdir(input_dir):
            log_message(f"Skipping missing condition dir: {input_dir}", log_callback)
            continue

        log_message(f"\nUpscaling: {condition}\n", log_callback)

        output_dir = os.path.join(upscaled_root, condition)
        os.makedirs(output_dir, exist_ok=True)

        outputs[condition] = []

        image_files = list_images(input_dir)
        log_message(f"{condition} dosya sayisi: {len(image_files)}", log_callback)

        for index, fname in enumerate(
            progress_iter(image_files, log_callback=log_callback),
            start=1,
        ):
            input_path = os.path.join(input_dir, fname)
            output_path = os.path.join(output_dir, fname)

            ok, status = upscale_image_file(
                sr=sr,
                input_path=input_path,
                output_path=output_path,
                target_size=target_size,
            )

            if ok:
                outputs[condition].append(output_path)
                if status == "cached":
                    cached += 1
            else:
                skipped.append(input_path)

            if log_callback and (index == 1 or index == len(image_files) or index % 10 == 0):
                log_message(
                    f"{condition}: {index}/{len(image_files)} upscale tamamlandi",
                    log_callback,
                )

    log_message("\nUPSCALE FINISHED.\n", log_callback)

    return {
        "upscaled_root": upscaled_root,
        "outputs": outputs,
        "skipped": skipped,
        "cached": cached,
    }


def main():
    run_upscale_generated_images()


if __name__ == "__main__":
    main()
