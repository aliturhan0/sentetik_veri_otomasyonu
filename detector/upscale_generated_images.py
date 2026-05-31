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


def load_super_resolution_model(model_path=MODEL_PATH, log_callback=None):
    import cv2

    log_message("\nLoading Super Resolution model...\n", log_callback)

    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_path)
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

    if skip_existing and os.path.exists(output_path):
        try:
            if os.path.getmtime(output_path) >= os.path.getmtime(input_path):
                return True, "cached"
        except OSError:
            pass

    image = cv2.imread(input_path)

    if image is None:
        return False, "unreadable"

    upscaled = sr.upsample(image)

    final = cv2.resize(
        upscaled,
        target_size,
        interpolation=cv2.INTER_CUBIC,
    )

    cv2.imwrite(output_path, final)
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
