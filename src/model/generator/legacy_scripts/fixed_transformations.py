import io
import random
from pathlib import Path
from PIL import Image
import torchvision.transforms as T

class RandomJPEG:
    """
    Re-encodes an image as JPEG with a random quality.
    This simulates social media / editor compression.
    """
    def __init__(self, qmin=75, qmax=95, p=1.0):
        self.qmin = qmin
        self.qmax = qmax
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        quality = random.randint(self.qmin, self.qmax)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        out = Image.open(buffer)
        out.load()  # Force decode before buffer is eligible for GC
        return out.convert("RGB")


def fixedTransformations(input_path, output_dir):
    """
    Outputs 6 different images with each transformation applied.
    """
    output_dir.mkdir(exist_ok=True)
    img = Image.open(input_path).convert("RGB")
    native_size = img.size[::-1]
    transforms_dict = {
        "clean_resize": T.Compose([
            T.Resize(native_size),
        ]),
        "blur": T.Compose([
            T.Resize(native_size),
            T.GaussianBlur(kernel_size=5, sigma=(1.0, 2.0)),
        ]),
        "color_jitter": T.Compose([
            T.Resize(native_size),
            T.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.10,
                hue=0.02
            ),
        ]),
        "crop": T.Compose([
            T.Resize(native_size),
            T.RandomResizedCrop(
                size=native_size,
                scale=(0.75, 1.0),
                ratio=(0.9, 1.1)
            ),
        ]),
        "jpeg": T.Compose([
            T.Resize(native_size),
            RandomJPEG(qmin=70, qmax=90, p=1.0),
        ]),
        "blur_then_jpeg": T.Compose([
            T.Resize(native_size),
            T.GaussianBlur(kernel_size=5, sigma=(1.0, 2.0)),
            RandomJPEG(qmin=70, qmax=90, p=1.0),
        ]),
    }
    for name, transform in transforms_dict.items():
        transformed = transform(img)
        ext = "png"
        save_path = output_dir / f"{name}.{ext}"
        transformed.save(save_path)
        print(f"Saved: {save_path}")


def fixedTransformationsSingle(input_path, output_dir):
    output_dir.mkdir(exist_ok=True)
    img = Image.open(input_path).convert("RGB")
    native_size = img.size[::-1]
    transform = T.Compose([
        T.Resize(native_size),
        T.GaussianBlur(kernel_size=5, sigma=(1.0, 2.0)),
        T.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.10,
            hue=0.02
        ),
        T.RandomResizedCrop(
            size=native_size,
            scale=(0.75, 1.0),
            ratio=(0.9, 1.1)
        ),
        RandomJPEG(qmin=70, qmax=90, p=1.0),
    ])
    transformed = transform(img)
    save_path = output_dir / f"{Path(input_path).stem}_transformed.png"
    transformed.save(save_path)
    print(f"Saved: {save_path}")


def process_multiple(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    valid_ext = {".png", ".jpg", ".jpeg", ".webp"}
    for input_path in input_dir.iterdir():
        if input_path.suffix.lower() not in valid_ext:
            continue
        fixedTransformationsSingle(input_path, output_dir)


# ---------------------------------------------------------------------------
# Sweep config — edit intensities here
# ---------------------------------------------------------------------------

SWEEP_CONDITIONS = {
    "clean": {
        "transform": lambda native_size: T.Compose([T.Resize(native_size)]),
        "save_as_jpeg": False,
    },

    # Blur sweep — increasing sigma
    "blur_low": {
        "transform": lambda native_size: T.Compose([
            T.Resize(native_size),
            T.GaussianBlur(kernel_size=5, sigma=0.5),
        ]),
        "save_as_jpeg": False,
    },
    "blur_mid": {
        "transform": lambda native_size: T.Compose([
            T.Resize(native_size),
            T.GaussianBlur(kernel_size=5, sigma=1.5),
        ]),
        "save_as_jpeg": False,
    },
    "blur_high": {
        "transform": lambda native_size: T.Compose([
            T.Resize(native_size),
            T.GaussianBlur(kernel_size=7, sigma=3.0),
        ]),
        "save_as_jpeg": False,
    },

    # JPEG sweep — decreasing quality (more compression)
    "jpeg_q90": {
        "transform": lambda native_size: T.Compose([T.Resize(native_size)]),
        "save_as_jpeg": True,
        "jpeg_quality": 90,
    },
    "jpeg_q75": {
        "transform": lambda native_size: T.Compose([T.Resize(native_size)]),
        "save_as_jpeg": True,
        "jpeg_quality": 75,
    },
    "jpeg_q60": {
        "transform": lambda native_size: T.Compose([T.Resize(native_size)]),
        "save_as_jpeg": True,
        "jpeg_quality": 60,
    },
    "jpeg_q45": {
        "transform": lambda native_size: T.Compose([T.Resize(native_size)]),
        "save_as_jpeg": True,
        "jpeg_quality": 45,
    },

    # Crop sweep — more aggressive scale lower bound
    "crop_mild": {
        "transform": lambda native_size: T.Compose([
            T.RandomResizedCrop(size=native_size, scale=(0.9, 1.0), ratio=(0.95, 1.05)),
            T.Resize(native_size),
        ]),
        "save_as_jpeg": False,
    },
    "crop_mid": {
        "transform": lambda native_size: T.Compose([
            T.RandomResizedCrop(size=native_size, scale=(0.75, 0.9), ratio=(0.9, 1.1)),
            T.Resize(native_size),
        ]),
        "save_as_jpeg": False,
    },
    "crop_aggro": {
        "transform": lambda native_size: T.Compose([
            T.RandomResizedCrop(size=native_size, scale=(0.6, 0.75), ratio=(0.85, 1.15)),
            T.Resize(native_size),
        ]),
        "save_as_jpeg": False,
    },

    # Color jitter sweep — subtle vs aggressive
    "jitter_subtle": {
        "transform": lambda native_size: T.Compose([
            T.Resize(native_size),
            T.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.01),
        ]),
        "save_as_jpeg": False,
    },
    "jitter_aggro": {
        "transform": lambda native_size: T.Compose([
            T.Resize(native_size),
            T.ColorJitter(brightness=0.30, contrast=0.30, saturation=0.20, hue=0.05),
        ]),
        "save_as_jpeg": False,
    },
}


def _save_image(img: Image.Image, path: Path, save_as_jpeg: bool, jpeg_quality: int = 75):
    """Save image as JPEG (for jpeg conditions) or PNG (for everything else)."""
    if save_as_jpeg:
        path = path.with_suffix(".jpg")
        img.save(path, format="JPEG", quality=jpeg_quality)
    else:
        img.save(path, format="PNG")
    return path


def process_sweep(input_dir, output_root):
    """
    For each image in input_dir, apply every condition in SWEEP_CONDITIONS
    and save into output_root/<condition_name>/<original_stem>.<ext>

    Output layout:
        output_root/
            clean/
                img001.png
                img002.png
                ...
            blur_low/
                img001.png
                ...
            jpeg_q75/
                img001.jpg
                ...
            ...
    """
    input_dir = Path(input_dir)
    output_root = Path(output_root)
    valid_ext = {".png", ".jpg", ".jpeg", ".webp"}

    image_paths = [p for p in input_dir.iterdir() if p.suffix.lower() in valid_ext]
    print(f"Found {len(image_paths)} images — running {len(SWEEP_CONDITIONS)} conditions each.")

    for condition_name, config in SWEEP_CONDITIONS.items():
        condition_dir = output_root / condition_name
        condition_dir.mkdir(parents=True, exist_ok=True)

        for input_path in image_paths:
            img = Image.open(input_path).convert("RGB")
            native_size = img.size[::-1]  # (W, H) -> (H, W) for torchvision

            transform = config["transform"](native_size)
            transformed = transform(img)

            save_as_jpeg = config.get("save_as_jpeg", False)
            jpeg_quality = config.get("jpeg_quality", 75)

            out_path = condition_dir / input_path.stem  # extension added by _save_image
            saved = _save_image(transformed, out_path, save_as_jpeg, jpeg_quality)
            print(f"[{condition_name}] Saved: {saved.name}")

    print("Sweep complete.")