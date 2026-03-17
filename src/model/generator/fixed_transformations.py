from diffusers.utils import load_image
from PIL import Image
from pathlib import Path
import random
import io

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
    Ouputs 6 different images with each transformation applied.
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
            T.RandomResizedCrop(
                size=native_size,
                scale=(0.75, 1.0),
                ratio=(0.9, 1.1)
            ),
            T.Resize(native_size),
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

def main():
    input_path = Path("/Users/andy/Desktop/T4/CV/Group/test_image.jpg")
    output_path = Path("/Users/andy/Desktop/T4/CV/Group/")

    """
    Takes a single image path and output path for a directory
    """
    fixedTransformationsSingle(input_path, output_path)

    """
    Takes an image directory and an output directory - processes multiple images
    """
    process_multiple(Path("/Users/andy/Desktop/T4/CV/Group/"), output_path)

if __name__ == "__main__":
    main()