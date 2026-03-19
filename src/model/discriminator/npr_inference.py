"""
NPR Deepfake Detector (CVPR 2024)
----------------------------------
Neighboring Pixel Relationships — detects upsampling artifacts
present in CNN-based generators (GANs and diffusion models).

Truncated ResNet50 (2 stages only, ~6M params). Extremely lightweight.
Weights must be downloaded manually from Google Drive:
    https://drive.google.com/drive/folders/1_mD17F94xMbJqEAsWRW1gVsZ5db6YamI

90.1% mean accuracy on GenImage (trained on SDv1.4 only).

Usage:
    python npr_inference.py <image_path>
"""

import sys
from math import ceil
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

WEIGHTS_PATH = Path(__file__).parent / "weights" / "NPR.pth"


# --- Custom architecture (truncated ResNet50 with NPR forward) ---

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class NPRResNet(nn.Module):
    """Truncated ResNet50 with NPR (Neighboring Pixel Relationships) forward pass.

    Only uses layer1 + layer2 (not layer3/layer4), giving 512-dim features.
    The NPR signal (high-freq upsampling artifacts) is computed inside forward().
    """

    def __init__(self, num_classes=1):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, num_classes)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * Bottleneck.expansion, stride),
                nn.BatchNorm2d(planes * Bottleneck.expansion),
            )
        layers = [Bottleneck(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes))
        return nn.Sequential(*layers)

    @staticmethod
    def interpolate(img, factor):
        return F.interpolate(
            F.interpolate(img, scale_factor=factor, mode='nearest',
                          recompute_scale_factor=True),
            scale_factor=1.0 / factor, mode='nearest',
            recompute_scale_factor=True,
        )

    def forward(self, x):
        # NPR: subtract downsampled-then-upsampled version to extract
        # high-frequency artifacts from generator upsampling operations
        npr = x - self.interpolate(x, 0.5)

        x = self.conv1(npr * 2.0 / 3.0)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc1(x)


# --- Public interface ---

def load_model(weights_path=None):
    """
    Loads the NPR detector.

    Args:
        weights_path: Path to NPR.pth. Defaults to weights/NPR.pth next to this file.

    Returns:
        (model, transform), device
    """
    if weights_path is None:
        weights_path = WEIGHTS_PATH

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = NPRResNet(num_classes=1)
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()

    def translate_duplicate(img, crop_size=224):
        """Tile small images instead of resizing (avoids interpolation artifacts)."""
        if min(img.size) < crop_size:
            w, h = img.size
            new_w = w * ceil(crop_size / w)
            new_h = h * ceil(crop_size / h)
            new_img = Image.new('RGB', (new_w, new_h))
            for i in range(0, new_w, w):
                for j in range(0, new_h, h):
                    new_img.paste(img, (i, j))
            return new_img
        return img

    transform = transforms.Compose([
        transforms.Lambda(lambda img: translate_duplicate(img)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    return (model, transform), device


@torch.no_grad()
def get_fakeness_score(image_path: str, model_tuple, device) -> float:
    """
    Computes a fakeness score for a single image.

    Returns:
        float in [0, 1] — higher means more likely AI-generated.
    """
    model, transform = model_tuple
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    logit = model(tensor)
    return torch.sigmoid(logit).item()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python npr_inference.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    print("Loading NPR Detector...")
    model_tuple, device = load_model()

    score = get_fakeness_score(image_path, model_tuple, device)
    print(f"Fakeness Score : {score:.4f}  ({'likely AI-generated' if score > 0.5 else 'likely real'})")
