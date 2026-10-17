"""Classify a single image with the trained model.

Usage:
    python predict.py path/to/image.jpg
"""

import argparse

import torch
from PIL import Image
from torchvision import transforms

from model import SimpleCNN


def load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = SimpleCNN(num_classes=len(ckpt["classes"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def predict(model, ckpt, image: Image.Image, device: torch.device, top_k: int = 3):
    tf = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(ckpt["mean"], ckpt["std"]),
    ])
    x = tf(image.convert("RGB")).unsqueeze(0).to(device)
    probs = torch.softmax(model(x), dim=1)[0]
    values, indices = probs.topk(top_k)
    return [(ckpt["classes"][i], v.item()) for v, i in zip(values, indices)]


def main():
    parser = argparse.ArgumentParser(description="Classify an image")
    parser.add_argument("image")
    parser.add_argument("--checkpoint", default="models/cifar10_cnn.pt")
    args = parser.parse_args()

    device = torch.device("cpu")
    model, ckpt = load_model(args.checkpoint, device)
    for label, prob in predict(model, ckpt, Image.open(args.image), device):
        print(f"{label:12s} {prob:.1%}")


if __name__ == "__main__":
    main()
