"""Dataset helpers for a user-provided folder of face images."""
from pathlib import Path
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class FaceFolderDataset(Dataset):
    def __init__(self, root, image_size=64):
        self.root = Path(root).expanduser()
        if not self.root.exists():
            raise FileNotFoundError(
                f"Dataset folder not found: {self.root}. Put face images in data/faces/."
            )
        self.files = sorted(
            p for p in self.root.rglob("*")
            if p.is_file() and p.suffix.lower() in EXTENSIONS
        )
        if not self.files:
            raise RuntimeError(
                f"No images found in {self.root}. Supported: {', '.join(sorted(EXTENSIONS))}"
            )
        self.transform = transforms.Compose([
            transforms.Resize(image_size + 8),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        try:
            with Image.open(path) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                return self.transform(image)
        except Exception as exc:
            raise RuntimeError(f"Could not read image {path}: {exc}") from exc
