"""Local ImageNet subsets. Official validation is reserved for final evaluation."""
from pathlib import Path
import hashlib
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader


class FolderBenchmark(Dataset):
    def __init__(self, root, name, train=True):
        root = Path(root).expanduser()
        folder = {"tinyimagenet": "tiny-imagenet-200", "imagenette": "imagenette2-160"}[name]
        self.root = root if root.name == folder else root / folder
        self.name = name
        self.split = "train" if train else "val"
        if name == "tinyimagenet":
            self.classes = sorted((self.root / "wnids.txt").read_text().split())
            self.class_to_idx = {c:i for i,c in enumerate(self.classes)}
            if train:
                self.samples = [(str(p), self.class_to_idx[c]) for c in self.classes
                                for p in sorted((self.root / "train" / c / "images").glob("*.JPEG"))]
            else:
                lines = (self.root / "val/val_annotations.txt").read_text().splitlines()
                self.samples = sorted((str(self.root / "val/images" / parts[0]), self.class_to_idx[parts[1]])
                                      for parts in (line.split() for line in lines))
        else:
            self.classes = sorted(p.name for p in (self.root / "train").iterdir() if p.is_dir())
            self.class_to_idx = {c:i for i,c in enumerate(self.classes)}
            found = ImageFolder(self.root / self.split)
            if set(found.classes) != set(self.classes):
                raise ValueError("Train/validation class directories differ")
            self.samples = [(p,self.class_to_idx[found.classes[y]]) for p,y in found.samples]
        if not self.samples:
            raise ValueError(f"No images found in {self.root}, split={self.split}")
        self.targets = [y for _,y in self.samples]
        self.sample_manifest_sha256 = hashlib.sha256("\n".join(
            f"{Path(p).relative_to(self.root)}\t{y}" for p,y in self.samples).encode()).hexdigest()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path,label = self.samples[index]
        return default_loader(path),label
