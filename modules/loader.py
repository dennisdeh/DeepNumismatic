import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from PIL import Image


def load_images_from_path(root_path: str, valid_exts=(".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff")):
    """
    Recursively loads all images from root_path and its subfolders.

    Returns:
        images: list[PIL.Image.Image] – loaded images (RGB)
        file_paths: list[str] – absolute file paths matching images list order
    """
    root = Path(os.path.abspath(root_path))
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Path does not exist or is not a directory: {root_path}")

    file_paths = []
    for ext in valid_exts:
        file_paths.extend(str(p) for p in root.rglob(f"*{ext}"))

    # De-duplicate and sort for determinism
    file_paths = sorted(set(file_paths))

    images = []
    for fp in file_paths:
        try:
            img = Image.open(fp).convert("RGB")
            images.append(img)
        except Exception:
            # Skip unreadable files
            continue

    # Keep paths only for successfully loaded images
    loaded_paths = []
    i = 0
    for fp in file_paths:
        if i < len(images):
            loaded_paths.append(fp)
            i += 1
        else:
            break

    return images, loaded_paths

def load_images_from_folder_train_test(root_path: str, split:float=0.9, valid_exts=(".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"), seed:int=42):
    tp = load_images_from_path(root_path, valid_exts)
    assert len(tp[0])==len(tp[1])
    n_images = len(tp[0])

    # create validation/train split
    np.random.seed(seed)
    idx = np.random.permutation(n_images)
    n_train = int(split*n_images)
    train_idx = idx[:n_train].tolist()
    val_idx = idx[n_train:].tolist()

    return {"train": ([tp[0][i] for i in train_idx],  [tp[1][i] for i in train_idx]),
            "validation": ([tp[0][i] for i in val_idx],  [tp[1][i] for i in val_idx])}

def _pair_dataset(images, labels, transformer=None):
    """
    Zips images with labels and applies transformer lazily in __getitem__.
    Labels here are file paths; they are kept aligned during shuffling by DataLoader.
    """
    class ImgPathDataset(torch.utils.data.Dataset):
        def __init__(self, imgs, ys, tfm):
            self.imgs = imgs
            self.ys = ys
            self.tfm = tfm

        def __len__(self):
            return len(self.imgs)

        def __getitem__(self, idx):
            x = self.imgs[idx]
            if self.tfm is not None:
                x = self.tfm(x)
            y = self.ys[idx]
            return x, y

    return ImgPathDataset(images, labels, transformer)

def pytorch_loader(root_path: str,batch_size:int=1, transformer=None, split:float=0.9, valid_exts=(".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"), seed:int=42):
    d = load_images_from_folder_train_test(root_path=root_path, split=split, valid_exts=valid_exts, seed=seed)

    # Prepare datasets: apply transformer (if provided) to PIL images before batching
    train_imgs = d["train"][0]
    val_imgs = d["validation"][0]

    if transformer is not None:
        # torchvision transforms expect PIL.Image or tensor; our inputs are PIL.Image
        train_imgs = [transformer(img) for img in train_imgs]
        val_imgs = [transformer(img) for img in val_imgs]

    # get target label from file path
    y_train = []
    y_val = []
    for x in d["train"][1]:
        y_train.append(x.split("/")[-2])
    for x in d["validation"][1]:
        y_val.append(x.split("/")[-2])

    # pair data and create a dataset
    train_dataset = _pair_dataset(train_imgs, y_train, transformer)
    val_dataset = _pair_dataset(val_imgs, y_val, transformer)

    data_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    data_val = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    return {"train": data_train,
            "validation": data_val}


if __name__ == "__main__2":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    #out = load_images_from_path("data/RRC-60/Observe")
    transformer = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                  torchvision.transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
    out = pytorch_loader(root_path="data/RRC-60/Observe_test",transformer=transformer)