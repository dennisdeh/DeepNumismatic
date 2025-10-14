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

    # Prepare datasets: keep PIL images; transformer will be applied lazily in __getitem__
    train_imgs = d["train"][0]
    val_imgs = d["validation"][0]

    # get target label from file path
    y_train = []
    y_val = []
    for x in d["train"][1]:
        y_train.append(x.split("/")[-2])
    for x in d["validation"][1]:
        y_val.append(x.split("/")[-2])

    # pair data and create a dataset (apply transformer lazily)
    train_dataset = _pair_dataset(train_imgs, y_train, transformer)
    val_dataset = _pair_dataset(val_imgs, y_val, transformer)

    data_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    data_val = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    return {"train": data_train,
            "validation": data_val,
            "labels": set(y_train + y_val)}


def visualise_batches(ds: dict, split: str = "train", max_images: int = 16, denormalise: bool = True,
                      mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    """
    Visualise a grid of images from the provided pytorch_loader output.

    Args:
        ds: dict with keys "train", "validation" (DataLoaders) and "labels".
        split: "train" or "validation".
        max_images: maximum number of images to show.
        denormalize: try to invert common normalisation before displaying.
        mean, std: tuples used for denormalisation if applicable.
    """
    assert split in ("train", "validation"), "split must be 'train' or 'validation'"
    loader = ds[split]

    try:
        images, labels = next(iter(loader))
    except StopIteration:
        print("Empty DataLoader.")
        return

    # If images are a list of PIL.Images or tensors, stack them
    if isinstance(images, (list, tuple)):
        images = torch.stack([
            img if isinstance(img, torch.Tensor)
            else torchvision.transforms.functional.to_tensor(img)
            for img in images
        ], dim=0)

    # number of images to plot and labels
    n = min(max_images, images.size(0))
    images = images[:n]
    if isinstance(labels, (list, tuple)):
        labels = list(labels)[:n]
    elif isinstance(labels, torch.Tensor):
        labels = labels[:n].tolist()
    else:
        labels = [labels] if n == 1 else [labels for _ in range(n)]

    # Denormalise if requested and looks like the normalised range
    imgs = images.detach().cpu()
    if denormalise and imgs.dtype in (torch.float16, torch.float32, torch.float64):
        # Heuristic: if values are outside [0,1], try inverse norm
        if imgs.min() < 0.0 or imgs.max() > 1.0:
            mean_t = torch.tensor(mean).view(1, 3, 1, 1)
            std_t = torch.tensor(std).view(1, 3, 1, 1)
            if imgs.size(1) == 3:
                imgs = imgs * std_t + mean_t
        imgs = imgs.clamp(0, 1)

    # Build titles as strings
    titles = []
    for y in labels:
        if isinstance(y, (list, tuple)):
            y = y[0]
        titles.append(str(y))

    # Determine grid structure and plot grid
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(4 * cols, 4 * rows))
    for i in range(n):
        plt.subplot(rows, cols, i + 1)
        img = imgs[i]
        if img.ndim == 3 and img.size(0) in (1, 3):
            if img.size(0) == 1:
                plt.imshow(img.squeeze(0).numpy(), cmap="gray")
            else:
                plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))
        else:
            arr = img.numpy()
            if arr.ndim == 3 and arr.shape[0] > 3:
                arr = np.transpose(arr[:3], (1, 2, 0))
            elif arr.ndim == 3 and arr.shape[-1] in (1, 3):
                pass
            plt.imshow(arr, cmap="gray" if arr.ndim == 2 or arr.shape[-1] == 1 else None)
        plt.title(titles[i], fontsize=16)
        plt.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__2":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    #out = load_images_from_path("data/RRC-60/Observe")
    img_size = (200, 200)
    n_channels = 3
    transformer = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=img_size),
        torchvision.transforms.CenterCrop(size=img_size),
        torchvision.transforms.Grayscale(n_channels),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(n_channels*(0.5,), n_channels*(0.5,))
    ])
    out = pytorch_loader(root_path="data/RRC-60/Observe_test",transformer=transformer,batch_size=16)
    # Visualise some batches
    visualise_batches(out, split="train", max_images=16, denormalise=True)