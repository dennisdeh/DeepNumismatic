import torch
import torchvision
from modules.loader import pytorch_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Train model

def train_cnn(ds: dict, num_epochs: int = 5, lr: float = 1e-3,  print_every: int = 50):
    """
    Train a simple CNN on ds['train'] and validate on ds['validation'].
    ds: dict with keys 'train' and 'validation' mapping to DataLoaders that yield (image, label).
    """
    assert "train" in ds and "validation" in ds and "labels" in ds, "ds must have 'train' and 'validation' loaders and 'labels'"

    # Build stable label mapping: class_name -> index in [0..K-1]
    class_names = sorted(list(ds["labels"]))
    label_to_idx = {name: i for i, name in enumerate(class_names)}
    num_classes = len(class_names)

    # Peek one batch to infer input channels/size
    train_loader = ds["train"]
    val_loader = ds["validation"]
    x0, y0 = next(iter(train_loader))
    in_channels = x0.shape[1] if isinstance(x0, torch.Tensor) and x0.ndim == 4 else 3

    # Simple CNN
    model = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(2),
        torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(2),
        torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
        torch.nn.ReLU(inplace=True),
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten(),
        torch.nn.Linear(64, num_classes),
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def prepare_targets(y):
        # y can be: tensor of longs already in 0..K-1, or list/tuple of strings, or scalar string
        if isinstance(y, torch.Tensor):
            if y.dtype == torch.long:
                return y.to(device)
            # If it's a tensor of strings (unlikely), fall through to generic handling
            y = y.tolist()
        if isinstance(y, (list, tuple)):
            mapped = []
            for t in y:
                if isinstance(t, str):
                    mapped.append(label_to_idx[t])
                elif isinstance(t, (int,)):
                    # If numeric, remap through class_names if possible, else assume already 0..K-1
                    mapped.append(int(t))
                else:
                    mapped.append(label_to_idx[str(t)])
            return torch.tensor(mapped, dtype=torch.long, device=device)
        # scalar
        if isinstance(y, str):
            return torch.tensor([label_to_idx[y]], dtype=torch.long, device=device)
        return torch.tensor([int(y)], dtype=torch.long, device=device)

    def run_epoch(loader, train: bool):
        model.train(mode=train)
        total_loss, total_correct, total_samples = 0.0, 0, 0
        with torch.set_grad_enabled(train):
            for step, (images, labels) in enumerate(loader):
                # images may be tensor or list of PIL/tensors
                if isinstance(images, (list, tuple)):
                    images = torch.stack([
                        img if isinstance(img, torch.Tensor)
                        else torchvision.transforms.functional.to_tensor(img)
                        for img in images
                    ], dim=0)
                images = images.to(device, non_blocking=True)
                targets = prepare_targets(labels)

                # Ensure shapes align
                if images.size(0) != targets.size(0):
                    raise RuntimeError(f"Batch size mismatch: images {images.size(0)} vs targets {targets.size(0)}")

                logits = model(images)
                loss = criterion(logits, targets)

                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                preds = logits.argmax(dim=1)
                batch_size = images.size(0)
                total_loss += loss.item() * batch_size
                total_correct += (preds == targets).sum().item()
                total_samples += batch_size

                if train and print_every and (step + 1) % print_every == 0:
                    print(f"Train step {step+1}: loss={loss.item():.4f}")

        avg_loss = total_loss / max(1, total_samples)
        acc = total_correct / max(1, total_samples)
        return avg_loss, acc

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = run_epoch(train_loader, train=True)
        val_loss, val_acc = run_epoch(val_loader, train=False)
        print(f"Epoch {epoch}/{num_epochs} - "
              f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f} | "
              f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")

    return {
        "model": model,
        "classes": class_names,
        "label_to_idx": label_to_idx,
    }

if __name__ == "__main__2":
    print(device)
    transformer = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(200, 200)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    ds = pytorch_loader("data/RRC-60/Observe", transformer=transformer)
    out = train_cnn(ds=ds, num_epochs=5, lr=1e-3, print_every=50)