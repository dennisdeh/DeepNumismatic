import torch
import torchvision
from modules.loader import pytorch_loader

def _to_target(y):
    return int(y)

# Train model

def train_cnn(ds: dict, num_epochs: int = 5, lr: float = 1e-3,  print_every: int = 50):
    """
    Train a simple CNN on ds['train'] and validate on ds['validation'].
    ds: dict with keys 'train' and 'validation' mapping to DataLoaders that yield (image, label).
        - image: torch.Tensor [C,H,W]
        - label: anything convertible to class index or kept as path; if label is a path string,
                 classes will be inferred from its parent directory name.
    """
    assert "train" in ds and "validation" in ds and "labels" in ds, "ds must have 'train' and 'validation' loaders and 'labels'"

    # Peek one batch to infer input channels/size
    train_loader = ds["train"]
    val_loader = ds["validation"]
    first_batch = next(iter(train_loader))
    x0, y0 = first_batch
    in_channels = x0.shape[1] if x0.ndim == 4 else 3
    num_classes = len(ds["labels"])

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
        if isinstance(y, torch.Tensor) and y.dtype == torch.long:
            return y.to(device)
        if isinstance(y, (list, tuple)):
            return torch.tensor([_to_target(t) for t in y], dtype=torch.long, device=device)
        # scalar or single element
        return torch.tensor([_to_target(y)], dtype=torch.long, device=device)

    def run_epoch(loader, train: bool):
        if train:
            model.train()
        else:
            model.eval()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        with torch.set_grad_enabled(train):
            for step, (images, labels) in enumerate(loader):
                # images can be list/tuple or tensor depending on transform pipeline
                if isinstance(images, (list, tuple)):
                    images = torch.stack([img if isinstance(img, torch.Tensor) else torchvision.transforms.functional.to_tensor(img)
                                          for img in images], dim=0)
                images = images.to(device)
                targets = prepare_targets(labels)

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
        "model": model
    }

if __name__ == "__main__2":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Load images
    transformer = torchvision.transforms.Compose([torchvision.transforms.Resize(size=(100, 100)),
                                                  torchvision.transforms.ToTensor(),
                                                  torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    ds = pytorch_loader("data/RRC-60/Observe_test", transformer=transformer)

    # Train model
    out = train_cnn(ds=ds, num_epochs = 5, lr = 1e-3, print_every = 50)