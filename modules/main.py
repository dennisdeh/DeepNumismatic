import torch
import torchvision
from modules.loader import pytorch_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Load images
transformer = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                              torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
out = pytorch_loader("data/RRC-60/Observe_test", transformer=transformer)

# Train model
model = torchvision.models.resnet18(pretrained=True)
model.to(device)
