import torch
from torch import nn
from torchvision import models


class resnet_model(nn.Module):
    def __init__(self, num_classes):
        super.__init__()
        self.num_classes = num_classes
        self.model = models.resnet18(pretrained = True)
        self.model.fc = nn.Linear(self.model.in_features, self.num_classes)

    def forward(self, x) -> torch.Tensor:
        return self.model(x)
    

    
