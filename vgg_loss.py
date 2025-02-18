import torch
import torch.nn as nn
from torchvision.models import vgg19
from torchvision import transforms

class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super(VGGFeatureExtractor, self).__init__()
        vgg = vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg.features)[:35])
        # VGGの重みを固定
        for param in self.parameters():
            param.requires_grad = False
        self.feature_extractor.eval()

    def forward(self, x):
        return self.feature_extractor(x)

class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = VGGFeatureExtractor()
        self.criterion = nn.MSELoss()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])

    def forward(self, x, y):
        x = torch.stack([self.normalize(x_i) for x_i in x])
        y = torch.stack([self.normalize(y_i) for y_i in y])
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        return self.criterion(x_vgg, y_vgg)
