import torch
import torchvision
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F
from torch.nn import init
s

class ResNet50(nn.Module):
    def __init__(self, res4_stride=1, **kwargs):
        super().__init__()

        resnet50 = torchvision.models.resnet50(pretrained=True)
        if res4_stride == 1:
            resnet50.layer4[0].conv2.stride = (1, 1)
            resnet50.layer4[0].downsample[0].stride = (1, 1)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])

        self.bn = nn.BatchNorm1d(2048)
        init.normal_(self.bn.weight.data, 1.0, 0.02)
        init.constant_(self.bn.bias.data, 0.0)

    def forward(self, x):
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        f = self.bn(x)

        return f


class NormalizedClassifier(nn.Module):
    def __init__(self, num_classes, feature_dim=2048):
        super().__init__()
        self.weight = Parameter(torch.Tensor(num_classes, feature_dim))
        self.weight.data.uniform_(-1, 1).renorm_(2,0,1e-5).mul_(1e5)

    def forward(self, x):
        w = self.weight

        x = F.normalize(x, p=2, dim=1)
        w = F.normalize(w, p=2, dim=1)

        return F.linear(x, w)
