import torch
import torch.nn as nn

from torch.autograd import Variable
from capslayer import CapsuleLayer
from capsloss import CapsuleLoss

class CapsNet(nn.Module):
    def __init__(self, num_classes=10):
        super(CapsNet, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=9, stride=1)
        self.caps_layer = CapsuleLayer(num_route_weights=-1,
                                       num_capsules=32,
                                       input_channels=256,
                                       output_channels=8)
        self.digit_caps = CapsuleLayer(num_route_weights=32*6*6,
                                       num_capsules=10,
                                       input_channels=8,
                                       output_channels=16)
        self.reconstruction = nn.Sequential(nn.Linear(num_classes*16,
                                                      512),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(512, 1024),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(1024, 784),
                                            nn.Sigmoid())

    def forward(self, x, y=None):
        x = self.conv1(x)
        x = self.caps_layer(x)
        x = self.digit_caps(x)

        reshape_x = x.view(-1, self.num_classes, 16)

        if y is None:
            # Get y which has max vector length
            pass

        mask = y.unsqueeze(dim=-1)
        masked_x = reshape_x * mask

        reconstruction = self.reconstruction(masked_x.view(masked_x.size()[0], -1))

        return x, reconstruction
