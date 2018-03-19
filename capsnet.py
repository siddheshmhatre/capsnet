import torch
import torch.nn as nn

from torch.autograd import Variable

class CapsuleLayer(nn.Module):
    def __init__(self, num_route_weights, num_capsules, input_channels,
                 output_channels, kernel_size=9, stride=2, num_iterations=3):
        super(CapsuleLayer, self).__init__()

        self.num_route_weights = num_route_weights
        self.num_iterations = num_iterations
        self.output_channels = output_channels

        if num_route_weights != -1:
            # If digit caps
            self.routing_weights = nn.Parameter(torch.randn(num_capsules,
                                                            num_route_weights,
                                                            input_channels,
                                                            output_channels))
        else:
            # If PrimaryCapsules
            self.conv_caps = nn.ModuleList([nn.Conv2d(input_channels,
                                                      output_channels,
                                                      kernel_size=kernel_size,
                                                      stride=stride,
                                                      padding=0)
                                            for _ in range(num_capsules)])

    def forward(self, x):

        if self.num_route_weights != -1:
            pass
        else:
            # If Primary capsule
            outputs = [conv_cap(x).view(x.size(0), self.output_channels, -1, 1)
                       for conv_cap in self.conv_caps]
            outputs = torch.cat(outputs, dim=-1)
            outputs = outputs.transpose(-1, 1).contiguous()
            outputs = outputs.view(outputs.size(0), -1, self.output_channels)
            return outputs
