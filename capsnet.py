import torch
import torch.nn as nn
import torch.nn.functional as F

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

            # Multiply output of primarycaps with weight matrices for digitcaps
            u = x[:, None, :, None, :] @ self.routing_weights[None, :, :, :, :]

            logits = Variable(torch.zeros(*u.size()))

            # Dynamic routing
            for i in range(self.num_iterations):
                probs = self.softmax(logits)
                outputs = (probs * u).sum(dim=2, keepdim=True)
                outputs = self.squash(outputs)

                if i < self.num_iterations:
                    change = (outputs * probs).sum(dim=-1, keepdim=True)

                    # update logits based on dot product between u and v
                    logits = logits + change

            return outputs
        else:
            # If Primary capsule
            outputs = [conv_cap(x).view(x.size(0), self.output_channels, -1, 1)
                       for conv_cap in self.conv_caps]
            outputs = torch.cat(outputs, dim=-1)
            outputs = outputs.transpose(-1, 1).contiguous()
            outputs = outputs.view(outputs.size(0), -1, self.output_channels)
            return outputs

    def softmax(self, x, dim=1):
        x_t = x.transpose(dim, len(x.size()) - 1)
        softmax_output = F.softmax(x_t.contiguous().view(-1, x_t.size(-1)), dim=-1)
        return softmax_output.view(*x_t.size()).transpose(dim, len(x_t.size()) - 1)

    def squash(self, x, dim=2):
        sqrd_norm = (x ** 2).sum(dim=dim, keepdim=True)
        scale = sqrd_norm / (1 + sqrd_norm)
        return scale * (x / torch.sqrt(sqrd_norm))
