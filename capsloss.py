import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

class CapsuleLoss(nn.Module):
    def __init__(self, reconstruction=True, m_plus=0.9, m_minus=0.1, lamb=0.5):
        super(CapsuleLoss, self).__init__()
        if reconstruction:
            self.reconstruction_loss = nn.MSELoss()
        self.m_plus = m_plus
        self.m_minus = m_minus
        self.lamb = lamb

    def forward(self, output, labels, images, reconstructions):

        # Compute norm of output
        output = torch.norm(output, dim=-1).view(-1, output.size()[1])

        # Get the left side of the margin loss
        left = labels * (F.relu(self.m_plus - output) ** 2)

        # Get the right side of the margin loss
        right = (1 - labels) * (F.relu(output - self.m_minus) ** 2)

        # Compute the margin loss
        margin_loss = left + self.lamb * right
        margin_loss = margin_loss.sum()

        # Get reconstruction loss
        reconstructions = reconstructions.view(reconstructions.size()[0], -1)
        images = images.view(images.size()[0], -1)
        reconstruction_loss = self.reconstruction_loss(reconstructions, images)

        # Add up total weighted loss
        total_loss = 0.0005 * reconstruction_loss + margin_loss

        return total_loss / output.size()[0]
