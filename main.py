import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms

from torch.autograd import Variable
from torchvision.utils import save_image
from capslayer import CapsuleLayer
from capsloss import CapsuleLoss
from capsnet import CapsNet

USE_CUDA = torch.cuda.is_available()
EPOCHS = 50
BATCH_SIZE = 64
CHECKPOINT = "model"

def main():
    dataset = torchvision.datasets.MNIST(root='./data', train=True,
                                         download=True,
                                         transform=transforms.ToTensor())

    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                                         num_workers=2)

    net = CapsNet()

    if USE_CUDA:
        net.cuda()

    loss = CapsuleLoss()
    optimizer = optim.Adam(net.parameters())

    for j in range(EPOCHS):
        for i, data in enumerate(loader):
            images, labels = data
            labels = torch.eye(10).index_select(dim=0, index=labels)

            if USE_CUDA:
                images, labels = images.cuda(), labels.cuda()

            images, labels = Variable(images), Variable(labels)
            optimizer.zero_grad()
            x, reconstructions = net(images, labels)
            caps_loss = loss(x, labels, images, reconstructions)
            caps_loss.backward()
            optimizer.step()

            print ('Epoch: {} {}/{} Loss: {}'.format(j, i, len(loader), caps_loss.data))

        test(net, loss)
        torch.save(net.state_dict(), 'model')

def test(net, loss):
    net.eval()
    dataset = torchvision.datasets.MNIST(root='./data', train=False,
                                         transform=transforms.ToTensor())

    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                                         num_workers=2)

    total_loss = 0
    correct_preds = 0
    for i, data in enumerate(loader):
        images, labels = data
        labels = torch.eye(10).index_select(dim=0, index=labels)
        if USE_CUDA:
            images, labels = images.cuda(), labels.cuda()

        images, labels = Variable(images, volatile=True), Variable(labels)
        x, reconstructions = net(images, labels)

        if i == 0:
            save_image(images.data, 'original_images.jpg')
            reconstructions = reconstructions.view(-1, 1, 28, 28)
            save_image(reconstructions.data, 'reconstructions.jpg')
        caps_loss = loss(x, labels, images, reconstructions)
        total_loss += caps_loss

        # Compute accuracy
        x = torch.norm(x, dim=-1).view(-1, x.size()[1])
        _, preds = x.data.max(dim=1)
        _, labels = labels.max(dim=1)
        correct_preds += preds.eq(labels.data.view_as(preds)).cpu().sum()

    print ('Accuracy: {}'.format((correct_preds/len(dataset)) * 100))

if __name__ == "__main__":
    main()
