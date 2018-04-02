import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms

from torch.autograd import Variable
from capslayer import CapsuleLayer
from capsloss import CapsuleLoss
from capsnet import CapsNet

USE_CUDA = torch.cuda.is_available()
EPOCHS = 5
BATCH_SIZE = 64

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
            labels = torch.sparse.torch.eye(10).index_select(dim=0, index=labels)

            if USE_CUDA:
                images, labels = images.cuda(), labels.cuda()

            images, labels = Variable(images), Variable(labels)
            optimizer.zero_grad()
            x, reconstructions = net(images, labels)
            caps_loss = loss(x, labels, images, reconstructions)
            caps_loss.backward()
            optimizer.step()

            print ('{}/{} Loss: {}'.format(i, len(loader), caps_loss.data))

        test(net, loss)

def test(net, loss):
    net.eval()
    dataset = torchvision.datasets.MNIST(root='./data', train=False,
                                         transform=transforms.ToTensor())

    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                                         num_workers=2)

    total_loss = 0
    for i, data in enumerate(loader):
        images, labels = data
        labels = torch.sparse.torch.eye(10).index_select(dim=0, index=labels)
        if USE_CUDA:
            images, labels = images.cuda(), labels.cuda()

        images, labels = Variable(images, volatile=True), Variable(labels)
        x, reconstructions = net(images, labels)
        caps_loss = loss(x, labels, images, reconstructions)
        total_loss += caps_loss

    print ('Loss on test set', total_loss / len(loader))

if __name__ == "__main__":
    main()
