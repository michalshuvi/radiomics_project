import itertools

import torch
import torch.nn as nn
from torch.optim import lr_scheduler

from unet_parts import *


class Classifier(nn.Module):
    def __init__(self, device, in_channels, lr=0.00002, scheduler_params=None):
        super(Classifier, self).__init__()
        self.device = device
        sequence = [nn.Conv2d(in_channels, 16, kernel_size=2, stride=2, padding=1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(32),
                    nn.LeakyReLU(0.2, True),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=1, bias=False),
                    nn.LeakyReLU(0.2, True),
                    nn.MaxPool2d(2, 2),
                    ]

        self.fc1 = nn.Linear(3136, 1500)
        self.fc2 = nn.Linear(1500, 750)
        self.fc3 = nn.Linear(750, 2)
        self.model = nn.Sequential(*sequence)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            itertools.chain(self.model.parameters(), self.fc1.parameters(),
                            self.fc2.parameters(), self.fc3.parameters()),
            lr=lr, betas=(0.99, 0.999), weight_decay=1e-5)
        if scheduler_params:
            self.scheduler = get_scheduler(self.optimizer, **scheduler_params)

    def forward(self, x, extract=False):
        relu = nn.ReLU()
        y = self.model(x)
        y = y.view(-1, 3136)
        y = relu(self.fc1(y))
        y = self.fc2(y)
        if extract:
            return y
        y = relu(y)
        y = self.fc3(y)
        return y

    def train_batch(self, x, y):
        self.optimizer.zero_grad()
        y_pred = self.forward(x)
        loss = self.criterion(y_pred, y)
        loss.backward()
        self.optimizer.step()
        return loss

    def test_batch(self, x, y):
        with torch.no_grad():
            y_pred = self.forward(x)
            cross_entropy_loss = self.criterion(y_pred, y)
            mistakes_num = torch.sum(abs(y - torch.argmax(y_pred, dim=1))).item()
            return cross_entropy_loss, mistakes_num

    def extract_features(self, x):
        with torch.no_grad():
            return self.forward(x, extract=True)

    def update_learning_rate(self):
        old_lr = self.optimizer.param_groups[0]['lr']
        self.scheduler.step()
        lr = self.optimizer.param_groups[0]['lr']
        if lr != old_lr:
            print(f'learning rate changed from {old_lr:.7f} to {lr:.7f}')


class Classifier3D(nn.Module):
    def __init__(self, device, in_channels, lr=0.00002, scheduler_params=None):
        super(Classifier3D, self).__init__()
        self.device = device
        sequence = [nn.Conv3d(in_channels, 16, kernel_size=2, stride=2, padding=1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv3d(16, 32, kernel_size=2, stride=1, padding=1, bias=False),
                    nn.BatchNorm3d(32),
                    nn.LeakyReLU(0.2, True),
                    nn.MaxPool3d(2, 2),
                    nn.Conv3d(32, 64, kernel_size=2, stride=1, padding=1, bias=False),
                    nn.LeakyReLU(0.2, True),
                    nn.MaxPool3d(2, 2),
                    ]

        self.fc1 = nn.Linear(4096, 2045)
        self.fc2 = nn.Linear(2045, 1024)
        self.fc3 = nn.Linear(1024, 2)
        self.model = nn.Sequential(*sequence)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            itertools.chain(self.model.parameters(), self.fc1.parameters(), self.fc2.parameters(), self.fc3.parameters()),
            lr=lr, betas=(0.99, 0.999))
        if scheduler_params:
            self.scheduler = get_scheduler(self.optimizer, **scheduler_params)

    def forward(self, x, extract=False):
        y = self.model(x)
        y = y.view(-1, 4096)
        y = nn.ReLU()(self.fc1(y))
        y = self.fc2(y)
        if not extract:
            y = nn.ReLU()(y)
            y = self.fc3(y)
        return y

    def train_batch(self, x, y):
        self.optimizer.zero_grad()
        y_pred = self.forward(x)
        loss = self.criterion(y_pred, y)
        loss.backward()
        self.optimizer.step()
        return loss

    def test_batch(self, x, y):
        with torch.no_grad():
            y_pred = self.forward(x)
            cross_entropy_loss = self.criterion(y_pred, y)
            mistakes_num = torch.sum(abs(y - torch.argmax(y_pred, dim=1))).item()
            return cross_entropy_loss, mistakes_num

    def extract_features(self, x):
        with torch.no_grad():
            return self.forward(x, True)

    def update_learning_rate(self):
        old_lr = self.optimizer.param_groups[0]['lr']
        self.scheduler.step()
        lr = self.optimizer.param_groups[0]['lr']
        if lr != old_lr:
            print(f'learning rate changed from {old_lr:.7f} to {lr:.7f}')


class UNet(nn.Module):
    def __init__(self, device, n_channels, n_classes, layers, criterion, optimizer, bilinear=True, lr=0.0002,
                 scheduler_params=None):
        super(UNet, self).__init__()
        self.device = device
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.n_layers = len(layers) - 1

        encoder = [DoubleConv(n_channels, layers[0])]
        for i in range(len(layers) - 1):
            if i == len(layers) - 2 and bilinear:
                factor = 2
            else:
                factor = 1
            encoder += [Down(layers[i], layers[i + 1] // factor)]
        decoder = []
        for i in range(len(layers) - 1):
            if i == len(layers) - 2:
                factor = 1
            decoder += [Up(layers[-i - 1], layers[-i - 2] // factor, bilinear)]
        decoder += [OutConv(layers[0], n_classes)]

        self.encoder = nn.Sequential(*encoder).to(self.device)
        self.decoder = nn.Sequential(*decoder).to(self.device)

        self.criterion = criterion
        self.optimizer = optimizer(itertools.chain(self.encoder.parameters(), self.decoder.parameters()),
                                   lr=lr, betas=(0.99, 0.999))
        self.scheduler = None
        if scheduler_params:
            self.scheduler = get_scheduler(self.optimizer, **scheduler_params)

    def forward(self, x):
        x_list = [x]
        x_list.append(self.encoder[0](x))
        for i in range(self.n_layers):
            x_list.append(self.encoder[i + 1](x_list[-1]))
        for i in range(self.n_layers):
            x1 = x_list[-(i + 1)]
            if not i == 0:
                x1 = x
            x = self.decoder[i](x1, x_list[-(i + 2)])
        logits = self.decoder[-1](x)
        return torch.sigmoid(logits)

    def extract_features(self, x):
        with torch.no_grad():
            return torch.flatten(self.encoder(x))

    def train_batch(self, x, y):
        self.optimizer.zero_grad()
        y_pred = self.forward(x)
        loss = self.criterion(y_pred, y)
        loss.backward()
        self.optimizer.step()
        return loss

    def test_batch(self, x, y):
        with torch.no_grad():
            y_pred = self.forward(x)
            loss = self.criterion(y_pred, y)
            iou = calc_iou(y, y_pred, self.device)
            return loss.item(), iou

    def update_learning_rate(self):
        old_lr = self.optimizer.param_groups[0]['lr']
        self.scheduler.step()
        lr = self.optimizer.param_groups[0]['lr']
        if lr != old_lr:
            print(f'learning rate changed from {old_lr:.7f} to {lr:.7f}')


def get_scheduler(optimizer, total_epochs_num=200, static_epochs_num=100):
    decay_epochs_num = total_epochs_num - static_epochs_num
    return lr_scheduler.LambdaLR(optimizer,
                                 lr_lambda=(lambda epoch: 1.0 - max(0, epoch - static_epochs_num) / float(decay_epochs_num + 1)))


def calc_iou(y_true, y_pred, device):
    size = y_true.size()
    y_pred_mask = torch.where(y_pred > 0.5, torch.ones(size, device=device), torch.zeros(size, device=device))
    intersection = y_true * y_pred_mask
    union = torch.where(y_true + y_pred_mask > 0, torch.ones(size, device=device), torch.zeros(size, device=device))
    intersection_count = torch.sum(intersection).item()
    union_count = torch.sum(union).item()
    iou = intersection_count / union_count
    return iou


def dice_loss(pred, target):
    """This definition generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """

    smooth = 1.

    # have to use contiguous since they may from a torch.view op
    i_flat = pred.contiguous().view(-1)
    t_flat = target.contiguous().view(-1)
    intersection = (i_flat * t_flat).sum()

    a_sum = torch.sum(i_flat * i_flat)
    b_sum = torch.sum(t_flat * t_flat)

    return 1 - ((2. * intersection + smooth) / (a_sum + b_sum + smooth))