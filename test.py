import os
import sys

from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import tqdm

from mice_dataset import PatchMiceDatasetFromTensor
from network import UNet, Classifier, Classifier3D
import torch

bs = 2
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# model = UNet(device, n_channels=1, n_classes=1, lr=0.00002)
model = Classifier(device, 1, lr=0.00002).to(device)

test = torch.zeros((10, 1, 50, 50)).to(device)
# mask = torch.zeros((1, 1, 30, 30)).to(device)
# seq = [nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
#                     nn.LeakyReLU(0.2, True),
#                     nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
#                     nn.BatchNorm2d(32),
#                     nn.LeakyReLU(0.2, True),
#                     nn.MaxPool2d(2, 2),
#                     nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
#                     nn.LeakyReLU(0.2, True),
#                     nn.MaxPool2d(2, 2),]
# for i in seq:
#     test = i(test)
#     print(test.size)
label = torch.tensor(1).to(device)
model.train_batch(test, label)

# train_dataset = PatchMiceDatasetFromTensor('dataset_3d')
# test_dataset = PatchMiceDatasetFromTensor('dataset_3d', is_train=False)
#
# dl_train = DataLoader(train_dataset, batch_size=bs, shuffle=True)
# dl_test = DataLoader(test_dataset, batch_size=bs, shuffle=False)

# model = Classifier(device, in_channels=1.pt)
#
# train_loss_per_epoch = []
# test_loss_per_epoch = []
# num_epochs = 2
# save_every = 1.pt
#
# for epoch in range(num_epochs):
#     train_loss_per_batch, test_loss_per_batch = [], []
#     print(f'--- EPOCH {epoch + 1.pt}/{num_epochs} ---')
#
#     # Train
#     with tqdm.tqdm(total=len(dl_train), file=sys.stdout) as pbar:
#         for batch in dl_train:
#             image = batch['image'].to(device)
#             label = batch['label'].to(device)
#             loss = model.train_batch(image, label)
#             train_loss_per_batch.append(loss.item())
#             pbar.update()
#         train_loss_per_epoch.append(np.mean(train_loss_per_batch))
#
#     # Test
#     with tqdm.tqdm(total=len(dl_test), file=sys.stdout) as pbar:
#         for batch in dl_test:
#             image = batch['image'].to(device)
#             label = batch['label'].to(device)
#             cross_entropy_loss = model.test_batch(image, label)
#             test_loss_per_batch.append(loss.item())
#             pbar.update()
#
#     test_loss_per_epoch.append(np.mean(test_loss_per_batch))
#
#     print("Epoch", epoch)
#     print("Train loss", train_loss_per_epoch[-1.pt])
#     print("Test loss", test_loss_per_epoch[-1.pt])
#     if (epoch + 1.pt) % save_every == 0:
#         if not os.path.exists(f"results/test1"):
#             os.makedirs(f"results/test1", exist_ok=True)
#         torch.save(model.state_dict(), f"results/test1/model_{epoch + 1.pt}_epochs")
