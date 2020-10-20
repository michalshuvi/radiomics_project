from matplotlib import pyplot as plt
import torch
import numpy as np


import math

from torch.nn import init


def tensors_as_images(tensors, nrows=1, figsize=(8, 8), titles=[],
                      wspace=0.1, hspace=0.2, cmap=None):
    """
    Plots a sequence of pytorch tensors as images.

    :param tensors: A sequence of pytorch tensors, should have shape CxWxH
    """
    assert nrows > 0

    num_tensors = len(tensors)

    ncols = math.ceil(num_tensors / nrows)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize,
                             gridspec_kw=dict(wspace=wspace, hspace=hspace),
                             subplot_kw=dict(yticks=[], xticks=[]))
    axes_flat = axes.reshape(-1)

    # Plot each tensor
    for i in range(num_tensors):
        ax = axes_flat[i]

        image_tensor = tensors[i]
        assert image_tensor.dim() == 3  # Make sure shape is CxWxH

        image = image_tensor.detach().numpy()
        image = image.transpose(1, 2, 0)
        image = image.squeeze()  # remove singleton dimensions if any exist

        # Scale to range 0..1
        min, max = np.min(image), np.max(image)
        image = (image-min) / (max-min)

        ax.imshow(image, cmap=cmap)

        if len(titles) > i and titles[i] is not None:
            ax.set_title(titles[i])

    # If there are more axes than tensors, remove their frames
    for j in range(num_tensors, len(axes_flat)):
        axes_flat[j].axis('off')

    return fig, axes


def init_model_weights(model, init_type='normal', init_var=0.02):

    def init_func(sub):
        classname = sub.__class__.__name__
        if hasattr(sub, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(sub.weight.data, 0.0, init_var)
            elif init_type == 'xavier':
                init.xavier_normal_(sub.weight.data, gain=init_var)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(sub, 'bias') and sub.bias is not None:
                init.constant_(sub.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(sub.weight.data, 1.0, init_var)
            init.constant_(sub.bias.data, 0.0)

    model.apply(init_func)