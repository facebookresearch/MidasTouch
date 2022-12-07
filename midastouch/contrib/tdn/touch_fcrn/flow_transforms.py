# Original source: https://github.com/XPFly1989/FCRN
# A Pytorch implementation of Laina, Iro, et al. "Deeper depth prediction with fully convolutional residual networks."
# 3D Vision (3DV), 2016 Fourth International Conference on. IEEE, 2016.

# Copyright (c) 2016, Iro Laina
# All rights reserved.

# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
# Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Copyright (c) Meta Platforms, Inc. All Rights Reserved

from __future__ import division
import torch
import random
import numpy as np
import numbers
import types
import scipy.ndimage as ndimage

"""Set of tranform random routines that takes both input and target as arguments,
in order to have random but coherent transformations.
inputs are PIL Image pairs and targets are ndarrays"""


class Compose(object):
    """Composes several co_transforms together.
    For example:
    >>> co_transforms.Compose([
    >>>     co_transforms.CenterCrop(10),
    >>>     co_transforms.ToTensor(),
    >>>  ])
    """

    def __init__(self, co_transforms):
        self.co_transforms = co_transforms

    def __call__(self, input, target_depth, target_label=None):
        for i, t in enumerate(self.co_transforms):
            # print('After transform {}'.format(i))
            # print(np.max(target_label))
            if target_label == None:
                input, target_depth, _ = t(input, target_depth, target_depth)
                return input, target_depth
            else:
                input, target_depth, target_label = t(input, target_depth, target_label)
                return input, target_depth, target_label


class ArrayToTensor(object):
    """Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)."""

    def __call__(self, array):
        assert isinstance(array, np.ndarray)
        # handle numpy array
        try:
            tensor = torch.from_numpy(array).permute(2, 0, 1)
        except:
            tensor = torch.from_numpy(np.expand_dims(array, axis=2)).permute(2, 0, 1)
        # put it from HWC to CHW format
        return tensor.float()


class Lambda(object):
    """Applies a lambda as a transform"""

    def __init__(self, lambd):
        assert type(lambd) is types.LambdaType
        self.lambd = lambd

    def __call__(self, input, target):
        return self.lambd(input, target)


class CenterCrop(object):
    """Crops the given inputs and target arrays at the center to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    Careful, img1 and img2 may not be the same size
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, inputs, target):
        h1, w1, _ = inputs[0].shape
        h2, w2, _ = inputs[1].shape
        th, tw = self.size
        x1 = int(round((w1 - tw) / 2.0))
        y1 = int(round((h1 - th) / 2.0))
        x2 = int(round((w2 - tw) / 2.0))
        y2 = int(round((h2 - th) / 2.0))

        inputs[0] = inputs[0][y1 : y1 + th, x1 : x1 + tw]
        inputs[1] = inputs[1][y2 : y2 + th, x2 : x2 + tw]
        target = target[y1 : y1 + th, x1 : x1 + tw]
        return inputs, target


class Scale_Single(object):
    """Rescales a single object, for example only the ground truth dpeth map"""

    def __init__(self, size, order=2):
        self.size = size
        self.order = order

    def __call__(self, inputs):
        h, w = inputs.shape

        if (w <= h and w == self.size) or (h <= w and h == self.size):
            return inputs

        if w < h:
            ratio = self.size / w
        else:
            ratio = self.size / h

        inputs = ndimage.interpolation.zoom(inputs, ratio, order=self.order)

        return inputs


class Scale(object):
    """Rescales the inputs and target arrays to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation order: Default: 2 (bilinear)
    """

    def __init__(self, size, order=2):
        self.size = size
        self.order = order

    def __call__(self, inputs, target_depth=None, target_label=None):
        h, w, _ = inputs.shape

        if (w <= h and w == self.size) or (h <= w and h == self.size):
            if target_depth is not None and target_label is not None:
                return inputs, target_depth, target_label
            elif target_depth is not None:
                return inputs, target_depth
            elif target_label is not None:
                return inputs, target_label

        if w < h:
            ratio = self.size / w
        else:
            ratio = self.size / h

        inputs = np.stack(
            (
                ndimage.interpolation.zoom(inputs[:, :, 0], ratio, order=self.order),
                ndimage.interpolation.zoom(inputs[:, :, 1], ratio, order=self.order),
                ndimage.interpolation.zoom(inputs[:, :, 2], ratio, order=self.order),
            ),
            axis=2,
        )

        if target_label is not None and target_depth is not None:

            target_label = ndimage.interpolation.zoom(
                target_label, ratio, order=self.order
            )
            target_depth = ndimage.interpolation.zoom(
                target_depth, ratio, order=self.order
            )
            return inputs, target_depth, target_label

        elif target_depth is not None:
            target_depth = ndimage.interpolation.zoom(
                target_depth, ratio, order=self.order
            )
            return inputs, target_depth

        elif target_label is not None:
            target_label = ndimage.interpolation.zoom(
                target_label, ratio, order=self.order
            )
            return inputs, target_label

        else:
            return inputs


class RandomCrop(object):
    """Crops the given PIL.Image at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, inputs, target_depth, target_label):
        h, w, _ = inputs.shape
        th, tw = self.size
        if w == tw and h == th:
            return inputs, target_depth, target_label

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        inputs = inputs[y1 : y1 + th, x1 : x1 + tw]
        return (
            inputs,
            target_depth[y1 : y1 + th, x1 : x1 + tw],
            target_label[y1 : y1 + th, x1 : x1 + tw],
        )


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5"""

    def __call__(self, inputs, target_depth, target_label):
        if random.random() < 0.5:
            inputs = np.flip(inputs, axis=0).copy()
            target_depth = np.flip(target_depth, axis=0).copy()
            target_label = np.flip(target_label, axis=0).copy()
        return inputs, target_depth, target_label


class RandomVerticalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5"""

    def __call__(self, inputs, target):
        if random.random() < 0.5:
            inputs[0] = np.flipud(inputs[0])
            inputs[1] = np.flipud(inputs[1])
            target = np.flipud(target)
            target[:, :, 1] *= -1
        return inputs, target


class RandomRotate(object):
    """Random rotation of the image from -angle to angle (in degrees)
    This is useful for dataAugmentation, especially for geometric problems such as FlowEstimation
    angle: max angle of the rotation
    interpolation order: Default: 2 (bilinear)
    reshape: Default: false. If set to true, image size will be set to keep every pixel in the image.
    diff_angle: Default: 0. Must stay less than 10 degrees, or linear approximation of flowmap will be off.
    """

    def __init__(self, angle, diff_angle=0, order=2, reshape=False):
        self.angle = angle
        self.reshape = reshape
        self.order = order

    def __call__(self, inputs, target_depth, target_label):
        applied_angle = random.uniform(-self.angle, self.angle)
        angle1 = applied_angle
        angle1_rad = angle1 * np.pi / 180

        inputs = ndimage.interpolation.rotate(
            inputs, angle1, reshape=self.reshape, order=self.order
        )
        target_depth = ndimage.interpolation.rotate(
            target_depth, angle1, reshape=self.reshape, order=self.order
        )
        target_label = ndimage.interpolation.rotate(
            target_label, angle1, reshape=self.reshape, order=self.order
        )

        return inputs, target_depth, target_label


class RandomCropRotate(object):
    """Random rotation of the image from -angle to angle (in degrees)
    A crop is done to keep same image ratio, and no black pixels
    angle: max angle of the rotation, cannot be more than 180 degrees
    interpolation order: Default: 2 (bilinear)
    """

    def __init__(self, angle, size, diff_angle=0, order=2):
        self.angle = angle
        self.order = order
        self.diff_angle = diff_angle
        self.size = size

    def __call__(self, inputs, target):
        applied_angle = random.uniform(-self.angle, self.angle)
        diff = random.uniform(-self.diff_angle, self.diff_angle)
        angle1 = applied_angle - diff / 2
        angle2 = applied_angle + diff / 2

        angle1_rad = angle1 * np.pi / 180
        angle2_rad = angle2 * np.pi / 180

        h, w, _ = inputs[0].shape

        def rotate_flow(i, j, k):
            return -k * (j - w / 2) * (diff * np.pi / 180) + (1 - k) * (i - h / 2) * (
                diff * np.pi / 180
            )

        rotate_flow_map = np.fromfunction(rotate_flow, target.shape)
        target += rotate_flow_map

        inputs[0] = ndimage.interpolation.rotate(
            inputs[0], angle1, reshape=True, order=self.order
        )
        inputs[1] = ndimage.interpolation.rotate(
            inputs[1], angle2, reshape=True, order=self.order
        )
        target = ndimage.interpolation.rotate(
            target, angle1, reshape=True, order=self.order
        )

        # flow vectors must be rotated too!
        target_ = np.array(target, copy=True)
        target[:, :, 0] = (
            np.cos(angle1_rad) * target_[:, :, 0]
            - np.sin(angle1_rad) * target_[:, :, 1]
        )
        target[:, :, 1] = (
            np.sin(angle1_rad) * target_[:, :, 0]
            + np.cos(angle1_rad) * target_[:, :, 1]
        )

        # keep angle1 and angle2 within [0,pi/2] with a reflection at pi/2: -1rad is 1rad, 2rad is pi - 2 rad
        angle1_rad = np.pi / 2 - np.abs(angle1_rad % np.pi - np.pi / 2)
        angle2_rad = np.pi / 2 - np.abs(angle2_rad % np.pi - np.pi / 2)

        c1 = np.cos(angle1_rad)
        s1 = np.sin(angle1_rad)
        c2 = np.cos(angle2_rad)
        s2 = np.sin(angle2_rad)
        c_diag = h / np.sqrt(h * h + w * w)
        s_diag = w / np.sqrt(h * h + w * w)

        ratio = c_diag / max(c1 * c_diag + s1 * s_diag, c2 * c_diag + s2 * s_diag)

        crop = CenterCrop((int(h * ratio), int(w * ratio)))
        scale = Scale(self.size)
        inputs, target = crop(inputs, target)
        return scale(inputs, target)


class RandomTranslate(object):
    def __init__(self, translation):
        if isinstance(translation, numbers.Number):
            self.translation = (int(translation), int(translation))
        else:
            self.translation = translation

    def __call__(self, inputs, target):
        h, w, _ = inputs[0].shape
        th, tw = self.translation
        tw = random.randint(-tw, tw)
        th = random.randint(-th, th)
        if tw == 0 and th == 0:
            return inputs, target
        # compute x1,x2,y1,y2 for img1 and target, and x3,x4,y3,y4 for img2
        x1, x2, x3, x4 = max(0, tw), min(w + tw, w), max(0, -tw), min(w - tw, w)
        y1, y2, y3, y4 = max(0, th), min(h + th, h), max(0, -th), min(h - th, h)

        inputs[0] = inputs[0][y1:y2, x1:x2]
        inputs[1] = inputs[1][y3:y4, x3:x4]
        target = target[y1:y2, x1:x2]
        target[:, :, 0] += tw
        target[:, :, 1] += th

        return inputs, target
