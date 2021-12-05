# compatible with albumentations
# warning: this code used np.random other than torch.random (the same as albumentations);
#          should use tu.tools.worker_init_fn to avoid bug in seed inherit when using pytorch dataloader.
# seefun 2021.10
# uppdate 2021.11

import cv2
import math
import copy
import numpy as np
from albumentations import ImageOnlyTransform

__all__ = ['RandomBrightnessNoise', 'RandomBrightnessContrastPointwise', 'RandomSnowNoise', 'RandomEraseNoise']


def generate_perlin_noise_2d(shape, res):
    assert len(shape) == 2
    assert len(res) == 2
    if shape[0] % res[0] != 0:
        res = [1, res[1]]
    if shape[1] % res[1] != 0:
        res = [res[0], 1]

    def f(t):
        return 6 * t**5 - 15 * t**4 + 10 * t**3

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    # Gradients
    angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    g00 = gradients[0:-1, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1, 1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:, 1:].repeat(d[0], 0).repeat(d[1], 1)
    # Ramps
    n00 = np.sum(grid * g00, 2)
    n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)
    # Interpolation
    t = f(grid)
    n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
    n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
    return np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)


def random_perlin_brightness(image, noise_strength=0, max_delta=0.4):
    # noise_strength in [0,1,2,....]
    noise_strength = 2**noise_strength
    h = image.shape[0]
    w = image.shape[1]
    h_gen = math.ceil(h / noise_strength) * noise_strength
    w_gen = math.ceil(w / noise_strength) * noise_strength
    noise = generate_perlin_noise_2d(
        (h_gen, w_gen), (noise_strength, noise_strength))
    if image.dtype == np.uint8:
        noise = noise[:h, :w] * 255 * max_delta
        noise = noise[:, :, np.newaxis]
        image = image.astype(np.float32) + noise
        image = np.clip(image, 0, 255)
        return image.astype(np.uint8)
    else:
        noise = noise[:h, :w] * max_delta
        noise = noise[:, :, np.newaxis]
        image = image.astype(np.float32) + noise
        image = np.clip(image, 0, 1)
        return image.astype(np.float32)


def random_snow_white_noise(image, max_noise_ratio=0.02, noise_range=[0.5, 1.0], min_noise_size=1, max_noise_size=3):
    h = image.shape[0]
    w = image.shape[1]
    noise_ratio = max_noise_ratio * np.random.rand()
    min_noise_size = max(min_noise_size, 1)
    max_noise_size = max(min_noise_size + 1, max_noise_size)
    assert max_noise_size < min(h, w)
    noise_size = np.random.randint(min_noise_size, max_noise_size + 1)
    h_noise = h // noise_size
    w_noise = w // noise_size
    noise = np.random.rand(h_noise, w_noise)
    noise = (noise / noise_ratio * (noise_range[1] - noise_range[0]) +
             noise_range[0]) * (noise < noise_ratio)
    noise = cv2.resize(noise, (w, h))
    # noise = cv2.GaussianBlur(noise, (3, 3), 0)
    if image.dtype == np.uint8:
        noise = noise[:, :, np.newaxis] * 255
        image = image.astype(np.float32) + noise
        image = np.clip(image, 0, 255)
        return image.astype(np.uint8)
    else:
        noise = noise[:, :, np.newaxis]
        image = image.astype(np.float32) + noise
        image = np.clip(image, 0, 1)
        return image.astype(np.float32)


def point_wise_colorjitter(image, brightness=0.1, contrast=0.1, channel_wise=False, size=1):
    h = image.shape[0]
    w = image.shape[1]
    assert size < min(h, w)
    if channel_wise:
        c = image.shape[2]
    else:
        c = 1
    h_noise = h // size
    w_noise = w // size
    brightness_noise = np.random.rand(
        h_noise, w_noise, c) * brightness * 2 - brightness
    contrast_noise = np.random.rand(
        h_noise, w_noise, c) * contrast * 2 - contrast + 1
    brightness_noise = cv2.resize(brightness_noise, (w, h))
    contrast_noise = cv2.resize(contrast_noise, (w, h))
    if c == 1:
        brightness_noise = brightness_noise[:, :, np.newaxis]
        contrast_noise = contrast_noise[:, :, np.newaxis]
    if image.dtype == np.uint8:
        brightness_noise = brightness_noise * 255
        image = image.astype(np.float32) + brightness_noise
        image = image * contrast_noise
        image = np.clip(image, 0, 255)
        return image.astype(np.uint8)
    else:
        image = image.astype(np.float32) + brightness_noise
        image = image * contrast_noise
        image = np.clip(image, 0, 1)
        return image.astype(np.float32)


def random_erase_noise(image, max_noise_ratio=0.3, min_noise_size=0, max_noise_size=2):
    h = image.shape[0]
    w = image.shape[1]
    img_mean = np.mean(image[:2], axis=(0, 1)) / 4.0 \
        + np.mean(image[:, :2], axis=(0, 1)) / 4.0 \
        + np.mean(image[-2:], axis=(0, 1)) / 4.0 \
        + np.mean(image[:, -2:], axis=(0, 1)) / 4.0
    noise_ratio = max_noise_ratio * np.random.rand()
    min_noise_size = max(min_noise_size, 1)
    max_noise_size = max(min_noise_size + 1, max_noise_size)
    assert max_noise_size < min(h, w)
    noise_size = np.random.randint(min_noise_size, max_noise_size + 1)
    h_noise = h // noise_size
    w_noise = w // noise_size
    noise = np.random.rand(h_noise, w_noise)
    noise = np.ones(noise.shape) * (noise < noise_ratio)
    noise = cv2.resize(noise, (w, h)) > 0
    noise = noise[:, :, np.newaxis] * img_mean[np.newaxis, np.newaxis, :]
    noise = (np.random.rand(h, w, 3) * 0.2 - 0.1 + 1) * noise
    image_raw = copy.deepcopy(image)
    if image.dtype == np.uint8:
        image = image.astype(np.float32)
        image[noise > 0] = noise[noise > 0]
        image = image_raw / 2.0 + cv2.blur(image, (5, 5)) / 2.0
        image = np.clip(image, 0, 255)
        return image.astype(np.uint8)
    else:
        image = image.astype(np.float32)
        image[noise > 0] = noise[noise > 0]
        image = image_raw / 2.0 + cv2.blur(image, (5, 5)) / 2.0
        image = np.clip(image, 0, 1)
        return image.astype(np.float32)


class RandomBrightnessNoise(ImageOnlyTransform):
    """ Add random 2d perlin noise to the image
    Args:
        noise_strength (int): resolution of perlin noise from [0,1,2,3,...], 0,1,2 suggested, the bigger the stronger;
        max_delta  (float): max brightness deleta in [0,1], the bigger the stronger;
    Targets:
        image
    Image types:
        uint8, float32
    """

    def __init__(
        self,
        noise_strength=0,
        max_delta=0.4,
        always_apply=False,
        p=0.5,
    ):
        super(RandomBrightnessNoise, self).__init__(always_apply, p)
        self.noise_strength = noise_strength
        self.max_delta = max_delta

    def apply(self, image, **params):
        return random_perlin_brightness(image, self.noise_strength, self.max_delta)

    def get_transform_init_args_names(self):
        return ("noise_strength", "max_delta")


class RandomBrightnessContrastPointwise(ImageOnlyTransform):
    """ Random brightness and contrast noise
    Args:
        noise_strength (int): resolution of perlin noise from [0,1,2,3,...], 0,1,2 suggested, the bigger the stronger;
        max_delta  (float): max brightness deleta in [0,1], the bigger the stronger;
    Targets:
        image
    Image types:
        uint8, float32
    """

    def __init__(
        self,
        brightness=0.1,
        contrast=0.1,
        channel_wise=False,
        size=1,
        always_apply=False,
        p=0.5,
    ):
        super(RandomBrightnessContrastPointwise,
              self).__init__(always_apply, p)
        self.brightness = brightness
        self.contrast = contrast
        self.channel_wise = channel_wise
        self.size = size

    def apply(self, image, **params):
        return point_wise_colorjitter(image, self.brightness, self.contrast, self.channel_wise, self.size)

    def get_transform_init_args_names(self):
        return ("brightness", "contrast", "channel_wise", "size")


class RandomSnowNoise(ImageOnlyTransform):
    """ Add random 2d snow noise to the image
    Args:
        max_noise_ratio (float): max noise ratio;
        noise_range  (list): range of noise value;
        min_noise_size (int): min pixel size of snow white point (noise_size*noise_size);
        max_noise_size (int): max pixel size of snow white point (noise_size*noise_size).
    Targets:
        image
    Image types:
        uint8, float32
    """

    def __init__(
        self,
        max_noise_ratio=0.02,
        noise_range=[0.5, 1.0],
        min_noise_size=1,
        max_noise_size=3,
        always_apply=False,
        p=0.5,
    ):
        super(RandomSnowNoise, self).__init__(always_apply, p)
        self.max_noise_ratio = max_noise_ratio
        self.noise_range = noise_range
        self.min_noise_size = min_noise_size
        self.max_noise_size = max_noise_size

    def apply(self, image, **params):
        return random_snow_white_noise(image, self.max_noise_ratio, self.noise_range,
                                       self.min_noise_size, self.max_noise_size)

    def get_transform_init_args_names(self):
        return ("max_noise_ratio", "noise_range", "min_noise_size", "max_noise_size")


class RandomEraseNoise(ImageOnlyTransform):
    """ Add random noise by image boundary pixel value
    Args:
        max_noise_ratio (float): max noise ratio;
        min_noise_size  (list): min pixel size of noise point (noise_size*noise_size);
        max_noise_size (int): max pixel size of noise point (noise_size*noise_size).
    Targets:
        image
    Image types:
        uint8, float32
    """

    def __init__(
        self,
        max_noise_ratio=0.3,
        min_noise_size=0,
        max_noise_size=2,
        always_apply=False,
        p=0.5,
    ):
        super(RandomEraseNoise, self).__init__(always_apply, p)
        self.max_noise_ratio = max_noise_ratio
        self.min_noise_size = min_noise_size
        self.max_noise_size = max_noise_size

    def apply(self, image, **params):
        return random_erase_noise(image, self.max_noise_ratio, self.min_noise_size, self.max_noise_size)

    def get_transform_init_args_names(self):
        return ("max_noise_ratio", "min_noise_size", "max_noise_size")
