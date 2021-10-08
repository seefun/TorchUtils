# compatible with albumentations
import math
import numpy as np
from albumentations import ImageOnlyTransform


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
    noise = generate_perlin_noise_2d((h_gen, w_gen), (noise_strength, noise_strength))
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
        p=1.0,
    ):
        super(RandomBrightnessNoise, self).__init__(always_apply, p)
        self.noise_strength = noise_strength
        self.max_delta = max_delta

    def apply(self, image, **params):
        return random_perlin_brightness(image, self.noise_strength, self.max_delta)

    def get_transform_init_args_names(self):
        return ("noise_strength", "max_delta")
