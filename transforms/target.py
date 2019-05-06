import numpy as np

from PIL import Image
from skimage.morphology import opening, closing, convex_hull_image
from typing import Callable


def transform(transform_fn: Callable, image: Image, mult: int = 1, **kwargs):
    img = np.array(image)
    img = mult * transform_fn(img, **kwargs).astype(np.uint8)
    img = Image.fromarray(img)

    return img.point(lambda p: p > 255 // 2 and 255)


class Opening:
    def __init__(self, selem_fn: Callable, factor: int):
        self.selem = selem_fn(factor)

    def __call__(self, img: Image):
        return transform(opening, img, selem=self.selem)


class Closing:
    def __init__(self, selem_fn: Callable, factor: int):
        self.selem = selem_fn(factor)

    def __call__(self, img: Image):
        return transform(closing, img, selem=self.selem)


class ConvexHull:
    def __call__(self, img: Image):
        return transform(convex_hull_image, img, mult=255)

