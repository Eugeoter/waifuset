import cv2
import numpy as np
from PIL import Image


def load_image(
    image_path,
    mode=None,
):
    image = Image.open(image_path)
    if mode:
        image = image.convert(mode)
    image = np.array(image)
    return image


def fill_transparency(image: np.ndarray, bg_color=(255, 255, 255)):
    r"""
    Fill the transparent part of an image with a background color.
    Please pay attention that this function doesn't change the image type.
    """
    num_channels = image.shape[2]
    assert num_channels == 2 or num_channels == 4, f"Image must have 2 or 4 channels, but got {image.shape[2]} channels."
    if len(bg_color) > num_channels - 1:
        bg_color = bg_color[:num_channels - 1]
    elif len(bg_color) < num_channels - 1:
        bg_color = bg_color + (255,) * (num_channels - 1 - len(bg_color))
    bg = np.full_like(image, bg_color + (255,))
    bg[:, :, :num_channels-1] = image[:, :, :num_channels-1]
    return bg


def cvt2rgb(image: np.ndarray):
    r"""
    Convert an image to RGB.
    """
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 3:
        return image
    elif image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif image.shape[2] == 2:
        gray_channel, alpha_channel = cv2.split(image)
        return cv2.merge((gray_channel, gray_channel, gray_channel))
    else:
        raise ValueError(f"Invalid image shape {image.shape}.")
