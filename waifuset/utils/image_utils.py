import cv2
import numpy as np
import json
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


def cvt2gray(image: np.ndarray):
    r"""
    Convert an image to gray.
    """
    if len(image.shape) == 2:
        return image
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    elif image.shape[2] == 2:
        gray_channel, alpha_channel = cv2.split(image)
        return gray_channel
    else:
        raise ValueError(f"Invalid image shape {image.shape}.")


def parse_gen_info(metadata):
    gen_info = {}
    try:
        if 'parameters' in metadata:  # webui style
            params: str = metadata['parameters']
            if 'Negative prompt: ' in params:
                positive_prompt, params = params.split('Negative prompt: ', 1)
                negative_prompt, params = params.split('Steps: ', 1)
            else:
                positive_prompt, params = params.split('Steps: ', 1)
                negative_prompt = ''
            params = 'Steps: ' + params
            params = params.split(', ')
            params = [param.split(': ') for param in params]
            params = {param[0]: param[1] for param in params}

            gen_info['Positive prompt'] = positive_prompt
            gen_info['Negative prompt'] = negative_prompt
            gen_info.update(params)
        elif 'Title' in metadata:  # nai style
            gen_info.update(metadata)
            params = gen_info['Comment']  # str
            params = json.loads(params)  # dict
            del gen_info['Comment']
            params = {k.capitalize(): v for k, v in params.items()}
            params['Positive prompt'] = params.pop('Prompt')
            params['Negative prompt'] = params.pop('Uc')
            gen_info.update(params)
        else:
            if len(metadata) != 0:
                # print(f"unknown metadata: {metadata}")
                ...
    except ValueError:
        print(f"unknown metadata: {metadata}")
        raise
    return gen_info
