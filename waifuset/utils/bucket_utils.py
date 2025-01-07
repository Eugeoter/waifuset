import math
from typing import List, Optional, Tuple, Union


def around_reso(img_w, img_h, reso: Union[Tuple[int, int], int], divisible: Optional[int] = None, max_width=None, max_height=None) -> Tuple[int, int]:
    r"""
    w*h = reso*reso
    w/h = img_w/img_h
    => w = img_ar*h
    => img_ar*h^2 = reso
    => h = sqrt(reso / img_ar)
    """
    reso = reso if isinstance(reso, tuple) else (reso, reso)
    divisible = divisible or 1

    # check if already in the bucket
    max_area = reso[0] * reso[1]
    if img_w <= max_width and img_h <= max_height and img_w % divisible == 0 and img_h % divisible == 0 and img_w * img_h <= max_area:
        return (img_w, img_h)

    img_ar = img_w / img_h
    around_h = math.sqrt(max_area / img_ar)
    around_w = img_ar * around_h // divisible * divisible
    if max_width and around_w > max_width:
        around_h = around_h * max_width // around_w
        around_w = max_width
    elif max_height and around_h > max_height:
        around_w = around_w * max_height // around_h
        around_h = max_height
    around_h = min(around_h, max_height) if max_height else around_h
    around_w = min(around_w, max_width) if max_width else around_w
    around_h = int(around_h // divisible * divisible)
    around_w = int(around_w // divisible * divisible)
    return (around_w, around_h)


def closest_resolution(buckets: List[Tuple[int, int]], size: Tuple[int, int]) -> Tuple[int, int]:
    img_ar = size[0] / size[1]

    def distance(reso: Tuple[int, int]) -> float:
        return abs(img_ar - reso[0]/reso[1])

    return min(buckets, key=distance)


def get_bucket_size(
    image_size: Tuple[int, int],
    max_resolution: Optional[Union[Tuple[int, int], int]] = 1024,
    max_width: Optional[int] = None,
    max_height: Optional[int] = None,
    divisible: Optional[int] = 32,
    buckets: Optional[List[Tuple[int, int]]] = None,
    allow_upscale: bool = False,
):
    r"""
    Get the closest resolution to the image's resolution from the buckets. If the image's aspect ratio is too
    different from the closest resolution, then return the around resolution based on the max resolution.
    :param image: The image to be resized.
    :param buckets: The buckets of resolutions to choose from. Default to SDXL_BUCKETS. Set None to use max_resolution.
    :param max_resolution: The max resolution to be used to calculate the around resolution. It's used to calculate the 
        around resolution when `buckets` is None or no bucket can contain that image without exceeding the max aspect ratio.
        Default to 1024. Set `-1` to auto calculate from the buckets' max resolution. Set None to disable.
        Set None to auto calculate from the buckets' max resolution.
    :param max_aspect_ratio: The max aspect ratio difference between the image and the closest resolution. Default to 1.1.
        Set None to disable.
    :param divisible: The divisible number of bucket resolutions. Default to 32.
    :return: The closest resolution to the image's resolution.
    """
    if not buckets and (not max_resolution or max_resolution == -1):
        raise ValueError(
            "Either `buckets` or `max_resolution` must be provided.")

    img_w, img_h = image_size
    bucket_w, bucket_h = closest_resolution(buckets, image_size) if buckets else around_reso(img_w, img_h, reso=max_resolution, divisible=divisible, max_width=max_width, max_height=max_height)
    if not allow_upscale and (img_w < bucket_w or img_h < bucket_h):
        bucket_w = img_w // divisible * divisible
        bucket_h = img_h // divisible * divisible
    bucket_w = max(bucket_w, divisible)
    bucket_h = max(bucket_h, divisible)
    bucket_size = (bucket_w, bucket_h)
    return bucket_size
