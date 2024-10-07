import math
import random
from typing import List, Dict, Optional, Tuple, Union
from ....utils import image_utils

SDXL_BUCKET_SIZES = [
    (512, 1856), (512, 1920), (512, 1984), (512, 2048),
    (576, 1664), (576, 1728), (576, 1792), (640, 1536),
    (640, 1600), (704, 1344), (704, 1408), (704, 1472),
    (768, 1280), (768, 1344), (832, 1152), (832, 1216),
    (896, 1088), (896, 1152), (960, 1024), (960, 1088),
    (1024, 960), (1024, 1024), (1088, 896), (1088, 960),
    (1152, 832), (1152, 896), (1216, 832), (1280, 768),
    (1344, 704), (1344, 768), (1408, 704), (1472, 704),
    (1536, 640), (1600, 640), (1664, 576), (1728, 576),
    (1792, 576), (1856, 512), (1920, 512), (1984, 512), (2048, 512)
]


class AspectRatioBucketMixin(object):
    resolution: Union[Tuple[int, int], int] = 1024
    arb: bool = True
    max_aspect_ratio: float = 1.1
    bucket_reso_step: int = 32
    max_width: Optional[int] = None
    max_height: Optional[int] = None
    max_area: Optional[int] = None
    predefined_buckets: Optional[List[Tuple[int, int]]] = []

    def get_resolution(self):
        return self.resolution if isinstance(self.resolution, tuple) else (self.resolution, self.resolution)

    def get_bucket_size(self, img_md, image_size=None) -> Tuple[int, int]:
        if not self.arb:
            return self.get_resolution()
        elif (bucket_size := img_md.get('bucket_size')):
            return bucket_size
        elif image_size is not None or (image_size := img_md.get('image_size')) is not None or (image_size := image_utils.get_image_size(img_md.get('image_path'))) is not None:
            image_size = image_utils.convert_size_if_needed(image_size)
            bucket_size = get_bucket_size(
                image_size,
                self.resolution,
                divisible=self.bucket_reso_step,
                buckets=self.predefined_buckets,
                max_width=self.max_width,
                max_height=self.max_height,
            )
            return bucket_size
        else:
            raise ValueError("Either `bucket_size` or `image_size` must be provided in metadata.")

    def make_buckets(self) -> Dict[Tuple[int, int], List[str]]:
        if not self.arb:
            return {self.get_resolution(): list(self.dataset.keys())}
        bucket_keys = {}
        for img_key, img_md in self.logger.tqdm(self.dataset.items(), desc="make buckets"):
            if (bucket_size := img_md.get('bucket_size')) is not None:
                pass
            else:
                bucket_size = self.get_bucket_size(img_md)
            bucket_keys.setdefault(bucket_size, []).extend([img_key] * img_md.get('weight', 1))
        bucket_keys = self.shuffle_buckets(bucket_keys)
        return bucket_keys

    def shuffle_buckets(self, buckets: Dict[Tuple[int, int], List[str]]):
        bucket_sizes = list(buckets.keys())
        random.shuffle(bucket_sizes)
        buckets = {k: buckets[k] for k in bucket_sizes}
        for bucket in buckets.values():
            random.shuffle(bucket)
        return buckets


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
    img_ar = img_w / img_h
    around_h = math.sqrt(reso[0]*reso[1] / img_ar)
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
