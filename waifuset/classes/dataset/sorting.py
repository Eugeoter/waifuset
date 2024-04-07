from .. import ImageInfo


def key(image_info: ImageInfo):
    return image_info.key


def stem(image_info: ImageInfo):
    return image_info.stem


def extension(image_info: ImageInfo):
    return image_info.image_path.suffix


def category(image_info: ImageInfo):
    return image_info.category


def perceptual_hash(image_info: ImageInfo, target=None):
    if image_info.perceptual_hash is None or target is None:
        return float('inf')
    import imagehash
    if isinstance(target, str):
        target = imagehash.hex_to_hash(target)
    return imagehash.hex_to_hash(image_info.perceptual_hash) - target


def aesthetic_score(image_info: ImageInfo):
    if image_info.aesthetic_score is None:
        return -float('inf')
    return image_info.aesthetic_score


def original_size(image_info: ImageInfo):
    if image_info.original_size is None:
        return -float('inf')
    width, height = image_info.original_size
    return width * height


def original_width(image_info: ImageInfo):
    if image_info.original_size is None:
        return -float('inf')
    width, height = image_info.original_size
    return width


def original_height(image_info: ImageInfo):
    if image_info.original_size is None:
        return -float('inf')
    width, height = image_info.original_size
    return height


def original_aspect_ratio(image_info: ImageInfo):
    if image_info.original_size is None:
        return -float('inf')
    width, height = image_info.original_size
    return width / height


def caption_length(image_info: ImageInfo):
    if image_info.caption is None:
        return -float('inf')
    return len(image_info.caption)


def has_gen_info(image_info: ImageInfo):
    return len(image_info.gen_info) > 0


def quality(image_info: ImageInfo):
    if image_info.caption is None:
        quality_ = 'normal'
    else:
        quality_ = image_info.caption.quality or 'normal'
    return {
        'horrible': 0,
        'worst': 2,
        'low': 3.5,
        'normal': 5,
        'high': 6.5,
        'best': 8,
        'amazing': 10,
    }.get(quality_, 5)


def quality_or_score(image_info: ImageInfo):
    if image_info.aesthetic_score is not None:
        return aesthetic_score(image_info)
    else:
        return quality(image_info)


def random(image_info: ImageInfo):
    import random
    return random.random()


def safe_rating(image_info: ImageInfo):
    return image_info.safe_rating


LEVEL2KEY = {
    'g': 0,
    's': 1,
    'q': 2,
    'e': 3,
}


def safe_level(image_info: ImageInfo):
    lvl = image_info.safe_level
    return LEVEL2KEY.get(lvl, len(LEVEL2KEY))
