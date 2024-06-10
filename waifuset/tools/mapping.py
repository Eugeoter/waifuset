import re
from pathlib import Path
from ..classes.data.caption import tagging
from ..classes.data.caption.caption import Caption


def old2new(img_md):
    img_path = img_md['image_path']
    img_key = Path(img_path).stem
    category = Path(img_path).parent.name
    source = Path(img_path).parent.parent.name

    pattern1 = re.compile(rf"(?P<tagtypeA>(?:{'|'.join(tagging.TAG_TYPES)})+):(?P<tagtypeB>(?:{'|'.join(tagging.TAG_TYPES)})+):(?P<tagname>.+)")  # replace `artist:artist:` to `artist:`
    pattern2 = re.compile(r"((?:quality:\s?)?.*) quality")  # remove postfix `quality`
    pattern3 = re.compile(r"rating:\s?(general|sensitive|questionable|explicit)")  # replace `rating` to `safety`

    caption = Caption(img_md['caption'])
    if safe_level := img_md.get('safe_level'):
        caption += f"safety: {tagging.SAFE_LEVEL2TAG[safe_level]}"
    caption[pattern1] = r"\g<tagtypeA>:\g<tagname>"
    caption[pattern2] = r"\1"
    caption[pattern3] = r'safety: \1'
    caption.parse()

    new_img_md = {
        'image_key': img_key,
        'image_path': img_md['image_path'],
        'caption': caption.text,
        'description': img_md.get('description', None),
        'category': category,
        'source': source,
        'date': img_md.get('date'),
        'original_size': f"{img_md['original_size'][0]}x{img_md['original_size'][1]}" if img_md.get('original_size') else None,
        'aesthetic_score': img_md['aesthetic_score'],
        'perceptual_hash': img_md['perceptual_hash'],
        'safe_rating': img_md.get('safe_rating'),
        **{
            k: caption.sep.join(v) if v else None for k, v in caption.metadata.items()
        },
    }
    return new_img_md


def patch_image_path_info(img_md):
    img_path = img_md['image_path']
    if not img_path:
        return None
    img_path = Path(img_path)
    img_md['image_key'] = img_path.stem
    img_md['category'] = img_path.parent.name
    img_md['source'] = img_path.parent.parent.name
    return img_md


def patch_dirset(img_md):
    img_md = patch_image_path_info(img_md)
    img_md['caption'] = None
    return img_md


def patch_columns(img_md, columns):
    for col in columns:
        img_md.setdefault(col, None)
    return img_md


def redirect_image_path(img_md, tarset):
    img_key = img_md['image_key']
    if img_key not in tarset:
        return None
    img_md['image_path'] = tarset[img_key]['image_path']
    img_md = patch_image_path_info(img_md)
    return img_md


def redirect_columns(img_md, columns, tarset):
    for col in columns:
        if (img_key := img_md['image_key']) in tarset and col in (tar_md := tarset[img_key]):
            img_md[col] = tar_md[col]
    return img_md


def as_posix_path(img_md, columns):
    for col in columns:
        img_md[col] = Path(img_md[col]).as_posix()
    return img_md
