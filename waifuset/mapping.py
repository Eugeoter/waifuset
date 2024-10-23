import re
import os
from pathlib import Path
from . import tagging
from .classes.data.caption.caption import Caption
from .classes.data import data_utils


def old2new(img_md):
    img_path = img_md['image_path']
    img_key = Path(img_path).stem
    category = Path(img_path).parent.name
    source = Path(img_path).parent.parent.name

    pattern1 = re.compile(rf"(?P<tagtypeA>(?:{'|'.join(tagging.ALL_TAG_TYPES)})+):(?P<tagtypeB>(?:{'|'.join(tagging.ALL_TAG_TYPES)})+):(?P<tagname>.+)")  # replace `artist:artist:` to `artist:`
    pattern2 = re.compile(r"((?:quality:\s?)?.*) quality")  # remove postfix `quality`
    pattern3 = re.compile(r"rating:\s?(general|sensitive|questionable|explicit)")  # replace `rating` to `safety`

    caption = Caption(img_md['caption'])
    if safe_level := img_md.get('safe_level'):
        caption += f"safety: {tagging.RATING_TO_SAFETY[safe_level]}"
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
        **{
            k: caption.sep.join(v) if v else None for k, v in caption.metadata.items()
        },
    }
    return new_img_md


def redirect_columns(img_md, columns, tarset):
    for col in columns:
        if (img_key := img_md['image_key']) in tarset and col in (tar_md := tarset[img_key]):
            img_md[col] = tar_md[col]
    return img_md


def as_posix_path(img_md, columns):
    for col in columns:
        if col in img_md and '\\' in (fp := img_md[col]):
            img_md[col] = fp.as_posix() if isinstance(fp, Path) else fp.replace('\\', '/')
    return img_md


def attr_reader(img_md):
    img_path = img_md['image_path']
    attr_dict = data_utils.read_attrs(img_path, types=['danbooru', 'txt'])
    return attr_dict


def caption_processor(img_md):
    caption = img_md.get('caption')
    if caption is None:
        return None
    caption = Caption(caption)
    if (cat := os.path.basename(os.path.dirname(img_md['image_path']))).startswith('by '):
        caption += f"artist:{cat[3:]}"
    caption.parse()
    caption.alias()
    caption.deimplicate()
    caption.deduplicate()
    caption.sort()
    metadata = {k: ', '.join(v) for k, v in caption.metadata.items() if v}
    return {'caption': caption.text, **metadata}
