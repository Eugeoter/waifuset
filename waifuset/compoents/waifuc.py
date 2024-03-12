import os
import time
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from typing import Literal
from ..utils import log_utils


class Waifuc:
    def __init__(self):
        pass

    def __call__(
        self,
        tags,
        output_root,
        tag_type: Literal['default', 'character', 'artist'] = 'default',
        max_per_tag=1000,
        image_mode='RGB',
        force_background: str = 'white',
        filter_classes=['illustration', 'bangumi'],
        filter_similar=True,
        skip_when_image_exist=True,
        skip_when_dir_exist=False,
    ):
        from waifuc.export import SaveExporter
        from waifuc.source import DanbooruSource
        from waifuc import action

        output_root = Path(output_root).absolute()
        if isinstance(tags, str):
            tags = [tags]
        tags = [tag.lower() for tag in tags]
        pbar = tqdm(tags, desc="downloading")
        logs = dict(
            tag='',
        )
        for tag in tags:
            category = f"by {tag}" if tag_type == 'artist' else tag
            category = category.replace('_', ' ').replace(':', ' ').replace('/', '').replace('\\', '')
            output_dir = output_root / category
            if skip_when_dir_exist and output_dir.exists():
                pbar.update(1)
                continue
            output_dir.mkdir(parents=True, exist_ok=True)
            already_exist = len([f for f in os.listdir(output_dir) if f.startswith('.') and f.endswith('.json')])
            max_this_tag = max_per_tag - already_exist
            if max_this_tag <= 0:
                print(log_utils.blue(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"), log_utils.yellow("skip {tag} because already exist {already_exist} images"))
                pbar.update(1)
                continue
            else:
                print(log_utils.blue(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"), log_utils.yellow(f"downloading {max_this_tag} images for {tag}"))

            logs['tag'] = tag
            pbar.set_postfix(logs)

            actions = [
                action.ModeConvertAction(image_mode, force_background),
                # 图像预过滤
                action.NoMonochromeAction(),  # 丢弃单色、灰度或素描等单色图像
                action.ClassFilterAction(filter_classes),
                action.FirstNSelectAction(max_this_tag),
            ]

            if filter_similar:
                actions.append(action.FilterSimilarAction('all'))

            source = DanbooruSource([tag])
            source.attach(
                *actions
            ).export(
                # save images (with meta information from danbooru site)
                SaveExporter(output_dir, skip_when_image_exist=skip_when_image_exist)
            )
            pbar.update(1)
        pbar.close()
