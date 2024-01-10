import gradio as gr
import random
from tqdm import tqdm
from pathlib import Path
from typing import Callable, Any, Tuple, Dict
from . import custom_components as cc
from ..import tagging
from .emoji import Emoji
from ..utils import log_utils as logu


OPS = {
    'add': lambda x, y: x | y,
    'remove': lambda x, y: x - y,
    'replace': lambda x, y: x.replace(y),
}

CONDITION = {
    'any': any,
    'all': all,
}

INCLUSION_RELATIONSHIP = {
    'include': lambda x, y: y in x,
    'exclude': lambda x, y: y not in x,
}


def create_ui(
    source,
    write_to_database=False,
    write_to_txt=False,
    database_path=None,
):
    from .ui_dataset import UIDataset
    from ..classes import ImageInfo, Caption
    dataset = UIDataset(source, write_to_database=write_to_database, write_to_txt=write_to_txt, database_path=database_path,
                        read_caption=True, formalize_caption=False, verbose=True)
    database_path = Path(database_path)
    subsets = dataset.subsets

    # ========================================= UI ========================================= #

    with gr.Blocks() as demo:
        with gr.Tab(label='Dataset'):
            with gr.Row():
                with gr.Column():
                    subsets_selector = gr.Dropdown(
                        label='Category',
                        choices=[""] + sorted(subsets.keys()),
                        value="",
                        multiselect=False,
                        allow_custom_value=False,
                    )
                with gr.Column():
                    log_box = gr.TextArea(
                        label='Log',
                        lines=1,
                        max_lines=1,
                    )

            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        showcase = gr.Gallery(
                            label='Showcase',
                            rows=4,
                            columns=4,
                            container=True,
                            object_fit='scale-down',
                            height=512,
                            selected_index=0,
                        )
                        image_key = gr.Textbox(value=None, visible=False, label='Image Key')
                    with gr.Row():
                        reload_subset_btn = cc.EmojiButton(Emoji.anticlockwise, variant='primary')
                        remove_image_btn = cc.EmojiButton(Emoji.trash_bin, variant='stop')
                with gr.Column():
                    with gr.Row():
                        caption = gr.Textbox(
                            label='Caption',
                            value=None,
                            container=True,
                            show_copy_button=True,
                            lines=6,
                            max_lines=6,
                        )
                    with gr.Row():
                        random_roll_btn = cc.EmojiButton(Emoji.dice, variant='primary')
                        save_caption_edition_btn = cc.EmojiButton(Emoji.label, variant='primary')
                        save_to_disk_btn = cc.EmojiButton(Emoji.floppy_disk, variant='primary')
                        batch_proc = gr.Checkbox(
                            label='Batch',
                            value=False,
                            container=False,
                            scale=0,
                            min_width=144
                        )

            with gr.Row():
                with gr.Column(scale=3):
                    with gr.Row():
                        image_path = gr.Textbox(
                            label='Image Path',
                            value=None,
                            container=True,
                            max_lines=2,
                            lines=1,
                            show_copy_button=True,
                            interactive=False,
                        )
                with gr.Column(scale=5):
                    with gr.Tab(label='Quick Tagging'):
                        add_tag_btns = []
                        tag_selectors = []
                        remove_tag_btns = []
                        for r in range(3):
                            with gr.Row():
                                for c in range(2):
                                    add_tag_btns.append(cc.EmojiButton(Emoji.plus, variant='primary'))
                                    tag_selectors.append(gr.Dropdown(
                                        choices=list(tagging.CUSTOM_TAGS),
                                        value=None,
                                        multiselect=True,
                                        allow_custom_value=True,
                                        show_label=False,
                                        container=False
                                    ))
                                    remove_tag_btns.append(cc.EmojiButton(Emoji.minus, variant='stop'))
                    with gr.Tab(label='Caption Operation'):
                        with gr.Row():
                            op_dropdown = gr.Dropdown(
                                label='Op',
                                choices=[
                                    'add',
                                    'remove',
                                    'replace',
                                ],
                                value='add',
                                multiselect=False,
                                allow_custom_value=False,
                                scale=0,
                                min_width=72,
                            )
                            op_tag_dropdown = gr.Dropdown(
                                label='Tags',
                                choices=[],
                                value=None,
                                allow_custom_value=True,
                                multiselect=True,
                            )

                        with gr.Row():
                            condition_dropdown = gr.Dropdown(
                                label='Condition',
                                choices=[
                                    'any',
                                    'all',
                                ],
                                value='any',
                                multiselect=False,
                                allow_custom_value=False,
                                scale=0,
                                min_width=64,
                            )

                            cond_tag_dropdown = gr.Dropdown(
                                label='Tags',
                                choices=[],
                                value=None,
                                allow_custom_value=True,
                                multiselect=True,
                            )

                            inclusion_relationship_dropdown = gr.Dropdown(
                                label='Inclusion',
                                choices=[
                                    'include',
                                    'exclude',
                                ],
                                value='include',
                                multiselect=False,
                                allow_custom_value=False,
                                scale=0,
                                min_width=64,
                            )

        # ========================================= functions ========================================= #

        def track_image_key(image_key):
            if image_key is None or image_key == '':
                return None, None
            image_key = Path(image_key).stem
            image_info = dataset[image_key]
            image_path = str(image_info.image_path) if image_info.image_path.is_file() else None
            caption = str(image_info.caption) if image_info.caption is not None else None
            return image_path, caption

        image_key.change(
            fn=track_image_key,
            inputs=[image_key],
            outputs=[image_path, caption],
        )

        def show_subset(subset_key):
            if subset_key is None or subset_key == '':
                return None, None
            subset = subsets[subset_key]
            pre_idx = dataset.selected.index
            # print(f"pre_idx: {pre_idx} | len(subset): {len(subset)}")
            if pre_idx is not None and pre_idx < len(subset):
                new_img_key = subset.keys()[pre_idx]
            else:
                new_img_key = None
            # print(f"new_img_key: {new_img_key}")
            return gr.update(value=[(v.image_path, k) for k, v in subset.items()]), new_img_key

        subsets_selector.change(
            fn=show_subset,
            inputs=[subsets_selector],
            outputs=[showcase, image_key],
            show_progress=True,
        )

        reload_subset_btn.click(
            fn=show_subset,
            inputs=[subsets_selector],
            outputs=[showcase, image_key],
            show_progress=True,
        )

        def show_image_info(selected: gr.SelectData):
            if selected is None:
                return None, None
            image_key = dataset.select(selected)
            return image_key

        showcase.select(
            fn=show_image_info,
            outputs=[image_key],
        )

        def remove_image(image_key):
            if image_key is None:
                return f"image key is None."
            del dataset[image_key]
            return f"removed: {image_key}"

        remove_image_btn.click(
            fn=remove_image,
            inputs=[image_path],
            outputs=[log_box],
        )

        def random_sample(subset_key):
            subset = subsets[subset_key] if subset_key is not None and subset_key != "" else dataset
            image_key = random.choice(list(subset.keys()))
            image_info = dataset[image_key]
            dataset.select(image_key)
            return [(image_info.image_path, image_key)], image_key

        random_roll_btn.click(
            fn=random_sample,
            inputs=[subsets_selector],
            outputs=[showcase, image_key],
        )

        def edit_caption(func: Callable[[ImageInfo, Tuple[Any, ...], Dict[str, Any]], Caption]) -> str:
            def wrapper(image_key, batch, *args, progress: gr.Progress = gr.Progress(track_tqdm=True), **kwargs):
                if image_key is None or image_key == '':
                    return gr.update(), f"image key is None."
                image_key = Path(image_key).stem
                image_info = dataset[image_key]
                if batch:
                    category = image_info.category
                    subset = subsets[category]
                    for img_key, img_info in tqdm(subset.items()):
                        new_caption = func(img_info, *args, **kwargs)
                        img_info.caption = new_caption
                        dataset[img_key] = img_info  # add history
                else:
                    new_caption = func(image_info, *args, **kwargs)
                    image_info.caption = new_caption
                    dataset[image_key] = image_info  # add history
                return str(new_caption), f"{func.__name__.replace('_', ' ')}: {image_key}"
            return wrapper

        def write_caption(image_info, caption):
            return Caption(caption) if caption is not None and caption.strip() != '' else None

        save_caption_edition_btn.click(
            fn=edit_caption(write_caption),
            inputs=[image_path, gr.State(False), caption],
            outputs=[caption, log_box],
        )

        def save_to_disk(progress: gr.Progress = gr.Progress(track_tqdm=True)):
            dataset.save(progress=progress)
            return f"database saved: {database_path.absolute().as_posix()}"

        save_to_disk_btn.click(
            fn=save_to_disk,
            inputs=[],
            outputs=[log_box],
        )

        def add_tags(image_info, tags):
            caption = tags | image_info.caption
            return caption

        def remove_tags(image_info, tags):
            caption = image_info.caption - tags
            return caption

        for add_tag_btn, tag_selector, remove_tag_btn in zip(add_tag_btns, tag_selectors, remove_tag_btns):
            add_tag_btn.click(
                fn=edit_caption(add_tags),
                inputs=[image_path, batch_proc, tag_selector],
                outputs=[caption, log_box],
            )
            remove_tag_btn.click(
                fn=edit_caption(remove_tags),
                inputs=[image_path, batch_proc, tag_selector],
                outputs=[caption, log_box],
            )

        # ========================================= Caption Operation ========================================= #

        def caption_operation(image_info, op, op_tags, condition, cond_tags, inclusion_relationship):
            caption = image_info.caption
            if op_tags is None or len(op_tags) == 0:
                return caption
            op_func = OPS[op]
            op_caption = Caption(op_tags)
            do_condition = condition is not None and condition != ''
            if do_condition:
                cond_func = CONDITION[condition]
                cond_tags = set(cond_tags)
                inclusion_relationship_func = INCLUSION_RELATIONSHIP[inclusion_relationship]

                if not cond_func(inclusion_relationship_func(caption, cond_tag) for cond_tag in cond_tags):
                    return caption

            caption = op_func(caption, op_caption)
            return caption

    return demo
