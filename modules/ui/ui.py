import gradio as gr
from tqdm import tqdm
from pathlib import Path
from typing import Callable, Any, Tuple, Dict
from . import custom_components as cc
from ..import tagging
from .emoji import Emoji
from ..utils import log_utils as logu


def create_ui(
    source,
    write_to_database=False,
    write_to_txt=False,
    database_path=None,
):
    from .ui_dataset import UIDataset
    from ..classes import Dataset, ImageInfo, Caption
    dataset = UIDataset(source, read_caption=True, formalize_caption=False, verbose=True)
    database_path = Path(database_path)
    subsets = {}
    for k, v in tqdm(dataset.items(), desc='making subsets'):
        if v.category not in subsets:
            subsets[v.category] = Dataset()
        subsets[v.category][k] = v

    # ========================================= UI ========================================= #

    with gr.Blocks() as demo:
        with gr.Tab(label='Dataset'):
            with gr.Row():
                with gr.Column():
                    subsets_selector = gr.Dropdown(
                        label='Category',
                        choices=sorted(subsets.keys()),
                        value=None,
                        multiselect=False,
                        allow_custom_value=False,
                        min_width=384,
                        scale=0,
                    )
                with gr.Column():
                    log_box = gr.TextArea(
                        lines=1,
                        max_lines=1,
                        container=False,
                        show_label=False,
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
                        )
                        image_key = gr.State(value=None)
                    with gr.Row():
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
                        save_caption_edition_btn = cc.EmojiButton(Emoji.label, variant='secondary')
                        save_to_disk_btn = cc.EmojiButton(Emoji.floppy_disk, variant='secondary')

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

        # ========================================= functions ========================================= #

        def show_subset(subset_key):
            if subset_key is None:
                return None
            subset = subsets[subset_key]
            return [(v.image_path, k) for k, v in subset.items()]

        subsets_selector.change(
            fn=show_subset,
            inputs=[subsets_selector],
            outputs=[showcase],
        )

        def show_image_info(selected: gr.SelectData):
            if selected is None:
                return None, None
            image_info = dataset.select(selected)
            image_path = str(image_info.image_path)
            caption = str(image_info.caption) if image_info.caption else None
            return image_key, image_path, caption

        showcase.select(
            fn=show_image_info,
            outputs=[image_key, image_path, caption],
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

        def edit_caption(func: Callable[[ImageInfo, Tuple[Any, ...], Dict[str, Any]], Caption]) -> str:
            def wrapper(image_key, *args, **kwargs):
                image_key = Path(image_key).stem
                image_info = dataset[image_key]
                new_caption = func(image_info, *args, **kwargs)
                image_info.caption = new_caption
                dataset[image_key] = image_info  # add history
                return str(new_caption), f"{func.__name__.replace('_', ' ')}: {image_key}"
            return wrapper

        def write_caption(image_info, caption):
            return Caption(caption) if caption is not None and caption.strip() != '' else None

        save_caption_edition_btn.click(
            fn=edit_caption(write_caption),
            inputs=[image_path, caption],
            outputs=[caption, log_box],
        )

        def save_to_disk(progress: gr.Progress = gr.Progress(track_tqdm=True)):
            if not database_path.is_file():
                database_path.parent.mkdir(parents=True, exist_ok=True)
                mode = 'w'
            else:
                mode = 'a'
            dataset.save(database_path, mode=mode, sort_keys=False, progress=progress)
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
                inputs=[image_path, tag_selector],
                outputs=[caption, log_box],
            )
            remove_tag_btn.click(
                fn=edit_caption(remove_tags),
                inputs=[image_path, tag_selector],
                outputs=[caption, log_box],
            )

    return demo
