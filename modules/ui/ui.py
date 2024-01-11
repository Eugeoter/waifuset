import gradio as gr
import random
import pandas
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from typing import Callable, Any, Tuple, Dict, List, Union
from . import custom_components as cc
from ..import tagging
from .emoji import Emoji
from ..utils import log_utils as logu


OPS = {
    'add': lambda x, y: y | x,
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


def prepare_dataset(
    args,
):
    from .ui_dataset import UIDataset

    dataset = UIDataset(
        args.source,
        formalize_caption=args.formalize_caption,
        write_to_database=args.write_to_database,
        write_to_txt=args.write_to_txt,
        database_file=args.database_file,
        subset_chunk_size=args.subset_chunk_size,
        read_caption=True,
        verbose=True,
    )

    if args.change_source:
        old_img_src = args.old_img_src
        new_img_src = args.new_img_src

        def change_source(image_info):
            nonlocal old_img_src, new_img_src
            old_src = image_info.source.name
            new_src = old_img_src if old_src == new_img_src else None
            if new_src:
                image_info.image_path = image_info.source.parent / new_src / image_info.image_path.relative_to(image_info.source)
            return image_info

        dataset.apply_map(change_source)

    return dataset


def create_ui(
    args,
):
    from ..classes import ImageInfo, Caption
    from .utils import open_file_folder

    # ========================================= Base variables ========================================= #

    dataset = prepare_dataset(args)
    database_path = Path(args.database_file) if args.database_file else None
    subsets = dataset.subsets
    buffer = dataset.buffer

    wd14 = None
    character_feature_table = None
    random_sample_history = set()

    # ========================================= UI ========================================= #

    with gr.Blocks() as demo:
        with gr.Tab(label='Dataset'):
            with gr.Tab("Main") as main_tab:
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            subsets_selector = gr.Dropdown(
                                label='Category',
                                choices=[""] + sorted(subsets.keys()),
                                value="",
                                multiselect=False,
                                allow_custom_value=False,
                                min_width=256,
                            )
                            cur_chunk_index = gr.Number(label='Chunk', value=0, min_width=128, precision=0, scale=0)
                    with gr.Column():
                        log_box = gr.TextArea(
                            label='Log',
                            lines=1,
                            max_lines=1,
                        )
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            load_pre_chunk_btn = cc.EmojiButton(Emoji.black_left_pointing_double_triangle, scale=1)
                            load_next_chunk_btn = cc.EmojiButton(Emoji.black_right_pointing_double_triangle, scale=1)

                    with gr.Column():
                        ...
                with gr.Tab("Tagging") as tagging_tab:
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
                                cur_image_key = gr.Textbox(value=None, visible=False, label='Image Key')
                            with gr.Row():
                                reload_subset_btn = cc.EmojiButton(Emoji.anticlockwise)
                                remove_image_btn = cc.EmojiButton(Emoji.trash_bin, variant='stop')

                        with gr.Column():
                            with gr.Row():
                                with gr.Tab("Caption"):
                                    caption = gr.Textbox(
                                        label='Caption',
                                        value=None,
                                        container=True,
                                        show_copy_button=True,
                                        lines=6,
                                        max_lines=6,
                                    )
                                with gr.Tab("Metadata"):
                                    metadata_df = gr.Dataframe(
                                        value=None,
                                        label='Metadata',
                                        type='pandas',
                                        row_count=(1, 'fixed'),
                                    )

                            with gr.Row():
                                random_btn = cc.EmojiButton(Emoji.dice)
                                undo_btn = cc.EmojiButton(Emoji.leftwards)
                                redo_btn = cc.EmojiButton(Emoji.rightwards)
                                save_btn = cc.EmojiButton(Emoji.floppy_disk, variant='primary')
                            with gr.Row():
                                batch_proc = gr.Checkbox(
                                    label='Batch',
                                    value=False,
                                    container=False,
                                    scale=0,
                                    min_width=144
                                )

                            with gr.Tab("Quick Tagging"):
                                with gr.Row(variant='compact'):
                                    tagging_best_quality_btn = cc.EmojiButton(Emoji.love_emotion, variant='primary', scale=1)
                                    tagging_high_quality_btn = cc.EmojiButton(Emoji.heart, scale=1)
                                    tagging_low_quality_btn = cc.EmojiButton(Emoji.broken_heart, scale=1)
                                    tagging_hate_btn = cc.EmojiButton(Emoji.hate_emotion, variant='stop', scale=1)
                                with gr.Row(variant='compact'):
                                    tagging_color_btn = cc.EmojiButton(value='Color', scale=1, variant='primary')
                                    tagging_detailed_btn = cc.EmojiButton(value='Detail', scale=1, variant='primary')
                                    tagging_lowres_btn = cc.EmojiButton(value='Lowres', scale=1, variant='stop')
                                    tagging_messy_btn = cc.EmojiButton(value='Messy', scale=1, variant='stop')
                                with gr.Row(variant='compact'):
                                    tagging_aesthetic_btn = cc.EmojiButton(value='Aesthetic', scale=1, variant='primary')
                                    tagging_beautiful_btn = cc.EmojiButton(value='Beautiful', scale=1, variant='primary')
                                    tagging_x_btn = cc.EmojiButton(value='X', scale=1, variant='stop', visible=False)
                                    tagging_y_btn = cc.EmojiButton(value='Y', scale=1, variant='stop', visible=False)

                            with gr.Tab(label='Custom Tagging'):
                                add_tag_btns = []
                                tag_selectors = []
                                remove_tag_btns = []
                                for r in range(3):
                                    with gr.Row(variant='compact'):
                                        for c in range(2):
                                            add_tag_btns.append(cc.EmojiButton(Emoji.plus, variant='primary'))
                                            tag_selectors.append(gr.Dropdown(
                                                choices=list(tagging.CUSTOM_TAGS),
                                                value=None,
                                                multiselect=True,
                                                allow_custom_value=True,
                                                show_label=False,
                                                container=False,
                                                min_width=96,
                                            ))
                                            remove_tag_btns.append(cc.EmojiButton(Emoji.minus, variant='stop'))

                            with gr.Tab(label='Operational Tagging'):
                                with gr.Row():
                                    op_dropdown = gr.Dropdown(
                                        label='Op',
                                        choices=list(OPS.keys()),
                                        value=list(OPS.keys())[0],
                                        multiselect=False,
                                        allow_custom_value=False,
                                        scale=0,
                                        min_width=128,
                                    )
                                    op_tag_dropdown = gr.Dropdown(
                                        label='Tags',
                                        choices=[],
                                        value=None,
                                        allow_custom_value=True,
                                        multiselect=True,
                                    )

                                    caption_operation_btn = cc.EmojiButton(Emoji.black_right_pointing_triangle, variant='primary')

                                with gr.Row():
                                    condition_dropdown = gr.Dropdown(
                                        label='If',
                                        choices=list(CONDITION.keys()),
                                        value=list(CONDITION.keys())[0],
                                        multiselect=False,
                                        allow_custom_value=False,
                                        scale=0,
                                        min_width=128,
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
                                        choices=list(INCLUSION_RELATIONSHIP.keys()),
                                        value=list(INCLUSION_RELATIONSHIP.keys())[0],
                                        multiselect=False,
                                        allow_custom_value=False,
                                        scale=0,
                                        min_width=144,
                                    )

                            with gr.Tab(label="Optimizers"):
                                with gr.Row(variant='compact'):
                                    formalize_caption_btn = cc.EmojiButton("Formalize", scale=1, min_width=116)
                                    sort_caption_btn = cc.EmojiButton("Sort", scale=1, min_width=116)
                                    deduplicate_caption_btn = cc.EmojiButton("Deduplicate", scale=1, min_width=116)
                                    deoverlap_caption_btn = cc.EmojiButton("De-Overlap", scale=1, min_width=116)
                                with gr.Row(variant='compact'):
                                    defeature_caption_btn = cc.EmojiButton("De-Feature", scale=0, min_width=116)
                                    defeature_caption_threshold = gr.Slider(
                                        label='Threshold',
                                        value=0.3,
                                        minimum=0,
                                        maximum=1,
                                        step=0.01,
                                    )

                            with gr.Tab(label='WD14'):
                                with gr.Row():
                                    wd14_run_btn = cc.EmojiButton(Emoji.black_right_pointing_triangle, variant='primary', min_width=80)
                                    wd14_general_threshold = gr.Slider(
                                        label='General Threshold',
                                        value=0.35,
                                        minimum=0,
                                        maximum=1,
                                        step=0.01,
                                    )
                                    wd14_character_threshold = gr.Slider(
                                        label='Character Threshold',
                                        value=0.35,
                                        minimum=0,
                                        maximum=1,
                                        step=0.01,
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
                                open_folder_btn = cc.EmojiButton(Emoji.open_file_folder)
                        with gr.Column(scale=5):
                            ...

                with gr.Tab("Database") as database_tab:
                    database = gr.Dataframe(
                        value=None,
                        label='Database',
                        type='pandas',
                    )

            with gr.Tab("Buffer") as buffer_tab:
                buffer_df = gr.Dataframe(
                    value=None,
                    label='Buffer',
                    type='pandas',
                    row_count=(20, 'fixed'),
                )

        # ========================================= Functions ========================================= #

        def kwargs_setter(func, **kwargs):
            def wrapper(*args):
                return func(*args, **kwargs)
            return wrapper

        # ========================================= Tab changing parser ========================================= #
        activating_tab = gr.State(value='tagging')

        tagging_tab.select(
            fn=lambda subset_key, chunk_index: ('tagging', *show_subset(subset_key, chunk_index).values()),
            inputs=[subsets_selector, cur_chunk_index],
            outputs=[activating_tab, showcase, cur_image_key, cur_chunk_index],
        )

        database_tab.select(
            fn=lambda subset_key, chunk_index: ('database', *show_database(subset_key, chunk_index).values()),
            inputs=[subsets_selector, cur_chunk_index],
            outputs=[activating_tab, database, cur_chunk_index],
        )

        # ========================================= Subset changing parser ========================================= #

        def change_subset(activating_tab, subset_key, chunk_index):
            if activating_tab == 'tagging':
                res = show_subset(subset_key, chunk_index)
                return res
            elif activating_tab == 'database':
                res = show_database(subset_key, chunk_index)
                return res

        subset_change_inputs = [activating_tab, subsets_selector, cur_chunk_index]
        subset_change_outputs = [showcase, cur_image_key, database, cur_chunk_index]

        subsets_selector.input(
            fn=change_subset,
            inputs=subset_change_inputs,
            outputs=subset_change_outputs,
            show_progress=True,
            trigger_mode='multiple',
        )

        def _get_new_select_img_key(dset):
            pre_idx = dataset.selected.index
            if pre_idx is not None and pre_idx < len(dset):
                new_img_key = dset.keys()[pre_idx]
            else:
                new_img_key = None
            return new_img_key

        def show_subset(subset_key, chunk_index):
            if subset_key is None or subset_key == '':
                return {
                    showcase: gr.update(value=None, label='Showcase'),
                    cur_image_key: None,
                    cur_chunk_index: 0,
                }

            subset = subsets[subset_key]
            chunk_index = min(max(chunk_index, 1), subset.num_chunks)
            subset_chunk = subset.chunk(chunk_index - 1)

            new_select_img_key = _get_new_select_img_key(subset_chunk)

            return {
                showcase: gr.update(value=[(v.image_path, k) for k, v in subset_chunk.items()], label=f"Showcase  {subset_key}  {chunk_index}/{subset.num_chunks}"),
                cur_image_key: new_select_img_key,
                cur_chunk_index: chunk_index,
            }

        def show_database(subset_key, chunk_index):
            if subset_key is None or subset_key == '':
                return {
                    database: gr.update(value=None, label='Database'),
                    cur_chunk_index: 0,
                }

            subset = subsets[subset_key]
            chunk_index = min(max(chunk_index, 1), subset.num_chunks)
            subset_chunk = subset.chunk(chunk_index - 1)

            return {
                database: gr.update(value=subset_chunk.df(), label=f"Database  {subset_key}  {chunk_index}/{subset.num_chunks}"),
                cur_chunk_index: chunk_index,
            }

        def change_chunk_index(activating_tab, subset_key, chunk_index):
            return change_subset(activating_tab, subset_key, chunk_index) if chunk_index != 0 else {cur_chunk_index: 0}

        cur_chunk_index.input(
            fn=lambda activating_tab, subset_key, chunk_index: change_chunk_index(activating_tab, subset_key, chunk_index),
            inputs=subset_change_inputs,
            outputs=subset_change_outputs,
            show_progress=True,
        )

        load_pre_chunk_btn.click(
            fn=lambda activating_tab, subset_key, chunk_index: change_chunk_index(activating_tab, subset_key, chunk_index - 1),
            inputs=subset_change_inputs,
            outputs=subset_change_outputs,
            show_progress=True,
        )

        load_next_chunk_btn.click(
            fn=lambda activating_tab, subset_key, chunk_index: change_chunk_index(activating_tab, subset_key, chunk_index + 1),
            inputs=subset_change_inputs,
            outputs=subset_change_outputs,
            show_progress=True,
        )

        # ========================================= Showcase ========================================= #

        def select_image_key(selected: gr.SelectData):
            if selected is None:
                return None, None
            image_key = dataset.select(selected)
            return image_key

        showcase.select(
            fn=select_image_key,
            outputs=[cur_image_key],
        )

        def track_image_key(image_key):
            if image_key is None or image_key == '':
                return None, gr.update(value=None, label='Caption')
            image_key = Path(image_key).stem
            image_info = dataset.get(image_key, None)
            if image_info is None:
                raise ValueError(f"image key {image_key} not found in dataset")
            image_path = str(image_info.image_path) if image_info.image_path.is_file() else None
            caption = str(image_info.caption) if image_info.caption is not None else None
            return image_path, gr.update(value=caption, label=f"Caption: {image_key}")

        cur_image_key.change(
            fn=track_image_key,
            inputs=[cur_image_key],
            outputs=[image_path, caption],
        )

        def track_caption(image_key):
            if image_key is None or image_key == '' or image_key not in dataset:
                return None
            image_info: ImageInfo = dataset[image_key]
            info_dict = image_info.dict()
            artist = info_dict.get('artist', None)
            quality = info_dict.get('quality', None)
            styles = info_dict.get('styles', None)
            characters = info_dict.get('characters', None)
            data = [{'Artist': artist, 'Quality': quality, 'Styles': styles, 'Characters': characters}]
            df = pandas.DataFrame(data=data, columns=list(data[0].keys()))
            return gr.update(value=df, label=f"Metadata: {image_key}")

        caption.change(
            fn=track_caption,
            inputs=[cur_image_key],
            outputs=[metadata_df],
            trigger_mode='always_last',
        )

        reload_subset_btn.click(
            fn=show_subset,
            inputs=[subsets_selector, cur_chunk_index],
            outputs=[showcase, cur_image_key, cur_chunk_index],
            show_progress=True,
        )

        def remove_image(image_key, chunk_index):
            if image_key is None or image_key == '':
                return {log_box: f"empty image key"}
            subset_key = dataset[image_key].category
            dataset.remove(image_key)
            res = show_subset(subset_key, chunk_index)
            res.update({log_box: f"removed: {image_key}"})
            return res

        remove_image_btn.click(
            fn=remove_image,
            inputs=[cur_image_key, cur_chunk_index],
            outputs=subset_change_outputs + [log_box],
        )

        # ========================================= Base Tagging Buttons ========================================= #

        def random_sample(subset_key):
            subset = subsets[subset_key] if subset_key is not None and subset_key != "" else dataset
            if len(subset) == 0:
                return {log_box: f"empty subset {subset_key}"}
            subset = subset.make_subset(condition=lambda x: x.key not in random_sample_history)
            if len(subset) == 0:
                return {log_box: f"no more image to sample"}
            image_key = random.choice(list(subset.keys()))
            random_sample_history.add(image_key)
            image_info = dataset[image_key]
            dataset.select(image_key)
            return {
                showcase: gr.update(value=[(image_info.image_path, image_key)], label=f"Showcase"),
                cur_image_key: image_key,
                cur_chunk_index: 1,
            }

        random_btn.click(
            fn=random_sample,
            inputs=[subsets_selector],
            outputs=[showcase, cur_image_key, cur_chunk_index, log_box],
        )

        def edit_caption_wrapper(func: Callable[[ImageInfo, Tuple[Any, ...], Dict[str, Any]], Caption]) -> Tuple[str, str]:
            def wrapper(image_key, batch, *args, progress: gr.Progress = gr.Progress(track_tqdm=True), **kwargs):
                if image_key is None or image_key == '':
                    return gr.update(), f"empty image key"
                image_key = Path(image_key).stem
                image_info = dataset[image_key]
                proc_func_name = func.__name__

                def edit(image_key, image_info, *args, **kwargs):
                    new_caption = func(image_info.copy(), *args, **kwargs)
                    if image_info.caption == new_caption:
                        return False
                    new_img_info = image_info.copy()
                    new_img_info.caption = new_caption
                    dataset.set(image_key, new_img_info)
                    return True

                res = []
                if batch:
                    subset_key = image_info.category
                    subset = subsets[subset_key]
                    for img_key, img_info in tqdm(subset.items(), desc=f'{proc_func_name} batch processing'):
                        res.append(edit(img_key, img_info, *args, **kwargs))
                else:
                    res.append(edit(image_key, image_info, *args, **kwargs))

                new_img_info = dataset[image_key]
                new_caption = new_img_info.caption

                if any(res):
                    return str(new_caption) if new_caption else None, f"{proc_func_name.replace('_', ' ')}: {image_key}"
                else:
                    return gr.update(), f"no change"
            return wrapper

        def write_caption(image_info, caption):
            return Caption(caption) if caption is not None and caption.strip() != '' else None

        caption.blur(
            fn=edit_caption_wrapper(write_caption),
            inputs=[image_path, gr.State(False), caption],
            outputs=[caption, log_box],
        )

        def undo(image_info):
            image_key = image_info.key
            image_info = dataset.undo(image_key)
            return image_info.caption

        undo_btn.click(
            fn=edit_caption_wrapper(undo),
            inputs=[cur_image_key, batch_proc],
            outputs=[caption, log_box],
        )

        def redo(image_info):
            image_key = image_info.key
            image_info = dataset.redo(image_key)
            return image_info.caption

        redo_btn.click(
            fn=edit_caption_wrapper(redo),
            inputs=[cur_image_key, batch_proc],
            outputs=[caption, log_box],
        )

        def save_to_disk(progress: gr.Progress = gr.Progress(track_tqdm=True)):
            dataset.save(progress=progress)
            return f"database saved: {database_path.absolute().as_posix()}"

        save_btn.click(
            fn=save_to_disk,
            inputs=[],
            outputs=[log_box],
        )

        # ========================================= Quick Tagging ========================================= #

        def change_quality(image_info, quality):
            return image_info.caption.with_quality(quality)

        tagging_best_quality_btn.click(
            fn=edit_caption_wrapper(kwargs_setter(change_quality, quality='best')),
            inputs=[image_path, batch_proc],
            outputs=[caption, log_box],
        )

        tagging_high_quality_btn.click(
            fn=edit_caption_wrapper(kwargs_setter(change_quality, quality='high')),
            inputs=[image_path, batch_proc],
            outputs=[caption, log_box],
        )

        tagging_low_quality_btn.click(
            fn=edit_caption_wrapper(kwargs_setter(change_quality, quality='low')),
            inputs=[image_path, batch_proc],
            outputs=[caption, log_box],
        )

        tagging_hate_btn.click(
            fn=edit_caption_wrapper(kwargs_setter(change_quality, quality='hate')),
            inputs=[image_path, batch_proc],
            outputs=[caption, log_box],
        )

        def add_tags(image_info, tags):
            caption = tags | image_info.caption
            return caption

        def remove_tags(image_info, tags):
            caption = image_info.caption - tags
            return caption

        tagging_color_btn.click(
            fn=edit_caption_wrapper(kwargs_setter(add_tags, tags='beautiful color')),
            inputs=[image_path, batch_proc],
            outputs=[caption, log_box],
        )

        tagging_detailed_btn.click(
            fn=edit_caption_wrapper(kwargs_setter(add_tags, tags='detailed')),
            inputs=[image_path, batch_proc],
            outputs=[caption, log_box],
        )

        tagging_lowres_btn.click(
            fn=edit_caption_wrapper(kwargs_setter(add_tags, tags='lowres')),
            inputs=[image_path, batch_proc],
            outputs=[caption, log_box],
        )

        tagging_messy_btn.click(
            fn=edit_caption_wrapper(kwargs_setter(add_tags, tags='messy')),
            inputs=[image_path, batch_proc],
            outputs=[caption, log_box],
        )

        tagging_aesthetic_btn.click(
            fn=edit_caption_wrapper(kwargs_setter(add_tags, tags='aesthetic')),
            inputs=[image_path, batch_proc],
            outputs=[caption, log_box],
        )

        tagging_beautiful_btn.click(
            fn=edit_caption_wrapper(kwargs_setter(add_tags, tags='beautiful')),
            inputs=[image_path, batch_proc],
            outputs=[caption, log_box],
        )

        tagging_x_btn.click(
            fn=edit_caption_wrapper(kwargs_setter(add_tags, tags='x')),
            inputs=[image_path, batch_proc],
            outputs=[caption, log_box],
        )

        tagging_y_btn.click(
            fn=edit_caption_wrapper(kwargs_setter(add_tags, tags='y')),
            inputs=[image_path, batch_proc],
            outputs=[caption, log_box],
        )

        # ========================================= Custom Tagging ========================================= #

        for add_tag_btn, tag_selector, remove_tag_btn in zip(add_tag_btns, tag_selectors, remove_tag_btns):
            add_tag_btn.click(
                fn=edit_caption_wrapper(add_tags),
                inputs=[image_path, batch_proc, tag_selector],
                outputs=[caption, log_box],
            )
            remove_tag_btn.click(
                fn=edit_caption_wrapper(remove_tags),
                inputs=[image_path, batch_proc, tag_selector],
                outputs=[caption, log_box],
            )

        # ========================================= Caption Operation ========================================= #

        def caption_operation(image_info, op, op_tags, condition, cond_tags, inclusion_relationship):
            # print(f"op: {op} | op_tags: {op_tags} | condition: {condition} | cond_tags: {cond_tags} | inclusion_relationship: {inclusion_relationship}")
            caption = image_info.caption or Caption()
            if op_tags is None or len(op_tags) == 0:
                return caption
            op_func = OPS[op]
            op_caption = Caption(op_tags)
            do_condition = condition is not None and condition != ''
            if do_condition and cond_tags and len(cond_tags) > 0:
                cond_func = CONDITION[condition]
                cond_tags = set(cond_tags)
                inclusion_relationship_func = INCLUSION_RELATIONSHIP[inclusion_relationship]

                if not cond_func(inclusion_relationship_func(caption, cond_tag) for cond_tag in cond_tags):
                    return caption

            caption = op_func(caption, op_caption)
            # print(f"caption: {caption}")
            return caption

        caption_operation_btn.click(
            fn=edit_caption_wrapper(caption_operation),
            inputs=[image_path, batch_proc, op_dropdown, op_tag_dropdown, condition_dropdown, cond_tag_dropdown, inclusion_relationship_dropdown],
            outputs=[caption, log_box],
        )

        # ========================================= Optimizers ========================================= #
        def sort_caption(image_info):
            tagging.init_priority_tags()
            return image_info.caption @ tagging.PRIORITY_REGEX

        sort_caption_btn.click(
            fn=edit_caption_wrapper(sort_caption),
            inputs=[image_path, batch_proc],
            outputs=[caption, log_box],
        )

        def formalize_caption(image_info):
            return image_info.caption.formalized()

        formalize_caption_btn.click(
            fn=edit_caption_wrapper(formalize_caption),
            inputs=[image_path, batch_proc],
            outputs=[caption, log_box],
        )

        def deduplicate_caption(image_info):
            return image_info.caption.unique()

        deduplicate_caption_btn.click(
            fn=edit_caption_wrapper(deduplicate_caption),
            inputs=[image_path, batch_proc],
            outputs=[caption, log_box],
        )

        def deoverlap_caption(image_info):
            return image_info.caption.deovlped()

        deoverlap_caption_btn.click(
            fn=edit_caption_wrapper(deoverlap_caption),
            inputs=[image_path, batch_proc],
            outputs=[caption, log_box],
        )

        def defeature_caption(image_info, threshold):
            nonlocal character_feature_table
            if character_feature_table is None:
                from ..tools import make_character_feature_table
                character_feature_table = make_character_feature_table(dataset, threshold=None, verbose=True)
            return image_info.caption.defeatured(ref=character_feature_table, threshold=threshold)

        defeature_caption_btn.click(
            fn=edit_caption_wrapper(defeature_caption),
            inputs=[image_path, batch_proc, defeature_caption_threshold],
            outputs=[caption, log_box],
        )

        # ========================================= WD14 ========================================= #

        def wd14_tagging(image_info, general_threshold, character_threshold):
            nonlocal wd14
            if wd14 is None:
                from modules.compoents import WaifuTagger
                wd14 = WaifuTagger(verbose=True)
            image = Image.open(image_info.image_path)
            caption = wd14(image, general_threshold=general_threshold, character_threshold=character_threshold)
            return caption

        wd14_run_btn.click(
            fn=edit_caption_wrapper(wd14_tagging),
            inputs=[image_path, batch_proc, wd14_general_threshold, wd14_character_threshold],
            outputs=[caption, log_box],
        )

        # ========================================= Open file folder ========================================= #

        open_folder_btn.click(
            fn=open_file_folder,
            inputs=[image_path],
            outputs=[],
        )

        # ========================================= Buffer ========================================= #

        def show_buffer():
            return buffer.df()

        buffer_tab.select(
            fn=show_buffer,
            outputs=[buffer_df],
        )

    return demo
