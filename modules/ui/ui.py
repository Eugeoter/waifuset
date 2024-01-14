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

JOINER = {
    'and': lambda x, y: x & y,
    'or': lambda x, y: x | y,
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
        chunk_size=args.chunk_size,
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
    from ..classes import Dataset, ImageInfo, Caption
    from .ui_dataset import UIChunkedDataset, UISampleHistory
    from .utils import open_file_folder

    # ========================================= Base variables ========================================= #

    dataset = prepare_dataset(args)
    database_path = Path(args.database_file) if args.database_file else None
    buffer = dataset.buffer
    sample_history = UISampleHistory()
    tag_table = None

    cur_dataset = dataset
    wd14 = None
    character_feature_table = None

    # ========================================= UI ========================================= #

    with gr.Blocks() as demo:
        with gr.Tab(label='Dataset'):
            with gr.Tab("Main") as main_tab:
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            reload_category_btn = cc.EmojiButton(Emoji.anticlockwise)
                            category_selector = gr.Dropdown(
                                label='Category',
                                choices=[""] + dataset.categories,
                                value="",
                                multiselect=False,
                                allow_custom_value=False,
                                min_width=256,
                            )
                            cur_chunk_index = gr.Number(label=f'Chunk {1}/{cur_dataset.num_chunks}', value=1, min_width=128, precision=0, scale=0)
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
                                    value=[(v.image_path, k) for k, v in cur_dataset.chunk(0).items()],
                                    rows=4,
                                    columns=4,
                                    container=True,
                                    object_fit='scale-down',
                                    height=512,
                                )
                                cur_image_key = gr.Textbox(value=None, visible=False, label='Image Key')
                            with gr.Row():
                                pre_hist_btn = cc.EmojiButton(Emoji.black_left_pointing_double_triangle, scale=1)
                                reload_subset_btn = cc.EmojiButton(Emoji.anticlockwise)
                                remove_image_btn = cc.EmojiButton(Emoji.trash_bin, variant='stop')
                                next_hist_btn = cc.EmojiButton(Emoji.black_right_pointing_double_triangle, scale=1)

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
                                set_category_btn = cc.EmojiButton(Emoji.top_left_arrow)
                                undo_btn = cc.EmojiButton(Emoji.leftwards)
                                redo_btn = cc.EmojiButton(Emoji.rightwards)
                                save_btn = cc.EmojiButton(Emoji.floppy_disk, variant='primary')
                                cancel_btn = cc.EmojiButton(Emoji.no_entry, variant='stop')
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
                                def custom_tagging_row():
                                    with gr.Row(variant='compact'):
                                        add_tag_btn = cc.EmojiButton(Emoji.plus, variant='primary')
                                        tag_selector = gr.Dropdown(
                                            choices=list(tagging.CUSTOM_TAGS),
                                            value=None,
                                            multiselect=True,
                                            allow_custom_value=True,
                                            show_label=False,
                                            container=False,
                                            min_width=96,
                                        )
                                        remove_tag_btn = cc.EmojiButton(Emoji.minus, variant='stop')
                                    return add_tag_btn, tag_selector, remove_tag_btn

                                add_tag_btns = []
                                tag_selectors = []
                                remove_tag_btns = []
                                for r in range(3):
                                    add_tag_btn, tag_selector, remove_tag_btn = custom_tagging_row()
                                    add_tag_btns.append(add_tag_btn)
                                    tag_selectors.append(tag_selector)
                                    remove_tag_btns.append(remove_tag_btn)

                                with gr.Accordion(label='More', open=False):
                                    for r in range(6):
                                        add_tag_btn, tag_selector, remove_tag_btn = custom_tagging_row()
                                        add_tag_btns.append(add_tag_btn)
                                        tag_selectors.append(tag_selector)
                                        remove_tag_btns.append(remove_tag_btn)

                            with gr.Tab(label='Operational Tagging'):
                                with gr.Row(variant='compact'):
                                    cap_op_op_dropdown = gr.Dropdown(
                                        label='Op',
                                        choices=list(OPS.keys()),
                                        value=list(OPS.keys())[0],
                                        multiselect=False,
                                        allow_custom_value=False,
                                        scale=0,
                                        min_width=128,
                                    )
                                    cap_op_op_tag_dropdown = gr.Dropdown(
                                        label='Tags',
                                        choices=[],
                                        value=None,
                                        allow_custom_value=True,
                                        multiselect=True,
                                    )

                                    operate_caption_btn = cc.EmojiButton(Emoji.black_right_pointing_triangle, variant='primary')

                                with gr.Row(variant='compact'):
                                    cap_op_cond_dropdown = gr.Dropdown(
                                        label='If',
                                        choices=list(CONDITION.keys()),
                                        value=list(CONDITION.keys())[0],
                                        multiselect=False,
                                        allow_custom_value=False,
                                        scale=0,
                                        min_width=128,
                                    )

                                    cap_op_cond_tag_dropdown = gr.Dropdown(
                                        label='Tags',
                                        choices=[],
                                        value=None,
                                        allow_custom_value=True,
                                        multiselect=True,
                                    )

                                    cap_op_incl_rel_dropdown = gr.Dropdown(
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
                                with gr.Row(variant='compact'):
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
                                with gr.Row(variant='compact'):
                                    wd14_os_mode = gr.Radio(
                                        label='OS Mode',
                                        choices=['overwrite', 'append', 'prepend', 'ignore'],
                                        value='overwrite',
                                        scale=0,
                                        min_width=128,
                                    )

                    with gr.Row():
                        with gr.Column(scale=4):
                            with gr.Tab("Info"):
                                with gr.Row(variant='compact'):
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
                                with gr.Row(variant='compact'):
                                    resolution = gr.Textbox(
                                        label='Resolution',
                                        value=None,
                                        container=True,
                                        max_lines=2,
                                        lines=1,
                                        show_copy_button=True,
                                        interactive=False,
                                    )

                        with gr.Column(scale=4):
                            with gr.Tab("Query"):
                                with gr.Row(variant='compact'):
                                    query_include_condition = gr.Dropdown(
                                        label='If',
                                        choices=list(CONDITION.keys()),
                                        value=list(CONDITION.keys())[0],
                                        multiselect=False,
                                        allow_custom_value=False,
                                        min_width=128,
                                        scale=0,
                                    )
                                    query_include_tags = gr.Dropdown(
                                        label='Include',
                                        choices=None,
                                        allow_custom_value=True,
                                        multiselect=True,
                                    )
                                    query_joiner_dropdown = gr.Dropdown(
                                        label='Joiner',
                                        choices=list(JOINER.keys()),
                                        value=list(JOINER.keys())[0],
                                        multiselect=False,
                                        allow_custom_value=False,
                                        min_width=108,
                                        scale=0,
                                    )
                                with gr.Row(variant='compact'):
                                    query_exclude_condition = gr.Dropdown(
                                        label='If',
                                        choices=list(CONDITION.keys()),
                                        value=list(CONDITION.keys())[0],
                                        multiselect=False,
                                        allow_custom_value=False,
                                        min_width=128,
                                        scale=0,
                                    )
                                    query_exclude_tags = gr.Dropdown(
                                        label='Excludes',
                                        choices=None,
                                        allow_custom_value=True,
                                        multiselect=True,
                                    )
                                    query_btn = cc.EmojiButton(Emoji.right_pointing_magnifying_glass, variant='primary', min_width=100)

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

        def dataset_to_gallery(dset):
            return [(v.image_path if v.image_path.is_file() else None, k) for k, v in dset.items()]

        def get_new_img_key(dset):
            r"""
            Get the image key that should be selected if the showing dataset is changed by `dset`. Doesn't depend on what the current dataset is.
            """
            pre_idx = dataset.selected.index
            if pre_idx is not None and pre_idx < len(dset):
                new_img_key = dset.keys()[pre_idx]
            else:
                new_img_key = None
            return new_img_key

        def correct_chunk_idx(dset: UIChunkedDataset, chunk_index):
            r"""
            Correct the chunk index of `dset` to `chunk_index`
            """
            if chunk_index is None or chunk_index == 0:
                chunk_index = 1
            else:
                chunk_index = min(max(chunk_index, 1), dset.num_chunks)
            return chunk_index

        def change_current_dataset(dset):
            r"""
            Change the current dataset to `dset`
            """
            nonlocal cur_dataset
            cur_dataset = dset

        def show_dataset(dset=None, new_chunk_index=1):
            r"""
            Convert `dset` to gallery value and image key that should be selected
            """
            if isinstance(dset, UIChunkedDataset):
                new_chunk_index = correct_chunk_idx(dset, new_chunk_index)
                chunk = dset.chunk(new_chunk_index - 1) if isinstance(dset, UIChunkedDataset) else dset
            elif isinstance(dset, Dataset):
                chunk = dset
                new_chunk_index = 1
            new_img_key = get_new_img_key(chunk)
            gallery = dataset_to_gallery(chunk)
            return {
                showcase: gallery,
                cur_image_key: new_img_key,
                cur_chunk_index: gr.update(value=new_chunk_index, label=f'Chunk {new_chunk_index}/{cur_dataset.num_chunks}')
            }

        def show_database(dset=None, new_chunk_index=1):
            r"""
            Convert `dset` to dataframe
            """
            new_chunk_index = correct_chunk_idx(dset, new_chunk_index)
            chunk = dset.chunk(new_chunk_index - 1) if isinstance(dset, UIChunkedDataset) else dset
            df = chunk.df()
            return {
                database: df,
                cur_chunk_index: gr.update(value=new_chunk_index, label=f'Chunk {new_chunk_index}/{dset.num_chunks}'),
            }

        # ========================================= Tab changing parser ========================================= #
        activating_tab_name = "tagging"

        def change_activating_tab_name(tab_name):
            r"""
            Change the activating tab name to `tab_name`
            """
            nonlocal activating_tab_name
            activating_tab_name = tab_name

        # ========================================= Subset key selector ========================================= #

        def change_to_dataset(dset: UIChunkedDataset = None, new_chunk_index=1, new_activating_tab_name=activating_tab_name):
            r"""
            Change current dataset to another dataset `dset` and show its chunk
            """
            if dset is None:
                dset = cur_dataset
            change_activating_tab_name(new_activating_tab_name)
            change_current_dataset(dset)
            if new_activating_tab_name == 'tagging':
                res = show_dataset(dset, new_chunk_index)
            elif new_activating_tab_name == 'database':
                res = show_database(dset, new_chunk_index)
            return res

        def change_to_category(category):
            r"""
            Change current dataset to another dataset with category `category` and show its chunk
            """
            if category is None or category == '':
                dset = UIChunkedDataset(dataset, chunk_size=args.chunk_size)
            else:
                dset = dataset.make_subset(condition=lambda img_info: img_info.category == category, chunk_size=args.chunk_size)
            return change_to_dataset(dset, new_chunk_index=1)

        selector_change_outputs = [showcase, cur_image_key, database, cur_chunk_index, category_selector, log_box]

        category_selector.input(
            fn=change_to_category,
            inputs=[category_selector],
            outputs=selector_change_outputs,
            show_progress=True,
            trigger_mode='multiple',
        )

        reload_category_btn.click(
            fn=change_to_category,
            inputs=[category_selector],
            outputs=selector_change_outputs,
            show_progress=True,
            trigger_mode='multiple',
        )

        set_category_btn.click(
            fn=lambda image_key: {**change_to_category(cur_dataset[image_key].category), category_selector: cur_dataset[image_key].category},
            inputs=[cur_image_key],
            outputs=selector_change_outputs,
            show_progress=True,
            trigger_mode='multiple',
        )

        reload_subset_btn.click(  # same as above
            fn=lambda chunk_index: change_to_dataset(cur_dataset, chunk_index),
            inputs=[cur_chunk_index],
            outputs=selector_change_outputs,
            show_progress=True,
        )

        tagging_tab.select(
            fn=lambda chunk_index: change_to_dataset(cur_dataset, chunk_index, new_activating_tab_name='tagging'),
            inputs=[cur_chunk_index],
            outputs=selector_change_outputs,
            show_progress=True,
        )

        database_tab.select(
            fn=lambda chunk_index: change_to_dataset(cur_dataset, chunk_index, new_activating_tab_name='database'),
            inputs=[cur_chunk_index],
            outputs=selector_change_outputs,
            show_progress=True,
        )

        cur_chunk_index.submit(
            fn=lambda chunk_index: change_to_dataset(cur_dataset, chunk_index),
            inputs=[cur_chunk_index],
            outputs=selector_change_outputs,
            show_progress=True,
        )

        load_pre_chunk_btn.click(
            fn=lambda chunk_index: change_to_dataset(cur_dataset, chunk_index - 1),
            inputs=[cur_chunk_index],
            outputs=selector_change_outputs,
            show_progress=True,
        )

        load_next_chunk_btn.click(
            fn=lambda chunk_index: change_to_dataset(cur_dataset, chunk_index + 1),
            inputs=[cur_chunk_index],
            outputs=selector_change_outputs,
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
            if image_key is None or image_key == '':  # no image key selected
                return None, gr.update(value=None, label='Caption'), None
            image_key = Path(image_key).stem
            image_info = dataset.get(image_key, None)
            if image_info is None:
                raise ValueError(f"image key {image_key} not found in dataset")
            image_path = str(image_info.image_path) if image_info.image_path.is_file() else None
            caption = str(image_info.caption) if image_info.caption is not None else None
            return image_path, gr.update(value=caption, label=f"Caption: {image_key}"), f"{image_info.original_size[0]}x{image_info.original_size[1]}"

        cur_image_key.change(
            fn=track_image_key,
            inputs=[cur_image_key],
            outputs=[image_path, caption, resolution],
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

        def remove_image(image_key, chunk_index):
            if image_key is None or image_key == '':
                return {log_box: f"empty image key"}
            dataset.remove(image_key)
            del cur_dataset[image_key]  # remove from current dataset
            return change_to_dataset(new_chunk_index=chunk_index)

        remove_image_btn.click(
            fn=remove_image,
            inputs=[cur_image_key, cur_chunk_index],
            outputs=selector_change_outputs,
        )

        def show_i_th_sample(index):
            if len(sample_history) == 0:
                return {
                    log_box: f"empty sample history",
                }
            new_img_key = sample_history.select(index)
            return {
                showcase: dataset_to_gallery(Dataset(dataset[new_img_key])),
                cur_image_key: new_img_key,
            }

        pre_hist_btn.click(
            fn=lambda: show_i_th_sample(sample_history.index - 1),
            inputs=[],
            outputs=selector_change_outputs,
        )

        next_hist_btn.click(
            fn=lambda: show_i_th_sample(sample_history.index + 1),
            inputs=[],
            outputs=selector_change_outputs,
        )

        # ========================================= Base Tagging Buttons ========================================= #

        def random_sample(n=1):
            subset = cur_dataset
            if len(subset) == 0:
                return {log_box: f"empty dataset"}
            subset = subset.make_subset(condition=lambda x: x.key not in sample_history)
            if len(subset) == 0:
                return {log_box: f"no more image to sample"}
            samples: Dataset = subset.sample(n=n, randomly=True)
            for sample in samples.keys():
                sample_history.add(sample)
            new_img_key = sample_history.select(len(sample_history) - 1)
            return {
                showcase: dataset_to_gallery(Dataset(dataset[new_img_key])),
                cur_image_key: new_img_key,
                cur_chunk_index: gr.update(value=1, label=f'Chunk 1/{cur_dataset.num_chunks}'),
            }

        random_btn.click(
            fn=random_sample,
            inputs=[],
            outputs=selector_change_outputs,
        )

        def edit_caption_wrapper(func: Callable[[ImageInfo, Tuple[Any, ...], Dict[str, Any]], Caption]) -> Tuple[str, str]:
            def wrapper(image_key, batch, *args, progress: gr.Progress = gr.Progress(track_tqdm=True), **kwargs):
                if image_key is None or image_key == '':
                    return gr.update(), f"empty image key"
                image_key = Path(image_key).stem
                proc_func_name = func.__name__

                def edit(image_info, *args, **kwargs):
                    new_caption = func(image_info.copy(), *args, **kwargs)
                    if image_info.caption == new_caption:
                        return None
                    new_img_info = image_info.copy()
                    new_img_info.caption = new_caption
                    return new_img_info

                results = []
                if batch:
                    subset = cur_dataset
                    for image_info in tqdm(subset.values(), desc=f'{proc_func_name} batch processing'):
                        results.append(edit(image_info, *args, **kwargs))
                else:
                    results.append(edit(dataset[image_key], *args, **kwargs))

                # write to dataset
                for res in results:
                    if res is not None:
                        dataset.set(res.key, res)

                new_img_info = dataset[image_key]
                new_caption = new_img_info.caption

                if any(results):
                    return str(new_caption) if new_caption else None, f"{proc_func_name.replace('_', ' ')}: {image_key}"
                else:
                    return gr.update(), f"no change"
            return wrapper

        def cancel_edit_caption():
            return gr.update(), f"edit caption cancelled."

        cancel_event = cancel_btn.click(
            fn=cancel_edit_caption,
            outputs=[caption, log_box]
        )

        def write_caption(image_info, caption):
            return Caption(caption) if caption is not None and caption.strip() != '' else None

        caption.blur(
            fn=edit_caption_wrapper(write_caption),
            inputs=[image_path, gr.State(False), caption],
            outputs=[caption, log_box],
            cancels=cancel_event,
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
            fn=edit_caption_wrapper(kwargs_setter(change_quality, quality='worst')),
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
                incl_func = INCLUSION_RELATIONSHIP[inclusion_relationship]

                if not cond_func(incl_func(caption, cond_tag) for cond_tag in cond_tags):
                    return caption

            caption = op_func(caption, op_caption)
            # print(f"caption: {caption}")
            return caption

        operate_caption_btn.click(
            fn=edit_caption_wrapper(caption_operation),
            inputs=[image_path, batch_proc, cap_op_op_dropdown, cap_op_op_tag_dropdown, cap_op_cond_dropdown, cap_op_cond_tag_dropdown, cap_op_incl_rel_dropdown],
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

        def wd14_tagging(image_info, general_threshold, character_threshold, os_mode):
            nonlocal wd14
            old_caption = image_info.caption
            if old_caption is not None and os_mode == 'ignore':
                return old_caption
            if wd14 is None:
                from modules.compoents import WaifuTagger
                wd14 = WaifuTagger(verbose=True)
            image = Image.open(image_info.image_path)
            wd14_caption = wd14(image, general_threshold=general_threshold, character_threshold=character_threshold)
            if os_mode == 'overwrite' or os_mode == 'ignore':
                caption = wd14_caption
            elif os_mode == 'append':
                caption = old_caption | wd14_caption
            elif os_mode == 'prepend':
                caption = wd14_caption | old_caption
            else:
                raise ValueError(f"invalid os_mode: {os_mode}")
            return caption

        wd14_run_btn.click(
            fn=edit_caption_wrapper(wd14_tagging),
            inputs=[image_path, batch_proc, wd14_general_threshold, wd14_character_threshold, wd14_os_mode],
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

        # ========================================= Query ========================================= #

        def query_tag_table(include_condition, include_tags, joiner, exclude_condition, exclude_tags, progress: gr.Progress = gr.Progress(track_tqdm=True)):
            nonlocal tag_table
            if tag_table is None:
                dataset.init_tag_table()
                tag_table = dataset.tag_table

            joiner_func = JOINER[joiner]

            # print(f"subset_key: {subset_key}")
            # print(f"include_tags: {include_tags}")
            # print(f"exclude_tags: {exclude_tags}")

            subset = cur_dataset

            incl_set = set()
            for tag in tqdm(include_tags, desc='Including tags'):  # calculate the union of all key(s), s ∈ include_tags
                if tag not in tag_table:
                    continue
                tag_set = tag_table[tag]
                # filter by subset
                if subset:
                    tag_set = set(img_key for img_key in tag_set if img_key in subset)
                if len(tag_set) == 0:
                    continue

                if include_condition == 'any':
                    incl_set.update(tag_set)
                else:
                    incl_set.intersection_update(tag_set)

            excl_set = set()
            for tag in tqdm(exclude_tags, desc='Excluding tags'):  # calculate the union of all key(s), s ∈ exclude_tags
                if tag not in tag_table:
                    continue
                tag_set = tag_table[tag]
                if subset:
                    tag_set = set(img_key for img_key in tag_set if img_key in subset)
                if len(tag_set) == 0:
                    continue

                if exclude_condition == 'any':
                    excl_set.update(tag_set)
                else:
                    excl_set.intersection_update(tag_set)

            excl_set = set(cur_dataset.keys()) - excl_set  # calculate the complement of excl_set, because of DeMorgan's Law
            res_set = joiner_func(incl_set, excl_set)  # join

            res_lst = sorted(res_set)
            res_dataset = UIChunkedDataset({img_key: dataset[img_key] for img_key in res_lst}, chunk_size=args.chunk_size)

            # print(f"incl_set: {incl_set}")
            # print(f"excl_set: {excl_set}")
            # print(f"res_set: {res_set}")
            # print(f"res_dataset: {res_dataset}")

            search_range_size = len(cur_dataset)
            res = change_to_dataset(res_dataset)
            res.update({log_box: f"querying matches {len(res_dataset)} images over {search_range_size} images"})
            return res

        query_btn.click(
            fn=query_tag_table,
            inputs=[query_include_condition, query_include_tags, query_joiner_dropdown, query_exclude_condition, query_exclude_tags],
            outputs=selector_change_outputs,
            show_progress=True,
        )

    return demo
