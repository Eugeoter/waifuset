import gradio as gr
import random
import cv2
import numpy as np
import re
import os
import pandas
import inspect
from functools import wraps
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from typing import Callable, Any, Tuple, Dict, List, Union
from concurrent.futures import ThreadPoolExecutor, wait
from . import custom_components as cc
from ..import tagging, sorting
from .emoji import Emoji
from ..utils import log_utils as logu, image_utils as imgu


OPS = {
    'add': lambda x, y: y | x,
    'remove': lambda x, y: x - y,
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

FORMAT = {
    'space': lambda x: x.spaced(),
    'underline': lambda x: x.underlined(),
    'escape': lambda x: x.escaped(),
    'unescape': lambda x: x.unescaped(),
    'analytical': lambda x: x.formalized(),
    'standard': lambda x: x.deformalized(),
}

SORTING_METHODS: Dict[str, Callable] = {
    'stem': sorting.stem,
    'aesthetic_score': sorting.aesthetic_score,
    'quality': sorting.quality,
    'quality_or_score': sorting.quality_or_score,
    'perceptual_hash': sorting.perceptual_hash,
    'extension': sorting.extension,
    'original_size': sorting.original_size,
    'original_width': sorting.original_width,
    'original_height': sorting.original_height,
    'original_aspect_ratio': sorting.original_aspect_ratio,
    'caption_length': sorting.caption_length,
    'key': sorting.key,
    'has_gen_info': sorting.has_gen_info,
    'random': sorting.random,
}


def prepare_dataset(
    args,
):
    from .ui_dataset import UIDataset

    source = args.source
    if args.write_to_database:
        source = [args.database_file, source]

    dataset = UIDataset(
        source,
        formalize_caption=False,
        write_to_database=args.write_to_database,
        write_to_txt=args.write_to_txt,
        database_file=args.database_file,
        chunk_size=args.chunk_size,
        read_caption=True,
        verbose=True,
    )

    if args.change_source:
        old_img_src = args.old_source
        new_img_src = args.new_source

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
    global_args,
):
    # ========================================= UI ========================================= #

    from ..classes import Dataset, ImageInfo, Caption
    from ..classes.caption.caption import preprocess_tag
    from .ui_dataset import UIChunkedDataset, UISampleHistory, UITagPriorityManager, TagPriority, UITab
    from .utils import open_file_folder, translate

    def dataset_to_metadata_df(dset):
        num_images = len(dset)
        cats = sorted(list(set(img_info.category for img_info in dset.values()))) if len(dset) > 0 else []
        num_cats = len(cats)
        if num_cats > 5:
            cats = cats[:5] + ['...']
        df = pandas.DataFrame(
            data={
                translate('Number of images', global_args.language): [num_images],
                translate('Number of categories', global_args.language): [num_cats],
                translate('Categories', global_args.language): [cats],
            },
        )
        return df

    def import_priority_config():
        import json
        nonlocal tag_priority_manager
        with open(global_args.tag_priority_config_path, 'r') as f:
            config = json.load(f)
        for name, patterns in tagging.PRIORITY.items():
            if name in config:
                config[name].extend(patterns)
        tag_priority_manager = UITagPriorityManager(config)
        return f"priority config loaded"

    def init_everything():
        # args validation
        if global_args.language not in ['en', 'cn']:
            print(f"language {global_args.language} not supported, using `en` instead")
            global_args.language = 'en'
        global_args.max_workers = max(1, min(global_args.max_workers, os.cpu_count() - 1))
        global_args.chunk_size = max(1, global_args.chunk_size)
        if not global_args.write_to_database and not global_args.write_to_txt:
            print("neither `write_to_database` nor `write_to_txt` is True, nothing will be saved.")

        # import priority config
        tagging.init_priority_tags()
        tag_priority_manager = None

        try:
            import_priority_config()
            assert tag_priority_manager is not None
        except Exception as e:
            print(f"failed to load priority config: {e}")
            tag_priority_manager = UITagPriorityManager(tagging.PRIORITY)

        # init dataset
        main_dataset = prepare_dataset(global_args)
        cur_dataset = main_dataset
        buffer = main_dataset.buffer
        sample_history = UISampleHistory()
        tag_table = None
        tag_feature_table = {}

        waifu_tagger = None
        waifu_scorer = None

        return main_dataset, buffer, sample_history, tag_priority_manager, cur_dataset, waifu_tagger, waifu_scorer, tag_table, tag_feature_table

    main_dataset, buffer, sample_history, tag_priority_manager, cur_dataset, waifu_tagger, waifu_scorer, tag_table, tag_feature_table = init_everything()

    with open('./api.css', 'r', encoding='utf-8') as css_file:
        css = css_file.read()

    # demo
    with gr.Blocks(css=css) as demo:
        with gr.Tab(label=translate('Dataset', global_args.language)) as dataset_tab:
            with gr.Tab(translate('Main', global_args.language)) as main_tab:
                # ========================================= Base variables ========================================= #

                with gr.Row():
                    with gr.Column():
                        with gr.Tab(translate('Category', global_args.language)):
                            with gr.Row():
                                category_selector = gr.Dropdown(
                                    label=translate('Category', global_args.language),
                                    choices=main_dataset.categories,
                                    value=None,
                                    container=False,
                                    multiselect=True,
                                    allow_custom_value=False,
                                    min_width=256,
                                )
                                reload_category_btn = cc.EmojiButton(Emoji.anticlockwise)

                        with gr.Tab(translate('Sort', global_args.language)):
                            with gr.Row():
                                sorting_methods_dropdown = gr.Dropdown(
                                    label=translate('Sorting Methods', global_args.language),
                                    choices=translate(SORTING_METHODS.keys(), global_args.language),
                                    value=None,
                                    container=False,
                                    multiselect=True,
                                    allow_custom_value=False,
                                    min_width=256,
                                )
                                sorting_reverse_checkbox = gr.Checkbox(
                                    label=translate('Reverse', global_args.language),
                                    value=False,
                                    container=False,
                                    scale=0,
                                    min_width=128,
                                )

                        with gr.Tab(translate('Query', global_args.language)):
                            with gr.Row():
                                query_use_regex = gr.Checkbox(
                                    label=translate('Regex', global_args.language),
                                    value=False,
                                    container=False,
                                    scale=0,
                                    min_width=128,
                                )
                            with gr.Tab(translate("Tag", global_args.language)) as query_tag_tab:
                                with gr.Row(variant='compact'):
                                    query_refresh_tag_list_btn = cc.EmojiButton(Emoji.anticlockwise)
                                    query_unload_tag_list_btn = cc.EmojiButton(Emoji.no_entry)

                                with gr.Row(variant='compact'):
                                    query_include_condition = gr.Dropdown(
                                        label=translate('If', global_args.language),
                                        choices=translate(list(CONDITION.keys()), global_args.language),
                                        value=translate(list(CONDITION.keys())[0], global_args.language),
                                        multiselect=False,
                                        allow_custom_value=False,
                                        min_width=128,
                                        scale=0,
                                    )
                                    query_include_tags = gr.Dropdown(
                                        label=translate('Include', global_args.language),
                                        choices=None,
                                        allow_custom_value=True,
                                        multiselect=True,
                                    )
                                    query_joiner_dropdown = gr.Dropdown(
                                        label=translate('Joiner', global_args.language),
                                        choices=translate(list(JOINER.keys()), global_args.language),
                                        value=translate(list(JOINER.keys())[0], global_args.language),
                                        multiselect=False,
                                        allow_custom_value=False,
                                        min_width=108,
                                        scale=0,
                                    )

                                with gr.Row(variant='compact'):
                                    query_exclude_condition = gr.Dropdown(
                                        label=translate('If', global_args.language),
                                        choices=translate(list(CONDITION.keys()), global_args.language),
                                        value=translate(list(CONDITION.keys())[0], global_args.language),
                                        multiselect=False,
                                        allow_custom_value=False,
                                        min_width=128,
                                        scale=0,
                                    )
                                    query_exclude_tags = gr.Dropdown(
                                        label=translate('Exclude', global_args.language),
                                        choices=None,
                                        allow_custom_value=True,
                                        multiselect=True,
                                    )
                                    query_tab_table_btn = cc.EmojiButton(Emoji.right_pointing_magnifying_glass, variant='primary', min_width=100)

                            with gr.Tab(translate("Filename", global_args.language)) as query_filename_tab:
                                with gr.Row(variant='compact'):
                                    query_filename_selector = gr.Dropdown(
                                        label=translate('Include', global_args.language),
                                        choices=None,
                                        allow_custom_value=True,
                                        multiselect=True,
                                    )
                                    query_filename_btn = cc.EmojiButton(Emoji.right_pointing_magnifying_glass, variant='primary', min_width=100)

                            with gr.Tab(translate("Aesthetic Score", global_args.language)) as query_aes_score_tab:
                                with gr.Row(variant='compact'):
                                    query_min_aes_score = gr.Number(
                                        label=translate('Min', global_args.language),
                                        value=0,
                                        min_width=128,
                                        precision=4,
                                        scale=0,
                                    )
                                    query_max_aes_score = gr.Number(
                                        label=translate('Max', global_args.language),
                                        value=10,
                                        min_width=128,
                                        precision=4,
                                        scale=0,
                                    )
                                    query_aes_score_btn = cc.EmojiButton(Emoji.right_pointing_magnifying_glass, variant='primary', min_width=100)

                        with gr.Tab(translate('Source', global_args.language)):
                            with gr.Row():
                                source_file = gr.Textbox(
                                    value=os.path.abspath(global_args.source) if global_args.source else None,
                                    container=False,
                                    interactive=True,
                                    placeholder='Path/to/source',
                                    scale=1,
                                    min_width=128,
                                )
                                load_source_btn = cc.EmojiButton(Emoji.anticlockwise)

                    with gr.Column():
                        with gr.Row():
                            log_box = gr.TextArea(
                                label=translate('Log', global_args.language),
                                lines=1,
                                max_lines=1,
                            )

                with gr.Tab(translate('Dataset', global_args.language)) as tagging_tab:
                    with gr.Row():
                        with gr.Column():
                            with gr.Row():
                                showcase = gr.Gallery(
                                    label=translate('Showcase', global_args.language),
                                    value=[(v.image_path, k) for k, v in cur_dataset.chunk(0).items()],
                                    rows=4,
                                    columns=4,
                                    container=True,
                                    object_fit='scale-down',
                                    height=512,
                                )
                            with gr.Row():
                                load_pre_chunk_btn = cc.EmojiButton(Emoji.black_left_pointing_double_triangle, scale=1)
                                load_pre_hist_btn = cc.EmojiButton(Emoji.black_left_pointing_triangle, scale=1)
                                reload_subset_btn = cc.EmojiButton(Emoji.anticlockwise)
                                remove_image_btn = cc.EmojiButton(Emoji.trash_bin, variant='stop')
                                load_next_hist_btn = cc.EmojiButton(Emoji.black_right_pointing_triangle, scale=1)
                                load_next_chunk_btn = cc.EmojiButton(Emoji.black_right_pointing_double_triangle, scale=1)
                            with gr.Row():
                                with gr.Column():
                                    ...
                                with gr.Column():
                                    cur_chunk_index = gr.Number(label=f"{translate('Chunk', global_args.language)} {1}/{cur_dataset.num_chunks}", value=1, min_width=128, precision=0, scale=0)
                                with gr.Column():
                                    ...
                            with gr.Row():
                                dataset_metadata_df = gr.Dataframe(
                                    value=dataset_to_metadata_df(main_dataset),
                                    label=translate('Dataset Information', global_args.language),
                                    type='pandas',
                                    row_count=(1, 'fixed'),
                                )

                        with gr.Column():
                            with gr.Row():
                                with gr.Tab(translate('Caption', global_args.language)) as caption_tab:
                                    caption = gr.Textbox(
                                        show_label=False,
                                        value=None,
                                        container=True,
                                        show_copy_button=True,
                                        lines=6,
                                        max_lines=6,
                                        placeholder='empty',
                                    )
                                with gr.Tab(translate('Metadata', global_args.language)) as metadata_tab:
                                    with gr.Row():
                                        caption_metadata_df = gr.Dataframe(
                                            label=translate("Caption", global_args.language),
                                            value=None,
                                            type='pandas',
                                            row_count=(1, 'fixed'),
                                        )
                                    with gr.Row():
                                        other_metadata_df = gr.Dataframe(
                                            label=translate("Other", global_args.language),
                                            value=None,
                                            type='pandas',
                                            row_count=(1, 'fixed'),
                                        )
                                with gr.Tab(translate('Generation Information', global_args.language)) as gen_info_tab:
                                    with gr.Tab(label=translate('Positive Prompt', global_args.language)):
                                        positive_prompt = gr.Textbox(
                                            label=translate('Positive Prompt', global_args.language),
                                            value=None,
                                            container=False,
                                            show_copy_button=True,
                                            lines=6,
                                            max_lines=6,
                                            interactive=True,
                                            placeholder='empty',
                                        )
                                    with gr.Tab(label=translate('Negative Prompt', global_args.language)):
                                        negative_prompt = gr.Textbox(
                                            label=translate('Negative Prompt', global_args.language),
                                            value=None,
                                            container=False,
                                            show_copy_button=True,
                                            lines=6,
                                            max_lines=6,
                                            interactive=True,
                                            placeholder='empty',
                                        )
                                    with gr.Tab(label=translate('Generation Parameters', global_args.language)):
                                        gen_params_df = gr.Dataframe(
                                            value=None,
                                            show_label=False,
                                            type='pandas',
                                            row_count=(1, 'fixed'),
                                            interactive=True,
                                        )

                            with gr.Row():
                                random_btn = cc.EmojiButton(Emoji.dice)
                                set_category_btn = cc.EmojiButton(Emoji.top_left_arrow)
                                undo_btn = cc.EmojiButton(Emoji.leftwards)
                                redo_btn = cc.EmojiButton(Emoji.rightwards)
                                save_btn = cc.EmojiButton(Emoji.floppy_disk, variant='primary', visible=global_args.write_to_database or global_args.write_to_txt)
                                cancel_btn = cc.EmojiButton(Emoji.no_entry, variant='stop', visible=global_args.max_workers == 1)
                            with gr.Row():
                                proc_opts = gr.CheckboxGroup(
                                    label=translate('Process options', global_args.language),
                                    choices=translate(['Batch', 'Append', 'Regex'], global_args.language),
                                    value=None,
                                    container=False,
                                    scale=1,
                                )

                            with gr.Tab(translate('Quick Tagging', global_args.language)):
                                with gr.Row(variant='compact'):
                                    tagging_best_quality_btn = cc.EmojiButton(Emoji.love_emotion, variant='primary', scale=1)
                                    tagging_high_quality_btn = cc.EmojiButton(Emoji.heart, scale=1)
                                    tagging_low_quality_btn = cc.EmojiButton(Emoji.broken_heart, scale=1)
                                    tagging_worst_quality_btn = cc.EmojiButton(Emoji.hate_emotion, variant='stop', scale=1)
                                with gr.Row(variant='compact'):
                                    tagging_color_btn = cc.EmojiButton(value=translate('Color', global_args.language), scale=1, variant='primary')
                                    tagging_detailed_btn = cc.EmojiButton(value=translate('Detail', global_args.language), scale=1, variant='primary')
                                    tagging_lowres_btn = cc.EmojiButton(value=translate('Lowres', global_args.language), scale=1, variant='stop')
                                    tagging_messy_btn = cc.EmojiButton(value=translate('Messy', global_args.language), scale=1, variant='stop')
                                with gr.Row(variant='compact'):
                                    tagging_amazing_quality_btn = cc.EmojiButton(value=translate('Amazing', global_args.language), scale=1, variant='primary')
                                    tagging_aesthetic_btn = cc.EmojiButton(value=translate('Aesthetic', global_args.language), scale=1, variant='primary')
                                    tagging_beautiful_btn = cc.EmojiButton(value=translate('Beautiful', global_args.language), scale=1, variant='primary')
                                    tagging_y_btn = cc.EmojiButton(value='Y', scale=1, variant='stop', visible=False)

                            with gr.Tab(label=translate('Custom Tagging', global_args.language)):
                                with gr.Tab(translate("Add/Remove", global_args.language)) as add_remove_tab:
                                    def custom_add_rem_tagging_row():
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
                                        add_tag_btn, tag_selector, remove_tag_btn = custom_add_rem_tagging_row()
                                        add_tag_btns.append(add_tag_btn)
                                        tag_selectors.append(tag_selector)
                                        remove_tag_btns.append(remove_tag_btn)

                                    with gr.Accordion(label=translate('More', global_args.language), open=False):
                                        for r in range(6):
                                            add_tag_btn, tag_selector, remove_tag_btn = custom_add_rem_tagging_row()
                                            add_tag_btns.append(add_tag_btn)
                                            tag_selectors.append(tag_selector)
                                            remove_tag_btns.append(remove_tag_btn)

                                with gr.Tab(translate("Replace", global_args.language)) as replace_tab:
                                    def custom_replace_tagging_row():
                                        with gr.Row(variant='compact'):
                                            replace_tag_btn = cc.EmojiButton(Emoji.clockwise_downwards_and_upwards_open_circle_arrows, variant='primary')
                                            old_tag_selector = gr.Dropdown(
                                                # label=translate('Replacer', global_args.language),
                                                choices=list(tagging.CUSTOM_TAGS),
                                                value=None,
                                                container=False,
                                                multiselect=False,
                                                allow_custom_value=True,
                                                min_width=96,
                                            )
                                            new_tag_selector = gr.Dropdown(
                                                # label=translate('Replacement', global_args.language),
                                                choices=list(tagging.CUSTOM_TAGS),
                                                value=None,
                                                container=False,
                                                multiselect=False,
                                                allow_custom_value=True,
                                                min_width=96,
                                            )
                                        return replace_tag_btn, old_tag_selector, new_tag_selector

                                    replace_tag_btns = []
                                    old_tag_selectors = []
                                    new_tag_selectors = []
                                    for r in range(3):
                                        replace_tag_btn, old_tag_selector, new_tag_selector = custom_replace_tagging_row()
                                        replace_tag_btns.append(replace_tag_btn)
                                        old_tag_selectors.append(old_tag_selector)
                                        new_tag_selectors.append(new_tag_selector)

                                    with gr.Accordion(label=translate('More', global_args.language), open=False):
                                        for r in range(6):
                                            replace_tag_btn, old_tag_selector, new_tag_selector = custom_replace_tagging_row()
                                            replace_tag_btns.append(replace_tag_btn)
                                            old_tag_selectors.append(old_tag_selector)
                                            new_tag_selectors.append(new_tag_selector)

                            # ! Deprecated
                            # with gr.Tab(label=translate('Operational Tagging', global_args.language)):
                            #     with gr.Row(variant='compact'):
                            #         cap_op_op_dropdown = gr.Dropdown(
                            #             label=translate('Op', global_args.language),
                            #             choices=translate(list(OPS.keys()), global_args.language),
                            #             value=translate(list(OPS.keys())[0], global_args.language),
                            #             multiselect=False,
                            #             allow_custom_value=False,
                            #             scale=0,
                            #             min_width=128,
                            #         )
                            #         cap_op_op_tag_dropdown = gr.Dropdown(
                            #             label=translate('Tags', global_args.language),
                            #             choices=[],
                            #             value=None,
                            #             allow_custom_value=True,
                            #             multiselect=True,
                            #         )

                            #         operate_caption_btn = cc.EmojiButton(Emoji.black_right_pointing_triangle, variant='primary')

                            #     with gr.Row(variant='compact'):
                            #         cap_op_cond_dropdown = gr.Dropdown(
                            #             label=translate('If', global_args.language),
                            #             choices=translate(list(CONDITION.keys()), global_args.language),
                            #             value=translate(list(CONDITION.keys())[0], global_args.language),
                            #             multiselect=False,
                            #             allow_custom_value=False,
                            #             scale=0,
                            #             min_width=128,
                            #         )

                            #         cap_op_cond_tag_dropdown = gr.Dropdown(
                            #             label=translate('Tags', global_args.language),
                            #             choices=[],
                            #             value=None,
                            #             allow_custom_value=True,
                            #             multiselect=True,
                            #         )

                            #         cap_op_incl_rel_dropdown = gr.Dropdown(
                            #             label=translate('Inclusion', global_args.language),
                            #             choices=translate(list(INCLUSION_RELATIONSHIP.keys()), global_args.language),
                            #             value=translate(list(INCLUSION_RELATIONSHIP.keys())[0], global_args.language),
                            #             multiselect=False,
                            #             allow_custom_value=False,
                            #             scale=0,
                            #             min_width=144,
                            #         )

                            with gr.Tab(label=translate('Optimizers', global_args.language)):
                                with gr.Tab(translate('Sort', global_args.language)):
                                    with gr.Row(variant='compact'):
                                        sort_caption_btn = cc.EmojiButton(Emoji.black_right_pointing_triangle, min_width=40, variant='primary')
                                with gr.Tab(translate('Deduplicate', global_args.language)):
                                    with gr.Row(variant='compact'):
                                        deduplicate_caption_btn = cc.EmojiButton(Emoji.black_right_pointing_triangle, min_width=40, variant='primary')
                                with gr.Tab(translate('Deoverlap', global_args.language)):
                                    with gr.Row(variant='compact'):
                                        deoverlap_caption_btn = cc.EmojiButton(Emoji.black_right_pointing_triangle, min_width=40, variant='primary')
                                with gr.Tab(translate('Defeature', global_args.language)):
                                    with gr.Row(variant='compact'):
                                        defeature_caption_btn = cc.EmojiButton(Emoji.black_right_pointing_triangle, min_width=40, variant='primary')
                                    with gr.Row(variant='compact'):
                                        defeature_freq_thres = gr.Slider(
                                            label=translate('Frequency Threshold', global_args.language),
                                            value=0.3,
                                            minimum=0,
                                            maximum=1,
                                            step=0.01,
                                        )
                                        defeature_count_thres = gr.Slider(
                                            label=translate('Counting Threshold', global_args.language),
                                            value=1,
                                            minimum=1,
                                            maximum=1000,
                                            step=1,
                                        )
                                        defeature_least_sample_size = gr.Slider(
                                            label=translate('Least Sample Size', global_args.language),
                                            value=1,
                                            minimum=1,
                                            maximum=1000,
                                            step=1,
                                        )
                                with gr.Tab(translate('Formalize', global_args.language)):
                                    with gr.Row(variant='compact'):
                                        formalize_caption_btn = cc.EmojiButton(Emoji.black_right_pointing_triangle, scale=0, min_width=40, variant='primary')
                                    with gr.Row(variant='compact'):
                                        formalize_caption_dropdown = gr.Dropdown(
                                            label=translate('Format', global_args.language),
                                            choices=translate(list(FORMAT.keys()), global_args.language),
                                            value=None,
                                            multiselect=True,
                                            allow_custom_value=False,
                                            scale=1,
                                        )

                            with gr.Tab(label=translate('Tools', global_args.language)):

                                with gr.Tab(label=translate('Tagger', global_args.language)):
                                    with gr.Row(variant='compact'):
                                        wd14_run_btn = cc.EmojiButton(Emoji.black_right_pointing_triangle, variant='primary', min_width=40)
                                    with gr.Row(variant='compact'):
                                        wd14_general_threshold = gr.Slider(
                                            label=translate('General Threshold', global_args.language),
                                            value=0.35,
                                            minimum=0,
                                            maximum=1,
                                            step=0.01,
                                        )
                                        wd14_character_threshold = gr.Slider(
                                            label=translate('Character Threshold', global_args.language),
                                            value=0.35,
                                            minimum=0,
                                            maximum=1,
                                            step=0.01,
                                        )
                                    with gr.Row(variant='compact'):
                                        wd14_caption_proc_mode = gr.Radio(
                                            label=translate('Overwrite mode', global_args.language),
                                            choices=translate(['overwrite', 'append', 'prepend', 'ignore'], global_args.language),
                                            value=translate('overwrite', global_args.language),
                                            scale=1,
                                            min_width=128,
                                        )

                                with gr.Tab(label=translate('Scorer', global_args.language)):
                                    with gr.Row(variant='compact'):
                                        predict_aesthetic_score_btn = cc.EmojiButton(Emoji.black_right_pointing_triangle, variant='primary', min_width=40)
                                        label_aesthetic_btn = cc.EmojiButton(Emoji.label, min_width=40)
                                        clean_aesthetic_score_btn = cc.EmojiButton(Emoji.no_entry, variant='stop', min_width=40)
                                    with gr.Row():
                                        waifu_scorer_os_mode = gr.Radio(
                                            label=translate('Overwrite mode', global_args.language),
                                            choices=translate(['overwrite', 'ignore'], global_args.language),
                                            value=translate('overwrite', global_args.language),
                                            scale=1,
                                            min_width=128,
                                        )

                                with gr.Tab(label=translate('Hasher', global_args.language)):
                                    with gr.Row(variant='compact'):
                                        get_perceptual_hash_btn = cc.EmojiButton(Emoji.black_right_pointing_triangle, variant='primary', min_width=40)
                                        clean_perceptual_hash_btn = cc.EmojiButton(Emoji.no_entry, variant='stop', min_width=40)
                                    with gr.Row():
                                        hasher_os_mode = gr.Radio(
                                            label=translate('Overwrite mode', global_args.language),
                                            choices=translate(['overwrite', 'ignore'], global_args.language),
                                            value=translate('overwrite', global_args.language),
                                            scale=1,
                                            min_width=128,
                                        )

                    with gr.Row():
                        with gr.Column(scale=4):
                            with gr.Tab(translate('Image Key', global_args.language)):
                                with gr.Row(variant='compact'):
                                    cur_image_key = gr.Textbox(value=None, show_label=False, max_lines=1, lines=1)

                            with gr.Tab(translate('Image Path', global_args.language)):
                                with gr.Row(variant='compact'):
                                    image_path = gr.Textbox(
                                        value=None,
                                        container=False,
                                        show_label=False,
                                        max_lines=2,
                                        lines=1,
                                        show_copy_button=True,
                                        interactive=False,
                                    )
                                    open_folder_btn = cc.EmojiButton(Emoji.open_file_folder)
                            with gr.Tab(translate('Resolution', global_args.language)):
                                with gr.Row(variant='compact'):
                                    resolution = gr.Textbox(
                                        value=None,
                                        container=False,
                                        show_label=False,
                                        max_lines=2,
                                        lines=1,
                                        show_copy_button=True,
                                        interactive=False,
                                    )

                        with gr.Column(scale=4):
                            ...

                with gr.Tab(translate('Database', global_args.language)) as database_tab:
                    database = gr.Dataframe(
                        value=None,
                        label=translate('Database', global_args.language),
                        type='pandas',
                    )

                with gr.Tab(translate('Buffer', global_args.language)) as buffer_tab:
                    buffer_df = gr.Dataframe(
                        value=None,
                        label=translate('Buffer', global_args.language),
                        type='pandas',
                        row_count=(20, 'fixed'),
                    )

            with gr.Tab(translate('Config', global_args.language)) as config_tab:
                with gr.Row():
                    cfg_log_box = gr.TextArea(
                        label=translate('Log', global_args.language),
                        lines=1,
                        max_lines=1,
                    )

                with gr.Tab(translate('Tag Order', global_args.language)):
                    with gr.Row():
                        export_tag_priority_config_btn = cc.EmojiButton(Emoji.floppy_disk)
                        import_priority_config_btn = cc.EmojiButton(Emoji.inbox_tray)

                    tag_priority_comps = []
                    for i, name in enumerate(tag_priority_manager.keys()):
                        with gr.Row(variant='compact'):
                            tag_priority_level = gr.Number(label=translate(name.replace('_', ' ').title(), global_args.language), value=i, precision=0, scale=0, min_width=144)
                            tag_priority_custom_tags = gr.Dropdown(
                                label=translate('Tags', global_args.language),
                                choices=None,
                                value=None,
                                multiselect=True,
                                allow_custom_value=True,
                                scale=1,
                            )
                            tag_priority_comps.append((tag_priority_level, tag_priority_custom_tags))

            # ========================================= Functions ========================================= #

            def kwargs_setter(func, **preset_kwargs):
                @wraps(func)
                def wrapper(*args, **kwargs):
                    return func(*args, **kwargs, **preset_kwargs)
                return wrapper

            def dataset_to_gallery(dset):
                return [(v.image_path, k) for k, v in dset.items() if v.image_path.is_file()]

            def get_new_img_key(dset):
                r"""
                Get the image key that should be selected if the showing dataset is changed by `dset`. Doesn't depend on what the current dataset is.
                """
                pre_idx = main_dataset.selected.index
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

            def change_current_dataset(dset, sorting_methods=None, reverse=False):
                r"""
                Change the current dataset to `dset`
                """
                nonlocal cur_dataset

                # pre-sorting
                if sorting_methods is not None and len(sorting_methods) > 0:
                    if global_args.language != 'en':
                        sorting_methods = translate(sorting_methods, 'en')
                    sorting_methods = [method.replace(' ', '_') for method in sorting_methods]

                    extra_kwargs = {}
                    selected_img_key = main_dataset.selected.image_key
                    if selected_img_key is not None:
                        target = main_dataset[selected_img_key]
                        extra_kwargs['target'] = target.perceptual_hash

                    sorting_keys = []
                    for method in sorting_methods:
                        func = SORTING_METHODS[method]
                        func_params = inspect.signature(func).parameters
                        func_param_names = list(func_params.keys())
                        sorting_keys.append((func, {k: v for k, v in extra_kwargs.items() if k in func_param_names}))

                cur_dataset = dset

                # post-sorting
                if sorting_methods is not None and len(sorting_methods) > 0:
                    cur_dataset.sort(key=lambda item: tuple(func(item[1], **kwargs) for func, kwargs in sorting_keys), reverse=reverse)

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
                    dataset_metadata_df: dataset_to_metadata_df(dset),
                    cur_image_key: new_img_key,
                    cur_chunk_index: gr.update(value=new_chunk_index, label=f"{translate('Chunk', global_args.language)} {new_chunk_index}/{cur_dataset.num_chunks}")
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
                    cur_chunk_index: gr.update(value=new_chunk_index, label=f"{translate('Chunk', global_args.language)} {new_chunk_index}/{dset.num_chunks}"),
                }

            # ========================================= Tab changing parser ========================================= #
            ui_main_tab = UITab(tagging_tab)
            ui_data_tab = UITab(caption_tab)

            def change_activating_tab(ui_tab, new_tab, func=None):
                r"""
                Change the activating tab name to `tab_name`
                """
                if func is None:  # direct
                    ui_tab.tab = new_tab
                else:  # wrapper
                    @wraps(func)
                    def wrapper(*args, **kwargs):
                        change_activating_tab(ui_tab, new_tab)
                        return func(*args, **kwargs)
                    return wrapper

            # ========================================= Change Source ========================================= #

            def change_source(source):
                if source is None or source == '':
                    return {log_box: 'empty source'}
                source = os.path.abspath(source)
                if not os.path.exists(source):
                    return {log_box: f"source `{source}` not found"}
                elif not (os.path.isdir(source) or source.endswith(('.csv', '.json'))):
                    return {f"source `{source}` is not a file or directory"}

                nonlocal main_dataset, buffer, sample_history, cur_dataset, tag_table, tag_feature_table

                global_args.source = source
                main_dataset, buffer, sample_history, _, cur_dataset, _, _, tag_table, tag_feature_table = init_everything()
                return {
                    source_file: global_args.source,
                    category_selector: gr.update(choices=main_dataset.categories, value=None),
                    showcase: [(v.image_path, k) for k, v in cur_dataset.chunk(0).items()],
                    cur_chunk_index: gr.update(value=1, label=f"{translate('Chunk', global_args.language)} 1/{cur_dataset.num_chunks}"),
                    query_include_tags: gr.update(choices=None),
                    query_exclude_tags: gr.update(choices=None),
                    log_box: f"source reloaded: `{source}`",
                }

            load_source_btn.click(
                fn=change_source,
                inputs=[source_file],
                outputs=[source_file, category_selector, showcase, cur_chunk_index, query_include_tags, query_exclude_tags, log_box],
                concurrency_limit=1,
            )

            # ========================================= Subset key selector ========================================= #

            def change_to_dataset(dset: UIChunkedDataset = None, new_chunk_index=1, sorting_methods=None, reverse=False):
                r"""
                Change current dataset to another dataset `dset` and show its chunk
                """
                if dset is None:
                    dset = cur_dataset
                change_current_dataset(dset, sorting_methods=sorting_methods, reverse=reverse)
                if ui_main_tab.tab is tagging_tab:
                    res = show_dataset(dset, new_chunk_index)
                elif ui_main_tab.tab is database_tab:
                    res = show_database(dset, new_chunk_index)
                return res

            def change_to_categories(categories, sorting_methods=None, reverse=False, progress=gr.Progress(track_tqdm=True)):
                r"""
                Change current dataset to another dataset with category `category` and show its chunk
                """
                if categories is None or len(categories) == 0:
                    dset = main_dataset
                else:
                    dset = UIChunkedDataset(chunk_size=global_args.chunk_size)
                    for category in tqdm(sorted(categories), desc='loading subset'):
                        if category not in main_dataset.categories:
                            logu.warn(f"missing category `{category}`")
                            continue
                        dset.update(main_dataset.make_subset(condition=lambda img_info: img_info.category == category, chunk_size=global_args.chunk_size))
                return change_to_dataset(dset, new_chunk_index=1, sorting_methods=sorting_methods, reverse=reverse)

            dataset_change_inputs = [cur_chunk_index, sorting_methods_dropdown, sorting_reverse_checkbox]
            dataset_change_listeners = [showcase, dataset_metadata_df, cur_image_key, database, cur_chunk_index, category_selector, log_box]

            # category_selector.blur(
            #     fn=change_to_category,
            #     inputs=[category_selector],
            #     outputs=selector_change_outputs,
            #     show_progress=True,
            #     trigger_mode='multiple',
            #     concurrency_limit=1,
            # )

            reload_category_btn.click(
                fn=change_to_categories,
                inputs=[category_selector, sorting_methods_dropdown, sorting_reverse_checkbox],
                outputs=dataset_change_listeners,
                trigger_mode='multiple',
                concurrency_limit=1,
                scroll_to_output=True,
            )

            set_category_btn.click(
                fn=lambda image_key, *args: {**change_to_categories([cur_dataset[image_key].category], *args), category_selector: [cur_dataset[image_key].category]},
                inputs=[cur_image_key, sorting_methods_dropdown, sorting_reverse_checkbox],
                outputs=dataset_change_listeners,
                show_progress=True,
                trigger_mode='always_last',
                concurrency_limit=1,
            )

            reload_subset_btn.click(  # same as above
                fn=lambda *args: change_to_dataset(cur_dataset, *args),
                inputs=dataset_change_inputs,
                outputs=dataset_change_listeners,
                show_progress=True,
                concurrency_limit=1,
            )

            tagging_tab.select(
                fn=change_activating_tab(ui_main_tab, tagging_tab, lambda *args: change_to_dataset(cur_dataset, *args)),
                inputs=dataset_change_inputs,
                outputs=dataset_change_listeners,
                show_progress=True,
                concurrency_limit=1,
            )

            database_tab.select(
                fn=change_activating_tab(ui_main_tab, database_tab, lambda *args: change_to_dataset(cur_dataset, *args)),
                inputs=dataset_change_inputs,
                outputs=dataset_change_listeners,
                show_progress=True,
                concurrency_limit=1,
            )

            cur_chunk_index.submit(
                fn=lambda chunk_index: change_to_dataset(cur_dataset, chunk_index),
                inputs=[cur_chunk_index],  # no need to sort
                outputs=dataset_change_listeners,
                show_progress=True,
                concurrency_limit=1,
            )

            load_pre_chunk_btn.click(
                fn=lambda chunk_index: change_to_dataset(cur_dataset, chunk_index - 1),
                inputs=[cur_chunk_index],  # no need to sort
                outputs=dataset_change_listeners,
                show_progress=True,
                concurrency_limit=1,
            )

            load_next_chunk_btn.click(
                fn=lambda chunk_index: change_to_dataset(cur_dataset, chunk_index + 1),
                inputs=[cur_chunk_index],  # no need to sort
                outputs=dataset_change_listeners,
                show_progress=True,
                concurrency_limit=1,
            )

            # ========================================= Showcase ========================================= #

            def select_image_key(selected: gr.SelectData):
                if selected is None:
                    return None, None
                image_key = main_dataset.select(selected)
                return image_key

            showcase.select(
                fn=select_image_key,
                outputs=[cur_image_key],
                concurrency_limit=1,
            )

            cur_image_key_change_listeners = [image_path, resolution, caption, caption_metadata_df, other_metadata_df, positive_prompt, negative_prompt, gen_params_df, log_box]
            CAPTION_MD_KEYS = tuple(ImageInfo._caption_attrs)
            OTHER_MD_KEYS = ('aesthetic_score', 'perceptual_hash')

            def get_caption(image_key):
                if image_key is None or image_key == '':
                    return None
                img_info = main_dataset[image_key]
                if img_info.caption is None:
                    return None
                return str(img_info.caption)

            def get_metadata_df(image_key, keys):
                if global_args.render_mode == 'partial' and ui_data_tab.tab is not metadata_tab:
                    return None
                if image_key is None or image_key == '' or image_key not in main_dataset:
                    return None
                image_info: ImageInfo = main_dataset[image_key]
                info_dict = image_info.dict()
                data = [{translate(key.replace('_', ' ').title(), global_args.language): info_dict.get(key, None) for key in keys}]
                df = pandas.DataFrame(data=data, columns=data[0].keys())
                return df

            def get_gen_info(image_key):
                image_info = main_dataset[image_key]
                metadata_dict = image_info.gen_info
                pos_pmt = metadata_dict.pop('Positive prompt', None)
                neg_pmt = metadata_dict.pop('Negative prompt', None)
                # single row pandas dataframe for params
                params_df = pandas.DataFrame(data=[metadata_dict], columns=list(metadata_dict.keys())) if len(metadata_dict) > 0 else None
                return pos_pmt, neg_pmt, params_df

            def track_image_key(img_key):
                if img_key is None or img_key == '':  # no image key selected
                    return {k: None for k in cur_image_key_change_listeners}

                if img_key != main_dataset.selected.image_key:  # fix
                    main_dataset.select((main_dataset.selected.index, img_key))

                img_info = main_dataset.get(img_key)
                img_path = img_info.image_path
                if not img_path.is_file():
                    return {log_box: f"image `{img_path}` not found"}

                reso = f"{img_info.original_size[0]}x{img_info.original_size[1]}"

                if global_args.render_mode == 'full':
                    pos_pmt, neg_pmt, param_df = get_gen_info(img_key)
                    res = {
                        image_path: img_path,
                        resolution: reso,
                        caption: get_caption(img_key),
                        caption_metadata_df: get_metadata_df(img_key, keys=CAPTION_MD_KEYS),
                        other_metadata_df: get_metadata_df(img_key, keys=OTHER_MD_KEYS),
                        positive_prompt: pos_pmt,
                        negative_prompt: neg_pmt,
                        gen_params_df: param_df,
                    }
                else:
                    if ui_data_tab.tab is caption_tab:
                        res = {
                            caption: get_caption(img_key),
                        }
                    elif ui_data_tab.tab is metadata_tab:
                        res = {
                            caption_metadata_df: get_metadata_df(img_key, keys=CAPTION_MD_KEYS),
                            other_metadata_df: get_metadata_df(img_key, keys=OTHER_MD_KEYS),
                        }
                    elif ui_data_tab.tab is gen_info_tab:
                        pos_pmt, neg_pmt, param_df = get_gen_info(img_key)
                        res = {
                            positive_prompt: pos_pmt,
                            negative_prompt: neg_pmt,
                            gen_params_df: param_df,
                        }

                return {
                    image_path: img_path,
                    resolution: reso,
                    **res,
                }

            cur_image_key.change(
                fn=track_image_key,
                inputs=[cur_image_key],
                outputs=cur_image_key_change_listeners,
                concurrency_limit=1,
            )

            if global_args.render_mode == 'partial':
                caption_tab.select(
                    fn=change_activating_tab(ui_data_tab, caption_tab, get_caption),
                    inputs=[cur_image_key],
                    outputs=[caption],
                    concurrency_limit=1,
                )

                metadata_tab.select(
                    fn=lambda image_key: (
                        change_activating_tab(ui_data_tab, metadata_tab, get_metadata_df)(image_key, keys=CAPTION_MD_KEYS),
                        get_metadata_df(image_key, keys=OTHER_MD_KEYS)
                    ),
                    inputs=[cur_image_key],
                    outputs=[caption_metadata_df, other_metadata_df],
                    trigger_mode='always_last',
                    concurrency_limit=1,
                )

                gen_info_tab.select(
                    fn=change_activating_tab(ui_data_tab, gen_info_tab, get_gen_info),
                    inputs=[cur_image_key],
                    outputs=[positive_prompt, negative_prompt, gen_params_df],
                    concurrency_limit=1,
                )

            caption.change(
                fn=lambda image_key: get_metadata_df(image_key, keys=CAPTION_MD_KEYS),
                inputs=[cur_image_key],
                outputs=[caption_metadata_df],
                trigger_mode='always_last',
                concurrency_limit=1,
            )

            # ========================================= Below showcase ========================================= #

            def remove_image(image_key, chunk_index):
                if image_key is None or image_key == '':
                    return {log_box: f"empty image key"}
                main_dataset.remove(image_key)
                if cur_dataset is not main_dataset and image_key in cur_dataset:
                    del cur_dataset[image_key]  # remove from current dataset
                return change_to_dataset(new_chunk_index=chunk_index)

            remove_image_btn.click(
                fn=remove_image,
                inputs=[cur_image_key, cur_chunk_index],
                outputs=dataset_change_listeners,
                concurrency_limit=1,
            )

            def show_i_th_sample(index):
                if len(sample_history) == 0:
                    return {
                        log_box: f"empty sample history",
                    }
                new_img_key = sample_history.select(index)
                return {
                    showcase: dataset_to_gallery(Dataset(main_dataset[new_img_key])),
                    cur_image_key: new_img_key,
                }

            load_pre_hist_btn.click(
                fn=lambda: show_i_th_sample(sample_history.index - 1) if sample_history.index is not None else {log_box: f"empty sample history"},
                inputs=[],
                outputs=dataset_change_listeners,
                concurrency_limit=1,
            )

            load_next_hist_btn.click(
                fn=lambda: show_i_th_sample(sample_history.index + 1) if sample_history.index is not None else {log_box: f"empty sample history"},
                inputs=[],
                outputs=dataset_change_listeners,
                concurrency_limit=1,
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
                    showcase: dataset_to_gallery(Dataset(main_dataset[new_img_key])),
                    cur_image_key: new_img_key,
                    cur_chunk_index: gr.update(value=1, label=f"{translate('Chunk', global_args.language)} 1/{cur_dataset.num_chunks}"),
                }

            random_btn.click(
                fn=random_sample,
                inputs=[],
                outputs=dataset_change_listeners,
                concurrency_limit=1,
            )

            def data_edition_handler(func: Callable[[ImageInfo, Tuple[Any, ...], Dict[str, Any]], Caption]) -> Tuple[str, str]:
                proc_func_name = func.__name__
                max_workers = global_args.max_workers

                def wrapper(image_key, proc_opts, *args, progress: gr.Progress = gr.Progress(track_tqdm=True), **kwargs):
                    nonlocal proc_func_name, max_workers
                    proc_func_log_name = proc_func_name.replace('_', ' ')

                    if global_args.language != 'en':
                        proc_opts = translate(proc_opts, 'en')
                    proc_opts = [pm.lower() for pm in proc_opts]

                    batch = ('batch' in proc_opts) and (func is not write_caption)
                    extra_kwargs = dict(
                        append='append' in proc_opts,
                        regex='regex' in proc_opts,
                    )
                    # filter out extra kwargs
                    func_params = inspect.signature(func).parameters
                    func_param_names = list(func_params.keys())
                    extra_kwargs = {k: v for k, v in extra_kwargs.items() if k in func_param_names}

                    if image_key is None or image_key == '':
                        if not batch:
                            return {log_box: f"{proc_func_log_name}: empty image key"}
                    else:
                        image_key = Path(image_key).stem

                    # print(f"proc_mode: {proc_mode} | func_param_names: {func_param_names} | extra_kwargs: {extra_kwargs}")

                    def edit(image_info, *args, **kwargs):
                        if not image_info.image_path.is_file():
                            return None
                        new_img_info = func(image_info.copy(), *args, **extra_kwargs, **kwargs)
                        if image_info == new_img_info:
                            return None
                        return new_img_info

                    results = []
                    if batch:
                        subset = cur_dataset
                        pbar = tqdm(total=len(subset), desc=f'{proc_func_log_name} batch processing')
                        edit = logu.track_tqdm(pbar)(edit)
                        if max_workers == 1:
                            for image_info in subset.values():
                                results.append(edit(image_info, *args, **kwargs))
                        else:
                            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                                futures = [executor.submit(edit, image_info, *args, **kwargs) for image_info in subset.values()]
                                try:
                                    wait(futures)
                                    for future in futures:
                                        results.append(future.result())
                                except (gr.CancelledError, KeyboardInterrupt):
                                    for future in futures:
                                        future.cancel()
                                    raise
                    else:
                        results.append(edit(main_dataset[image_key].copy(), *args, **kwargs))

                    # write to dataset
                    for res in results:
                        if res is not None:
                            img_key = res.key
                            main_dataset.set(img_key, res)
                            if cur_dataset is not main_dataset and img_key in cur_dataset:
                                cur_dataset[img_key] = res

                    if any(results):
                        ret = track_image_key(image_key)
                        if image_key is None or image_key == '':
                            ret.update({log_box: f"{proc_func_log_name}: batch"})
                        else:
                            ret.update({log_box: f"{proc_func_log_name}: `{image_key}`"})
                        return ret
                    else:
                        return {log_box: f"{proc_func_log_name}: no change"}
                return wrapper

            def cancel():
                return {log_box: "cancelled."}

            cancel_event = cancel_btn.click(
                fn=cancel,
                outputs=[log_box],
                concurrency_limit=1,
            )

            def write_caption(image_info, caption):
                new_caption = Caption(caption) if caption is not None and caption.strip() != '' else None
                image_info.caption = new_caption
                return image_info

            caption.blur(
                fn=data_edition_handler(write_caption),
                inputs=[image_path, proc_opts, caption],
                outputs=cur_image_key_change_listeners,
                cancels=cancel_event,
                concurrency_limit=1,
            )

            def undo(image_info):
                image_info = main_dataset.undo(image_info.key)
                return image_info

            undo_btn.click(
                fn=data_edition_handler(undo),
                inputs=[cur_image_key, proc_opts],
                outputs=cur_image_key_change_listeners,
                concurrency_limit=1,
            )

            def redo(image_info):
                image_info = main_dataset.redo(image_info.key)
                return image_info

            redo_btn.click(
                fn=data_edition_handler(redo),
                inputs=[cur_image_key, proc_opts],
                outputs=cur_image_key_change_listeners,
                concurrency_limit=1,
            )

            def save_to_disk(progress: gr.Progress = gr.Progress(track_tqdm=True)):
                main_dataset.save(progress=progress)
                return f"saved!"

            save_btn.click(
                fn=save_to_disk,
                inputs=[],
                outputs=[log_box],
                concurrency_limit=1,
            )

            # ========================================= Quick Tagging ========================================= #

            def change_quality(image_info: ImageInfo, quality):
                caption = image_info.caption or Caption()
                caption.quality = quality
                image_info.caption = caption
                print(f"new caption: {caption}")
                return image_info

            tagging_best_quality_btn.click(
                fn=data_edition_handler(kwargs_setter(change_quality, quality='best')),
                inputs=[image_path, proc_opts],
                outputs=cur_image_key_change_listeners,
                concurrency_limit=1,
            )

            tagging_high_quality_btn.click(
                fn=data_edition_handler(kwargs_setter(change_quality, quality='high')),
                inputs=[image_path, proc_opts],
                outputs=cur_image_key_change_listeners,
                concurrency_limit=1,
            )

            tagging_low_quality_btn.click(
                fn=data_edition_handler(kwargs_setter(change_quality, quality='low')),
                inputs=[image_path, proc_opts],
                outputs=cur_image_key_change_listeners,
                concurrency_limit=1,
            )

            tagging_worst_quality_btn.click(
                fn=data_edition_handler(kwargs_setter(change_quality, quality='worst')),
                inputs=[image_path, proc_opts],
                outputs=cur_image_key_change_listeners,
                concurrency_limit=1,
            )

            def format_tag(image_info, tag):
                r"""
                %dir%: category (directory) name
                %cat%: same as %dirname%
                %stem%: filename
                """
                tag = tag.replace('%dir%', image_info.category)
                tag = tag.replace('%dirname%', image_info.category)
                tag = tag.replace('%cat%', image_info.category)
                tag = tag.replace('%category%', image_info.category)
                tag = tag.replace('%stem%', image_info.image_path.stem)
                tag = tag.replace('%filename%', image_info.image_path.stem)
                if re.search(r'%.*%', tag):
                    raise gr.Error(f"invalid tag format: {tag}")
                return tag

            def add_tags(image_info, tags, append):
                caption = image_info.caption or Caption()
                if isinstance(tags, str):
                    tags = [tags]
                tags = [format_tag(image_info, tag) for tag in tags]
                if append:
                    caption = (caption - tags) | tags
                else:
                    caption = tags | caption
                image_info.caption = caption
                return image_info

            def remove_tags(image_info, tags, regex):
                caption = image_info.caption
                if caption is None:
                    return image_info
                if isinstance(tags, str):
                    tags = [tags]
                tags = [format_tag(image_info, tag) for tag in tags]
                if regex:
                    try:
                        tags = [re.compile(tag) for tag in tags]
                    except re.error as e:
                        raise gr.Error(f"invalid regex: {e}")
                caption = image_info.caption - tags
                image_info.caption = caption
                return image_info

            tagging_color_btn.click(
                fn=data_edition_handler(kwargs_setter(add_tags, tags='beautiful color')),
                inputs=[image_path, proc_opts],
                outputs=cur_image_key_change_listeners,
                concurrency_limit=1,
            )

            tagging_detailed_btn.click(
                fn=data_edition_handler(kwargs_setter(add_tags, tags='detailed')),
                inputs=[image_path, proc_opts],
                outputs=cur_image_key_change_listeners,
                concurrency_limit=1,
            )

            tagging_lowres_btn.click(
                fn=data_edition_handler(kwargs_setter(add_tags, tags='lowres')),
                inputs=[image_path, proc_opts],
                outputs=cur_image_key_change_listeners,
                concurrency_limit=1,
            )

            tagging_messy_btn.click(
                fn=data_edition_handler(kwargs_setter(add_tags, tags='messy')),
                inputs=[image_path, proc_opts],
                outputs=cur_image_key_change_listeners,
                concurrency_limit=1,
            )

            tagging_aesthetic_btn.click(
                fn=data_edition_handler(kwargs_setter(add_tags, tags='aesthetic')),
                inputs=[image_path, proc_opts],
                outputs=cur_image_key_change_listeners,
                concurrency_limit=1,
            )

            tagging_beautiful_btn.click(
                fn=data_edition_handler(kwargs_setter(add_tags, tags='beautiful')),
                inputs=[image_path, proc_opts],
                outputs=cur_image_key_change_listeners,
                concurrency_limit=1,
            )

            tagging_amazing_quality_btn.click(
                fn=data_edition_handler(kwargs_setter(change_quality, quality='amazing')),
                inputs=[image_path, proc_opts],
                outputs=cur_image_key_change_listeners,
                concurrency_limit=1,
            )

            tagging_y_btn.click(
                fn=data_edition_handler(kwargs_setter(add_tags, tags='y')),
                inputs=[image_path, proc_opts],
                outputs=cur_image_key_change_listeners,
                concurrency_limit=1,
            )

            # ========================================= Custom Tagging ========================================= #

            for add_tag_btn, tag_selector, remove_tag_btn in zip(add_tag_btns, tag_selectors, remove_tag_btns):
                add_tag_btn.click(
                    fn=data_edition_handler(add_tags),
                    inputs=[image_path, proc_opts, tag_selector],
                    outputs=cur_image_key_change_listeners,
                    concurrency_limit=1,
                )
                remove_tag_btn.click(
                    fn=data_edition_handler(remove_tags),
                    inputs=[image_path, proc_opts, tag_selector],
                    outputs=cur_image_key_change_listeners,
                    concurrency_limit=1,
                )

            def replace_tag(image_info, old, new, regex):
                caption = image_info.caption
                if caption is None:
                    return image_info
                if regex:
                    try:
                        old = re.compile(old)
                    except re.error as e:
                        raise gr.Error(f"invalid regex `{old}`: {e}")
                    try:
                        caption = caption.sub(old, new)
                    except re.error as e:
                        raise gr.Error(f"regex error: {e}")
                else:
                    caption = caption.replace(old, new)
                image_info.caption = caption
                return image_info

            for replace_tag_btn, old_tag_selector, new_tag_selector in zip(replace_tag_btns, old_tag_selectors, new_tag_selectors):
                replace_tag_btn.click(
                    fn=data_edition_handler(replace_tag),
                    inputs=[image_path, proc_opts, old_tag_selector, new_tag_selector],
                    outputs=cur_image_key_change_listeners,
                    concurrency_limit=1,
                )

            # ========================================= Caption Operation ========================================= #

            # def caption_operation(image_info, op, op_tags, condition, cond_tags, inclusion_relationship):
            #     if global_args.language != 'en':
            #         op = translate(op, 'en')
            #         condition = translate(condition, 'en')
            #         inclusion_relationship = translate(inclusion_relationship, 'en')
            #     # print(f"op: {op} | op_tags: {op_tags} | condition: {condition} | cond_tags: {cond_tags} | inclusion_relationship: {inclusion_relationship}")
            #     caption = image_info.caption or Caption()
            #     if op_tags is None or len(op_tags) == 0:
            #         return caption
            #     op_tags = [format_tag(image_info, tag) for tag in op_tags]
            #     op_func = OPS[op]
            #     op_caption = Caption(op_tags)
            #     do_condition = condition is not None and condition != ''
            #     cond_tags = [format_tag(image_info, tag) for tag in cond_tags]
            #     if do_condition and cond_tags and len(cond_tags) > 0:
            #         cond_func = CONDITION[condition]
            #         cond_tags = set(cond_tags)
            #         incl_func = INCLUSION_RELATIONSHIP[inclusion_relationship]

            #         if not cond_func(incl_func(caption, cond_tag) for cond_tag in cond_tags):
            #             return caption

            #     caption = op_func(caption, op_caption)
            #     # print(f"caption: {caption}")
            #     return caption

            # operate_caption_btn.click(
            #     fn=edit_caption_wrapper(caption_operation),
            #     inputs=[image_path, proc_mode_checkbox_group, cap_op_op_dropdown, cap_op_op_tag_dropdown, cap_op_cond_dropdown, cap_op_cond_tag_dropdown, cap_op_incl_rel_dropdown],
            #     outputs=[caption, log_box],
            #     concurrency_limit=1,
            # )

            # ========================================= Optimizers ========================================= #
            def sort_caption(image_info):
                caption = image_info.caption
                if caption is None:
                    return image_info
                nonlocal tag_priority_manager
                tagging.init_priority_tags()
                image_info.caption = caption @ tag_priority_manager.priority_regex
                return image_info

            sort_caption_btn.click(
                fn=data_edition_handler(sort_caption),
                inputs=[image_path, proc_opts],
                outputs=cur_image_key_change_listeners,
                concurrency_limit=1,
            )

            def formalize_caption(image_info: ImageInfo, formats):
                caption = image_info.caption
                if caption is None:
                    return image_info
                if isinstance(formats, str):
                    formats = [formats]
                if global_args.language != 'en':
                    formats = [translate(fmt, 'en') for fmt in formats]
                for fmt in formats:
                    caption = FORMAT[fmt](caption)
                image_info.caption = caption
                return image_info

            formalize_caption_btn.click(
                fn=data_edition_handler(formalize_caption),
                inputs=[image_path, proc_opts, formalize_caption_dropdown],
                outputs=cur_image_key_change_listeners,
                concurrency_limit=1,
            )

            def deduplicate_caption(image_info):
                caption = image_info.caption
                if caption is None:
                    return image_info
                image_info.caption = caption.unique()
                return image_info

            deduplicate_caption_btn.click(
                fn=data_edition_handler(deduplicate_caption),
                inputs=[image_path, proc_opts],
                outputs=cur_image_key_change_listeners,
                concurrency_limit=1,
            )

            def deoverlap_caption(image_info):
                caption = image_info.caption
                if caption is None:
                    return image_info
                image_info.caption = caption.deovlped()
                return image_info

            deoverlap_caption_btn.click(
                fn=data_edition_handler(deoverlap_caption),
                inputs=[image_path, proc_opts],
                outputs=cur_image_key_change_listeners,
                concurrency_limit=1,
            )

            def defeature_caption(image_info: ImageInfo, freq_thres, count_thres, least_sample_size, ref):
                # make feature table using tag table
                nonlocal tag_table, tag_feature_table
                if tag_table is None:
                    main_dataset.init_tag_table()
                    tag_table = main_dataset.tag_table

                caption: Caption = image_info.caption
                if caption is None:
                    return image_info
                characters = caption.characters
                if characters is None or len(characters) == 0:
                    return image_info
                for character in characters:
                    character = preprocess_tag(character)
                    if character not in ref:
                        # print(f"making ref: `{character}`")
                        ref[character] = {}
                        img_keys = tag_table[character]
                        img_infos = [main_dataset[key] for key in img_keys]
                        if len(img_infos) < least_sample_size:
                            continue
                        for img_info in img_infos:
                            cap = img_info.caption
                            for tag in cap & tagging.REGEX_CHARACTER_FEATURES:
                                tag = preprocess_tag(tag)
                                if tag not in ref[character]:
                                    ref[character][tag] = 0
                                ref[character][tag] += 1
                        ref[character] = {tag: (count, count / len(img_infos)) for tag, count in ref[character].items()}

                image_info.caption = caption.defeatured(ref=ref, freq_thres=freq_thres, count_thres=count_thres)
                return image_info

            defeature_caption_btn.click(
                fn=lambda *args: data_edition_handler(defeature_caption)(*args, ref={}),
                inputs=[image_path, proc_opts, defeature_freq_thres, defeature_count_thres, defeature_least_sample_size],
                outputs=cur_image_key_change_listeners,
                concurrency_limit=1,
            )

            # ========================================= WD14 ========================================= #

            def wd14_tagging(image_info, general_threshold, character_threshold, os_mode):
                if global_args.language != 'en':
                    os_mode = translate(os_mode, 'en')
                nonlocal waifu_tagger
                old_caption = image_info.caption
                if old_caption is not None and os_mode == 'ignore':
                    return image_info
                if waifu_tagger is None:
                    from modules.compoents import WaifuTagger
                    waifu_tagger = WaifuTagger(model_path=global_args.wd14_model_path, label_path=global_args.wd14_label_path, verbose=True)
                image = Image.open(image_info.image_path)
                wd14_caption = waifu_tagger(image, general_threshold=general_threshold, character_threshold=character_threshold)
                if os_mode == 'overwrite' or os_mode == 'ignore':
                    caption = wd14_caption
                elif os_mode == 'append':
                    caption = old_caption | wd14_caption
                elif os_mode == 'prepend':
                    caption = wd14_caption | old_caption
                else:
                    raise ValueError(f"invalid os_mode: {os_mode}")
                image_info.caption = caption
                return image_info

            wd14_run_btn.click(
                fn=data_edition_handler(wd14_tagging),
                inputs=[image_path, proc_opts, wd14_general_threshold, wd14_character_threshold, wd14_caption_proc_mode],
                outputs=cur_image_key_change_listeners,
                concurrency_limit=1,
            )

            # ========================================= Aesthetic predictor ========================================= #
            def predict_aesthetic_score(image_info, os_mode):
                if global_args.language != 'en':
                    os_mode = translate(os_mode, 'en')

                nonlocal waifu_scorer
                if waifu_scorer is None:
                    from modules.compoents import WaifuScorer
                    waifu_scorer = WaifuScorer(model_path=global_args.waifu_scorer_model_path, device='cuda', verbose=True)

                image = Image.open(image_info.image_path)
                orig_score = image_info.aesthetic_score
                if orig_score is not None and os_mode == 'ignore':
                    return image_info

                pred_score = waifu_scorer(image)

                image_info.aesthetic_score = pred_score
                return image_info

            predict_aesthetic_score_btn.click(
                fn=data_edition_handler(predict_aesthetic_score),
                inputs=[cur_image_key, proc_opts, waifu_scorer_os_mode],
                outputs=cur_image_key_change_listeners,
                concurrency_limit=1,
                cancels=cancel_event,
            )

            def aesthetic_score_to_quality(score):
                if score is None:
                    return None
                if score >= 9:
                    return 'amazing'
                elif score >= 7.5:
                    return 'best'
                elif score >= 6:
                    return 'high'
                elif score >= 4:
                    return 'normal'
                elif score >= 2.5:
                    return 'low'
                elif score >= 1:
                    return 'worst'
                else:
                    return 'horrible'

            def change_quality_according_to_aesthetic_score(image_info: ImageInfo, os_mode):
                caption = image_info.caption
                orig_quality = caption.quality if caption is not None else None
                if global_args.language != 'en':
                    os_mode = translate(os_mode, 'en')
                if orig_quality is not None and os_mode == 'ignore':
                    return image_info
                score = image_info.aesthetic_score
                if score is None:
                    return image_info
                quality = aesthetic_score_to_quality(score)
                return change_quality(image_info, quality)

            label_aesthetic_btn.click(
                fn=data_edition_handler(change_quality_according_to_aesthetic_score),
                inputs=[cur_image_key, proc_opts, waifu_scorer_os_mode],
                outputs=cur_image_key_change_listeners,
                concurrency_limit=1,
            )

            def set_aesthetic_score(image_info, score):
                image_info.aesthetic_score = score
                return image_info

            clean_aesthetic_score_btn.click(
                fn=data_edition_handler(kwargs_setter(set_aesthetic_score, score=None)),
                inputs=[cur_image_key, proc_opts],
                outputs=cur_image_key_change_listeners,
                concurrency_limit=1,
            )

            # ========================================= Perceptual Hash ========================================= #
            def get_perceptual_hash(image_info, os_mode):
                if global_args.language != 'en':
                    os_mode = translate(os_mode, 'en')

                try:
                    import imagehash
                except ImportError:
                    raise gr.Error("imagehash package is not installed!")

                orig_p_hash = image_info.perceptual_hash
                if orig_p_hash is not None and os_mode == 'ignore':
                    return image_info
                image = Image.open(image_info.image_path)
                p_hash = imagehash.phash(image)
                image_info.perceptual_hash = p_hash
                return image_info

            get_perceptual_hash_btn.click(
                fn=data_edition_handler(get_perceptual_hash),
                inputs=[cur_image_key, proc_opts, hasher_os_mode],
                outputs=cur_image_key_change_listeners,
                concurrency_limit=1,
            )

            def set_perceptual_hash(image_info, p_hash):
                image_info.perceptual_hash = p_hash
                return image_info

            clean_perceptual_hash_btn.click(
                fn=data_edition_handler(kwargs_setter(set_perceptual_hash, p_hash=None)),
                inputs=[cur_image_key, proc_opts],
                outputs=cur_image_key_change_listeners,
                concurrency_limit=1,
            )

            # ========================================= Open file folder ========================================= #

            open_folder_btn.click(
                fn=open_file_folder,
                inputs=[image_path],
                outputs=[],
                concurrency_limit=1,
            )

            # ========================================= Buffer ========================================= #

            def show_buffer():
                return buffer.df()

            buffer_tab.select(
                fn=show_buffer,
                outputs=[buffer_df],
                concurrency_limit=1,
            )

            # ========================================= Query ========================================= #
            def query(func: Callable[[Tuple[Any, ...], Dict[str, Any]], UIChunkedDataset]):
                proc_func_name = func.__name__

                def wrapper(sorting_methods=None, reverse=False, *args, **kwargs):
                    nonlocal cur_dataset
                    if cur_dataset is None or len(cur_dataset) == 0:
                        return {log_box: f"empty dataset"}
                    res_dset = func(*args, **kwargs) or UIChunkedDataset()
                    res = change_to_dataset(res_dset, sorting_methods=sorting_methods, reverse=reverse)
                    res.update({log_box: f"{proc_func_name} matches {len(res_dset)} images over {len(cur_dataset)} images"})
                    return res
                return wrapper

            def query_tag_table(include_condition, include_tags, joiner, exclude_condition, exclude_tags, enable_regex, progress: gr.Progress = gr.Progress(track_tqdm=True)):
                # translate
                if global_args.language != 'en':
                    include_condition = translate(include_condition, 'en')
                    exclude_condition = translate(exclude_condition, 'en')
                    joiner = translate(joiner, 'en')

                nonlocal tag_table
                if tag_table is None:
                    main_dataset.init_tag_table()
                    tag_table = main_dataset.tag_table

                joiner_func = JOINER[joiner]

                # print(f"subset_key: {subset_key}")
                # print(f"include_tags: {include_tags}")
                # print(f"exclude_tags: {exclude_tags}")

                subset = cur_dataset

                incl_set = set()
                for pattern in tqdm(include_tags, desc='Including tags'):  # calculate the union of all key(s), s ∈ include_tags
                    if enable_regex:
                        regex = re.compile(pattern)
                        tags = [tag for tag in tag_table.keys() if regex.match(tag)]
                    else:
                        if pattern not in tag_table:
                            continue
                        tags = [pattern]

                    for tag in tags:
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
                for pattern in tqdm(exclude_tags, desc='Excluding tags'):  # calculate the union of all key(s), s ∈ exclude_tags
                    if enable_regex:
                        regex = re.compile(pattern)
                        tags = [tag for tag in tag_table.keys() if regex.match(tag)]
                    else:
                        if pattern not in tag_table:
                            continue
                        tags = [pattern]

                    for tag in tags:
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
                res_dataset = UIChunkedDataset({img_key: main_dataset[img_key] for img_key in res_set}, chunk_size=global_args.chunk_size)

                # print(f"incl_set: {incl_set}")
                # print(f"excl_set: {excl_set}")
                # print(f"res_set: {res_set}")
                # print(f"res_dataset: {res_dataset}")

                # search_range_size = len(cur_dataset)
                # res = change_to_dataset(res_dataset)
                # res.update({log_box: f"querying matches {len(res_dataset)} images over {search_range_size} images"})
                return res_dataset

            query_tab_table_btn.click(
                fn=query(query_tag_table),
                inputs=[sorting_methods_dropdown, sorting_reverse_checkbox, query_include_condition, query_include_tags,
                        query_joiner_dropdown, query_exclude_condition, query_exclude_tags, query_use_regex],
                outputs=dataset_change_listeners,
                show_progress=True,
                concurrency_limit=1,
            )

            def query_filename(image_keys, enable_regex):
                if image_keys is None or image_keys == '':
                    return None
                if enable_regex:
                    regex = [re.compile(img_key) for img_key in image_keys]
                    matched_img_keys = [img_key for img_key in cur_dataset.keys() if any(r.match(img_key) for r in regex)]
                else:
                    matched_img_keys = [img_key for img_key in image_keys if img_key in cur_dataset.keys()]
                if len(matched_img_keys) == 0:
                    return None
                res_dataset = UIChunkedDataset({img_key: main_dataset[img_key] for img_key in matched_img_keys}, chunk_size=global_args.chunk_size)
                return res_dataset

            query_filename_btn.click(
                fn=query(query_filename),
                inputs=[sorting_methods_dropdown, sorting_reverse_checkbox, query_filename_selector, query_use_regex],
                outputs=dataset_change_listeners,
                show_progress=True,
                concurrency_limit=1,
            )

            def query_aesthetic_score(min_score, max_score):
                if min_score is None and max_score is None:
                    return cur_dataset
                if min_score is None:
                    min_score = -float('inf')
                if max_score is None:
                    max_score = float('inf')
                if min_score > max_score:
                    min_score, max_score = max_score, min_score
                res_dataset = UIChunkedDataset(cur_dataset, condition=lambda img_info: img_info.aesthetic_score is not None and min_score <=
                                               img_info.aesthetic_score <= max_score, chunk_size=global_args.chunk_size)
                return res_dataset

            query_aes_score_btn.click(
                fn=query(query_aesthetic_score),
                inputs=[sorting_methods_dropdown, sorting_reverse_checkbox, query_min_aes_score, query_max_aes_score],
                outputs=dataset_change_listeners,
                show_progress=True,
                concurrency_limit=1,
            )

            def reload_query_tag_list(progress: gr.Progress = gr.Progress(track_tqdm=True)):
                nonlocal tag_table
                if tag_table is None:
                    main_dataset.init_tag_table()
                    tag_table = main_dataset.tag_table

                tag_list = sorted(tag_table.keys(), key=lambda x: len(tag_table[x]), reverse=True)
                return gr.update(choices=tag_list), gr.update(choices=tag_list)

            query_refresh_tag_list_btn.click(
                fn=reload_query_tag_list,
                outputs=[query_include_tags, query_exclude_tags],
                concurrency_limit=1,
            )

            def unload_tag_list():
                return gr.update(choices=None), gr.update(choices=None)

            query_unload_tag_list_btn.click(
                fn=unload_tag_list,
                outputs=[query_include_tags, query_exclude_tags],
                concurrency_limit=1,
            )

            # ========================================= Config ========================================= #

            def update_tag_priority(name):
                def wrapper(level, tags):
                    nonlocal tag_priority_manager
                    patterns = tagging.PRIORITY.get(name, [])
                    patterns.extend(tags)
                    tp = TagPriority(name, patterns, level)
                    tag_priority_manager[name] = tp
                    return f"priority {name} updated"
                return wrapper

            for tag_priority_level, tag_priority_custom_tags in tag_priority_comps:
                params = dict(
                    fn=update_tag_priority(name),
                    inputs=[tag_priority_level, tag_priority_custom_tags],
                    outputs=[cfg_log_box],
                )
                tag_priority_level.change(**params)
                tag_priority_custom_tags.change(**params)

            def export_tag_priority_config():
                import json
                nonlocal tag_priority_manager
                config = tag_priority_manager.config
                with open(global_args.tag_priority_config_path, 'w') as f:
                    json.dump(config, f, indent=4)
                return f"priority config saved"

            export_tag_priority_config_btn.click(
                fn=export_tag_priority_config,
                outputs=[cfg_log_box],
            )

            import_priority_config_btn.click(
                fn=import_priority_config,
                outputs=[cfg_log_box],
            )

    return demo
