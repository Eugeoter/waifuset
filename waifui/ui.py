import os
import re
import gradio as gr
import inspect
import pandas
from pathlib import Path
from PIL import Image
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Callable, Any, Tuple, Dict, List, Iterable, Union, Optional, Literal
from pathlib import Path
from copy import deepcopy
from waifuset.const import IMAGE_EXTS
from waifuset import logging
from waifuset import Dataset, Caption, SQLite3Dataset, AutoDataset, FastDataset
from waifuset.utils import image_utils, class_utils
from waifuset.classes.dataset.dataset_mixin import ToDiskMixin
from waifuset.classes.data import data_utils
from waifuset.components.waifu_tagger.predict import WD_REPOS
from waifuset.components.waifu_scorer.predict import WS_REPOS
from .emoji import Emoji
from .ui_utils import *
from .ui_dataset import UIDataset, UISubset


class UIManager(class_utils.FromConfigMixin):
    dataset_source: Union[str, List[str], Dict[str, Any], List[Dict[str, Any]]] = None
    share: bool = False
    port: Optional[int] = None
    language: Literal['en', 'cn'] = 'cn'
    page_size: int = 40
    cpu_max_workers: int = 1
    verbose: bool = False

    logger: logging.ConsoleLogger

    @classmethod
    def from_config(cls, config, **kwargs):
        logger = logging.get_logger('UI', disable=not config.verbose)
        self: UIManager = super().from_config(config, logger=logger, **kwargs)
        self.check_config(config)
        self.setup()
        return self

    # TODO: check_config
    def check_config(self, config):
        pass

    def get_default_kwargs(self):
        return {
            "primary_key": "image_key",
            "fp_key": "image_path",
            "recur": True,
            "exts": IMAGE_EXTS,
            "tbname": "metadata",
        }

    def load_dataset(self):
        with self.logger.timer('loading UI dataset'):
            with self.logger.timer('load dataset'):
                dataset = FastDataset(self.dataset_source, verbose=self.verbose, **self.get_default_kwargs())
            with self.logger.timer('patch dataset'):
                if any(col not in dataset.header for col in ('image_key', 'category')):
                    self.logger.print('patching image path base info')
                    dataset.add_columns(['image_path', 'image_key', 'category', 'source', 'caption', 'description'])
                    dataset.apply_map(patch_image_path_base_info)
            dataset = UIDataset(
                dataset,
                page_size=self.page_size
            )
        # self.logger.print(dataset, no_prefix=True)
        self.logger.print(f"dataset size: {len(dataset)}x{len(dataset.header)}")
        return dataset

    def setup(self):
        self.logger.print("setting up UI")
        with self.logger.timer('setup'):
            self.dataset = self.load_dataset()
            with self.logger.timer('launch ui'):
                self.ui = create_ui(
                    univset=self.dataset,
                    buffer=UIBuffer(),
                    cpu_max_workers=self.cpu_max_workers,
                    language=self.language,
                    render='full',
                )

    def launch(self):
        self.logger.print("launching UI")
        self.ui.launch(
            share=self.share,
            server_port=self.port,
        )


def create_ui(
    univset: UIDataset,
    buffer: UIBuffer,
    cpu_max_workers=1,
    language='en',
    render='full',
):
    # ========================================= UI ========================================= #
    logger = logging.get_logger('UI')
    state = UIState(
        page_index=0,
        selected=UIGallerySelectData(),
    )
    waifu_tagger = None
    waifu_scorer = None
    tagging.init_custom_tags()

    def convert_dataset_to_statistic_dataframe(dataset: UISubset):
        num_images = len(dataset)
        categories = sorted(dataset.get_categories()) if len(dataset) > 0 else []
        num_cats = len(categories)
        if num_cats > 5:
            categories = categories[:5] + ['...']
        df = pandas.DataFrame(
            data={
                translate('Number of images', language): [num_images],
                translate('Number of categories', language): [num_cats],
                translate('Categories', language): [categories],
            },
        )
        return df

    def convert_dataset_to_gallery(dataset: Dataset):
        return [(v['image_path'], k) for k, v in dataset.items()]

    # demo
    with gr.Blocks() as demo:
        with gr.Tab(label=translate('Dataset', language)) as dataset_tab:
            with gr.Tab(translate('Main', language)) as main_tab:
                # ========================================= Base variables ========================================= #

                with gr.Row():
                    with gr.Column():
                        with gr.Tab(translate('Category', language)):
                            with gr.Row():
                                category_selector = gr.Dropdown(
                                    label=translate('Category', language),
                                    choices=univset.get_categories(),
                                    value=None,
                                    container=False,
                                    multiselect=True,
                                    allow_custom_value=False,
                                    min_width=256,
                                )
                                reload_category_btn = EmojiButton(Emoji.anticlockwise)

                        with gr.Tab(translate('Sort', language)):
                            with gr.Row():
                                sorting_methods_dropdown = gr.Dropdown(
                                    label=translate('Sorting Methods', language),
                                    choices=translate(SORTING_METHODS.keys(), language),
                                    value=None,
                                    container=False,
                                    multiselect=True,
                                    allow_custom_value=False,
                                    min_width=256,
                                )
                                reload_sort_btn = EmojiButton(Emoji.anticlockwise)
                            with gr.Row():
                                sorting_reverse_checkbox = gr.Checkbox(
                                    label=translate('Reverse', language),
                                    value=False,
                                    scale=0,
                                    min_width=128,
                                    container=False,
                                )

                        with gr.Tab(translate('Query', language)):
                            with gr.Row():
                                query_opts = gr.CheckboxGroup(
                                    choices=[translate('Subset', language), translate('Complement', language), translate('Regex', language)],
                                    value=None,
                                    container=False,
                                    scale=1,
                                )
                            with gr.Tab(translate("Attribute", language)) as query_tag_tab:
                                with gr.Row(variant='compact'):
                                    query_attr_selector = gr.Dropdown(
                                        choices=univset.header,
                                        value=None,
                                        multiselect=False,
                                        allow_custom_value=False,
                                        show_label=False,
                                        container=False,
                                        min_width=96,
                                    )
                                    query_attr_pattern = gr.Textbox(
                                        value=None,
                                        show_label=False,
                                        container=False,
                                        min_width=128,
                                        lines=1,
                                        max_lines=1,
                                        placeholder='Pattern',
                                    )
                                    query_attr_btn = EmojiButton(Emoji.right_pointing_magnifying_glass, variant='primary')

                    with gr.Column():
                        with gr.Row():
                            log_box = gr.TextArea(
                                label=translate('Log', language),
                                lines=1,
                                max_lines=1,
                            )

                with gr.Tab(translate('Dataset', language)) as tagging_tab:
                    with gr.Row():
                        with gr.Column():
                            with gr.Row():
                                showcase = gr.Gallery(
                                    label=translate('Showcase', language),
                                    value=convert_dataset_to_gallery(univset.curset.page(0)),
                                    rows=4,
                                    columns=4,
                                    container=True,
                                    object_fit='scale-down',
                                    height=512,
                                )
                            with gr.Row():
                                load_pre_page_btn = EmojiButton(Emoji.black_left_pointing_double_triangle, scale=1)
                                load_next_page_btn = EmojiButton(Emoji.black_right_pointing_double_triangle, scale=1)
                            with gr.Row():
                                with gr.Column():
                                    ...
                                with gr.Column():
                                    cur_page_number = gr.Number(label=f"{translate('Chunk', language)} {1}/{univset.num_pages}", value=1, min_width=128, precision=0, scale=0)
                                with gr.Column():
                                    ...
                            with gr.Row():
                                dataset_metadata_df = gr.Dataframe(
                                    value=convert_dataset_to_statistic_dataframe(univset.curset),
                                    label=translate('Dataset Information', language),
                                    type='pandas',
                                    row_count=(1, 'fixed'),
                                )

                        with gr.Column():
                            with gr.Row():
                                with gr.Tab(translate('Caption', language)) as caption_tab:
                                    caption = gr.Textbox(
                                        show_label=False,
                                        value=None,
                                        container=True,
                                        show_copy_button=True,
                                        lines=6,
                                        max_lines=6,
                                        placeholder=translate('empty', language),
                                    )
                                with gr.Tab(translate('Metadata', language)) as metadata_tab:
                                    with gr.Row():
                                        caption_metadata_df = gr.Dataframe(
                                            label=translate("Caption", language),
                                            value=None,
                                            type='pandas',
                                            row_count=(1, 'fixed'),
                                        )
                                    with gr.Row():
                                        other_metadata_df = gr.Dataframe(
                                            label=translate("Other", language),
                                            value=None,
                                            type='pandas',
                                            row_count=(1, 'fixed'),
                                        )
                                with gr.Tab(translate('Description', language)) as description_tab:
                                    description = gr.Textbox(
                                        show_label=False,
                                        value=None,
                                        container=True,
                                        show_copy_button=True,
                                        lines=6,
                                        max_lines=6,
                                        placeholder=translate('empty', language),
                                    )

                                with gr.Tab(translate('Generation Information', language)) as gen_info_tab:
                                    with gr.Tab(label=translate('Positive Prompt', language)):
                                        positive_prompt = gr.Textbox(
                                            label=translate('Positive Prompt', language),
                                            value=None,
                                            container=False,
                                            show_copy_button=True,
                                            lines=6,
                                            max_lines=6,
                                            interactive=True,
                                            placeholder=translate('empty', language),
                                        )
                                    with gr.Tab(label=translate('Negative Prompt', language)):
                                        negative_prompt = gr.Textbox(
                                            label=translate('Negative Prompt', language),
                                            value=None,
                                            container=False,
                                            show_copy_button=True,
                                            lines=6,
                                            max_lines=6,
                                            interactive=True,
                                            placeholder=translate('empty', language),
                                        )
                                    with gr.Tab(label=translate('Generation Parameters', language)):
                                        gen_params_df = gr.Dataframe(
                                            value=None,
                                            show_label=False,
                                            type='pandas',
                                            row_count=(1, 'fixed'),
                                        )

                            with gr.Row():
                                # random_btn = EmojiButton(Emoji.dice)
                                # set_category_btn = EmojiButton(Emoji.top_left_arrow)
                                undo_btn = EmojiButton(Emoji.leftwards)
                                redo_btn = EmojiButton(Emoji.rightwards)
                                save_btn = EmojiButton(Emoji.floppy_disk, variant='primary')
                                save_path = gr.Textbox(
                                    value=None,
                                    show_label=False,
                                    max_lines=1,
                                    lines=1,
                                    show_copy_button=True,
                                    interactive=True,
                                )
                                cancel_btn = EmojiButton(Emoji.no_entry, variant='stop')
                            with gr.Row():
                                general_edit_opts = gr.CheckboxGroup(
                                    label=translate('Process options', language),
                                    choices=translate(['Batch', 'Append', 'Regex', 'Progress'], language),
                                    value=None,
                                    container=False,
                                    scale=1,
                                )

                            with gr.Tab(label=translate('Custom Tagging', language)):
                                with gr.Tab(translate("Add/Remove", language)) as add_remove_tab:
                                    def custom_add_rem_tagging_row():
                                        with gr.Row(variant='compact'):
                                            add_tag_btn = EmojiButton(Emoji.plus, variant='primary')
                                            tag_selector = gr.Dropdown(
                                                choices=list(tagging.CUSTOM_TAGS or []),
                                                value=None,
                                                multiselect=True,
                                                allow_custom_value=True,
                                                show_label=False,
                                                container=False,
                                                min_width=96,
                                            )
                                            remove_tag_btn = EmojiButton(Emoji.minus, variant='stop')
                                        return add_tag_btn, tag_selector, remove_tag_btn

                                    add_tag_btns = []
                                    tag_selectors = []
                                    remove_tag_btns = []
                                    for r in range(3):
                                        add_tag_btn, tag_selector, remove_tag_btn = custom_add_rem_tagging_row()
                                        add_tag_btns.append(add_tag_btn)
                                        tag_selectors.append(tag_selector)
                                        remove_tag_btns.append(remove_tag_btn)

                                    with gr.Accordion(label=translate('More', language), open=False):
                                        for r in range(6):
                                            add_tag_btn, tag_selector, remove_tag_btn = custom_add_rem_tagging_row()
                                            add_tag_btns.append(add_tag_btn)
                                            tag_selectors.append(tag_selector)
                                            remove_tag_btns.append(remove_tag_btn)

                                with gr.Tab(translate("Replace", language)) as replace_tab:
                                    def custom_replace_tagging_row():
                                        with gr.Row(variant='compact'):
                                            replace_tag_btn = EmojiButton(Emoji.clockwise_downwards_and_upwards_open_circle_arrows, variant='primary')
                                            old_tag_selector = gr.Dropdown(
                                                # label=translate('Replacer', global_args.language),
                                                choices=list(tagging.CUSTOM_TAGS or []),
                                                value=None,
                                                container=False,
                                                multiselect=False,
                                                allow_custom_value=True,
                                                min_width=96,
                                            )
                                            new_tag_selector = gr.Dropdown(
                                                # label=translate('Replacement', global_args.language),
                                                choices=list(tagging.CUSTOM_TAGS or []),
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

                                    match_tag_checkbox = gr.Checkbox(
                                        label=translate('Match Tag', language),
                                        value=True,
                                        scale=0,
                                        min_width=128,
                                    )

                                    with gr.Accordion(label=translate('More', language), open=False):
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

                            #         operate_caption_btn = EmojiButton(Emoji.black_right_pointing_triangle, variant='primary')

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

                            with gr.Tab(label=translate('Optimizers', language)):
                                with gr.Tab(translate('Read', language)):
                                    with gr.Row(variant='compact'):
                                        read_attrs_btn = EmojiButton(Emoji.black_right_pointing_triangle, min_width=40, variant='primary')
                                    with gr.Row(variant='compact'):
                                        read_attrs_types = gr.CheckboxGroup(
                                            label=translate('Types', language),
                                            choices=translate(['txt', 'danbooru'], language),
                                            value=translate(['txt', 'danbooru'], language),
                                            scale=1,
                                        )
                                with gr.Tab(translate('Parse', language)):
                                    with gr.Row(variant='compact'):
                                        parse_caption_attrs_btn = EmojiButton(Emoji.black_right_pointing_triangle, min_width=40, variant='primary')
                                with gr.Tab(translate('Sort', language)):
                                    with gr.Row(variant='compact'):
                                        sort_caption_btn = EmojiButton(Emoji.black_right_pointing_triangle, min_width=40, variant='primary')
                                with gr.Tab(translate('Deduplicate', language)):
                                    with gr.Row(variant='compact'):
                                        deduplicate_caption_btn = EmojiButton(Emoji.black_right_pointing_triangle, min_width=40, variant='primary')
                                with gr.Tab(translate('Deoverlap', language)):
                                    with gr.Row(variant='compact'):
                                        deoverlap_caption_btn = EmojiButton(Emoji.black_right_pointing_triangle, min_width=40, variant='primary')
                                with gr.Tab(translate('Defeature', language)):
                                    with gr.Row(variant='compact'):
                                        decharacterize_caption_btn = EmojiButton(Emoji.black_right_pointing_triangle, min_width=40, variant='primary')
                                    with gr.Row(variant='compact'):
                                        dechar_feature_type = gr.Radio(
                                            label=translate('Feature Type', language),
                                            choices=translate(['physics', 'clothes'], language),
                                            value=translate('physics', language),
                                            scale=1,
                                            min_width=128,
                                        )
                                        dechar_freq_thres = gr.Slider(
                                            label=translate('Frequency Threshold', language),
                                            value=0.3,
                                            minimum=0,
                                            maximum=1,
                                            step=0.01,
                                            scale=1,
                                            min_width=128,
                                        )
                                with gr.Tab(translate('Formalize', language)):
                                    with gr.Row(variant='compact'):
                                        formalize_caption_btn = EmojiButton(Emoji.black_right_pointing_triangle, scale=0, min_width=40, variant='primary')
                                    with gr.Row(variant='compact'):
                                        formalize_caption_dropdown = gr.Dropdown(
                                            label=translate('Format', language),
                                            choices=translate(list(FORMAT_PRESETS.keys()), language),
                                            value=None,
                                            multiselect=True,
                                            allow_custom_value=False,
                                            scale=1,
                                        )

                            with gr.Tab(label=translate('Tools', language)):

                                with gr.Tab(label=translate('Tagger', language)):
                                    with gr.Row(variant='compact'):
                                        wd_run_btn = EmojiButton(Emoji.black_right_pointing_triangle, variant='primary', min_width=40)

                                    with gr.Row(variant='compact'):
                                        wd_model = gr.Dropdown(
                                            label=translate('Model', language),
                                            choices=WD_REPOS,
                                            value=WD_REPOS[0],
                                            multiselect=False,
                                            allow_custom_value=True,
                                            scale=1,
                                        )

                                    with gr.Row(variant='compact'):
                                        wd_batch_size = gr.Number(
                                            value=1,
                                            label=translate('Batch Size', language),
                                            min_width=128,
                                            precision=0,
                                            scale=1,
                                        )

                                        wd_overwrite_mode = gr.Radio(
                                            label=translate('Overwrite mode', language),
                                            choices=translate(['overwrite', 'ignore', 'append', 'prepend'], language),
                                            value=translate('overwrite', language),
                                            scale=1,
                                            min_width=128,
                                        )

                                    with gr.Row(variant='compact'):
                                        wd_general_threshold = gr.Slider(
                                            label=translate('General Threshold', language),
                                            value=0.35,
                                            minimum=0,
                                            maximum=1,
                                            step=0.01,
                                        )
                                        wd_character_threshold = gr.Slider(
                                            label=translate('Character Threshold', language),
                                            value=0.35,
                                            minimum=0,
                                            maximum=1,
                                            step=0.01,
                                        )

                                with gr.Tab(label=translate('Scorer', language)):
                                    with gr.Row(variant='compact'):
                                        ws_run_btn = EmojiButton(Emoji.black_right_pointing_triangle, variant='primary', min_width=40)
                                        ws_delabel_btn = EmojiButton(Emoji.no_entry, variant='stop', min_width=40)

                                    with gr.Row(variant='compact'):
                                        ws_model = gr.Dropdown(
                                            label=translate('Model', language),
                                            choices=WS_REPOS,
                                            value=WS_REPOS[0],
                                            multiselect=False,
                                            allow_custom_value=True,
                                            scale=1,
                                        )

                                    with gr.Row(variant='compact'):
                                        ws_batch_size = gr.Number(
                                            value=1,
                                            label=translate('Batch Size', language),
                                            min_width=128,
                                            precision=0,
                                            scale=1,
                                        )
                                        ws_overwrite_mode = gr.Radio(
                                            label=translate('Overwrite mode', language),
                                            choices=translate(['overwrite', 'ignore'], language),
                                            value=translate('overwrite', language),
                                            scale=1,
                                            min_width=128,
                                        )

                                with gr.Tab(label=translate('Hasher', language)):
                                    with gr.Row(variant='compact'):
                                        hasher_run_btn = EmojiButton(Emoji.black_right_pointing_triangle, variant='primary', min_width=40)
                                        hasher_dehash_btn = EmojiButton(Emoji.no_entry, variant='stop', min_width=40)

                                    with gr.Row(variant='compact'):
                                        hasher_overwrite_mode = gr.Radio(
                                            label=translate('Overwrite mode', language),
                                            choices=translate(['overwrite', 'ignore'], language),
                                            value=translate('overwrite', language),
                                            scale=1,
                                            min_width=128,
                                        )

                    with gr.Row():
                        with gr.Column(scale=4):
                            with gr.Tab(translate('Image Key', language)):
                                with gr.Row(variant='compact'):
                                    cur_img_key = gr.Textbox(value=None, show_label=False, max_lines=1, lines=1)

                            with gr.Tab(translate('Image Path', language)):
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
                                    open_folder_btn = EmojiButton(Emoji.open_file_folder)
                            with gr.Tab(translate('Resolution', language)):
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

                with gr.Tab(translate('Database', language)) as database_tab:
                    database = gr.Dataframe(
                        value=None,
                        label=translate('Database', language),
                        type='pandas',
                    )

                with gr.Tab(translate('Buffer', language)) as buffer_tab:
                    buffer_metadata_df = gr.Dataframe(
                        value=None,
                        label=translate('Buffer information', language),
                        type='pandas',
                        row_count=(1, 'fixed'),
                    )
                    buffer_df = gr.Dataframe(
                        value=None,
                        label=translate('Buffer', language),
                        type='pandas',
                        row_count=(20, 'fixed'),
                    )

            # ========================================= Functions ========================================= #

            def partial(func, **preset_kwargs):
                @wraps(func)
                def wrapper(*args, **kwargs):
                    kwargs.update(preset_kwargs)
                    return func(*args, **kwargs)
                return wrapper

            def get_new_img_key(dataset):
                r"""
                Get the image key that should be selected if the showing dataset is changed by `dset`. Doesn't depend on what the current dataset is.
                """
                pre_idx = state.selected.index
                if pre_idx is not None and pre_idx < len(dataset):
                    new_img_key = dataset[pre_idx]['image_key']
                else:
                    new_img_key = None
                return new_img_key

            def correct_page_index(dataset: UISubset, page_index):
                r"""
                Correct the chunk index of `dset` to `chunk_index`
                """
                if page_index is None or page_index == 0:
                    page_index = 1
                else:
                    page_index = min(max(page_index, 1), dataset.num_pages)
                return page_index

            def change_current_subset(newset: UISubset, sorting_methods=None, reverse=False):
                r"""
                Change the current dataset to `dset`
                """
                # pre-sorting
                # if sorting_methods is not None and len(sorting_methods) > 0:
                #     if language != 'en':
                #         sorting_methods = translate(sorting_methods, 'en')
                #     sorting_methods = [method.replace(' ', '_') for method in sorting_methods]

                #     extra_kwargs = {}
                #     selected_img_key = uiset.selected.key
                #     if selected_img_key is not None:
                #         target = uiset[selected_img_key]
                #         extra_kwargs['target'] = target.perceptual_hash

                #     sorting_keys = []
                #     for method in sorting_methods:
                #         func = SORTING_METHODS[method]
                #         func_params = inspect.signature(func).parameters
                #         func_param_names = list(func_params.keys())
                #         sorting_keys.append((func, {k: v for k, v in extra_kwargs.items() if k in func_param_names}))

                univset.change_curset(newset)

                # post-sorting
                # if sorting_methods is not None and len(sorting_methods) > 0:
                #     subset.sort(key=lambda item: tuple(func(item[1], **kwargs) for func, kwargs in sorting_keys), reverse=reverse)

            def update_showcase_from_dataset(showset: UISubset = None, new_page_index=1):
                r"""
                Convert `showset` to gallery value and image key that should be selected
                """
                if issubclass(type(showset), Dataset):
                    new_page_index = correct_page_index(showset, new_page_index)
                    page = showset.page(new_page_index - 1)
                # elif isinstance(showset, Dataset):
                #     page = showset
                #     new_page_index = 1
                new_img_key = get_new_img_key(page)
                gallery = convert_dataset_to_gallery(page)
                return {
                    showcase: gallery,
                    dataset_metadata_df: convert_dataset_to_statistic_dataframe(showset),
                    cur_img_key: new_img_key,
                    cur_page_number: gr.update(value=new_page_index, label=f"{translate('Page', language)} {new_page_index}/{univset.curset.num_pages}"),
                }

            def show_database(dataset: UISubset = None, new_page_index=1):
                r"""
                Convert `dataset` to dataframe
                """
                new_page_index = correct_page_index(dataset, new_page_index)
                page = dataset.page(new_page_index - 1) if isinstance(dataset, UISubset) else dataset
                df = page.df()
                return {
                    database: df,
                    cur_page_number: gr.update(value=new_page_index, label=f"{translate('Chunk', language)} {new_page_index}/{dataset.num_pages}"),
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

            # def load_source(source):
            #     if source is None or len(source) == 0:
            #         return {log_box: 'empty source'}
            #     source = [os.path.abspath(src) for src in source]
            #     for src in source:
            #         if not os.path.exists(src):
            #             return {log_box: f"source `{src}` not found"}
            #         elif not (os.path.isdir(src) or src.endswith(('.csv', '.json'))):
            #             return {f"source `{src}` is not a file or directory"}

            #     nonlocal uiset, buffer, sample_history, subset, tag_table, tag_feature_table

            #     univargs.source = source
            #     uiset, buffer, sample_history, subset, _, _, tag_table, tag_feature_table = init_everything()
            #     return {
            #         source_file: univargs.source,
            #         category_selector: gr.update(choices=uiset.get_categories(), value=None),
            #         showcase: [(v.image_path, k) for k, v in subset.chunk(0).items()],
            #         cur_page_number: gr.update(value=1, label=f"{translate('Chunk', language)} 1/{subset.num_chunks}"),
            #         query_include_tags: gr.update(choices=None),
            #         query_exclude_tags: gr.update(choices=None),
            #         log_box: f"source reloaded: `{source}`",
            #     }

            # load_source_btn.click(
            #     fn=load_source,
            #     inputs=[source_file],
            #     outputs=[source_file, category_selector, showcase, cur_page_number, query_include_tags, query_exclude_tags, log_box],
            #     concurrency_limit=1,
            # )

            # ========================================= Subset key selector ========================================= #

            def load_subset_from_dataset(newset: UISubset = None, new_page_index=1, sorting_methods=None, reverse=False):
                r"""
                Change current dataset to another dataset `newset` and show its page
                """
                if newset is None:
                    newset = univset.curset
                change_current_subset(newset, sorting_methods=sorting_methods, reverse=reverse)
                if ui_main_tab.tab is tagging_tab:
                    res = update_showcase_from_dataset(newset, new_page_index)
                elif ui_main_tab.tab is database_tab:
                    res = show_database(newset, new_page_index)
                return res

            def load_subset_from_categories(categories, sorting_methods=None, reverse=False):
                r"""
                Change current dataset to another dataset with category `category` and show its chunk
                """
                logger.print(f"loading subset from categories: {categories}")
                if not categories:
                    catset = univset.fullset
                elif isinstance(rootset := univset.rootset, SQLite3Dataset):
                    if 'category' in rootset.header:
                        catset = UISubset(rootset.select_in('category', categories) if len(categories) > 1 else rootset.select_is('category', categories[0]), univset)
                    else:
                        catset = rootset.select_like('image_path', f"%{categories[0]}%")
                        catset = catset.subset(condition=lambda img_md: os.path.basename(os.path.dirname(img_md['image_path'])) in set(categories))
                        catset = UISubset(catset, univset)
                else:
                    catset = univset.subset(condition=lambda img_md: img_md.get('category', None) or os.path.basename(os.path.dirname(img_md['image_path'])) in set(categories))
                return load_subset_from_dataset(catset, new_page_index=1, sorting_methods=sorting_methods, reverse=reverse)

            dataset_change_inputs = [cur_page_number, sorting_methods_dropdown, sorting_reverse_checkbox]
            dataset_change_listeners = [showcase, dataset_metadata_df, cur_img_key, database, cur_page_number, category_selector, log_box]

            # category_selector.blur(
            #     fn=change_to_category,
            #     inputs=[category_selector],
            #     outputs=selector_change_outputs,
            #     show_progress=True,
            #     trigger_mode='multiple',
            #     concurrency_limit=1,
            # )

            reload_category_btn.click(
                fn=load_subset_from_categories,
                inputs=[category_selector, sorting_methods_dropdown, sorting_reverse_checkbox],
                outputs=dataset_change_listeners,
                trigger_mode='multiple',
                concurrency_limit=1,
                scroll_to_output=True,
            )

            # set_category_btn.click(
            #     fn=lambda img_key, *args: {**change_to_categories([subset[img_key].category], *args), category_selector: [subset[img_key].category]},
            #     inputs=[cur_img_key, sorting_methods_dropdown, sorting_reverse_checkbox],
            #     outputs=dataset_change_listeners,
            #     show_progress=True,
            #     trigger_mode='always_last',
            #     concurrency_limit=1,
            # )

            # reload_subset_btn.click(  # same as above
            #     fn=lambda *args: change_to_dataset(subset, *args),
            #     inputs=dataset_change_inputs,
            #     outputs=dataset_change_listeners,
            #     show_progress=True,
            #     concurrency_limit=1,
            # )

            reload_sort_btn.click(
                fn=lambda *args: load_subset_from_dataset(univset.curset, *args),
                inputs=dataset_change_inputs,
                outputs=dataset_change_listeners,
                show_progress=True,
                concurrency_limit=1,
            )

            tagging_tab.select(
                fn=change_activating_tab(ui_main_tab, tagging_tab, lambda *args: load_subset_from_dataset(univset.curset, *args)),
                inputs=dataset_change_inputs,
                outputs=dataset_change_listeners,
                show_progress=True,
                concurrency_limit=1,
            )

            database_tab.select(
                fn=change_activating_tab(ui_main_tab, database_tab, lambda *args: load_subset_from_dataset(univset.curset, *args)),
                inputs=dataset_change_inputs,
                outputs=dataset_change_listeners,
                show_progress=True,
                concurrency_limit=1,
            )

            cur_page_number.submit(
                fn=lambda page_index: load_subset_from_dataset(univset.curset, page_index),
                inputs=[cur_page_number],  # no need to sort
                outputs=dataset_change_listeners,
                show_progress=True,
                concurrency_limit=1,
            )

            load_pre_page_btn.click(
                fn=lambda page_index: load_subset_from_dataset(univset.curset, page_index - 1),
                inputs=[cur_page_number],  # no need to sort
                outputs=dataset_change_listeners,
                show_progress=True,
                concurrency_limit=1,
            )

            load_next_page_btn.click(
                fn=lambda page_index: load_subset_from_dataset(univset.curset, page_index + 1),
                inputs=[cur_page_number],  # no need to sort
                outputs=dataset_change_listeners,
                show_progress=True,
                concurrency_limit=1,
            )

            # ========================================= Showcase ========================================= #

            def select_img_key(selected: gr.SelectData):
                if selected is None:
                    return None, None
                state.selected.select(selected)
                return state.selected.key

            showcase.select(
                fn=select_img_key,
                outputs=[cur_img_key],
                concurrency_limit=1,
            )

            cur_img_key_change_listeners = [image_path, resolution, caption, caption_metadata_df, description, other_metadata_df, positive_prompt, negative_prompt, gen_params_df, log_box]
            BASE_MD_KEYS = ('image_key', 'image_path', 'caption', 'description')
            CAPTION_MD_KEYS = tagging.TAG_TYPES
            OTHER_MD_KEYS = [key for key in univset.header if key not in (*BASE_MD_KEYS, *CAPTION_MD_KEYS)]

            def get_caption(img_key):
                if img_key is None or img_key == '':
                    return None
                img_md = univset[img_key]
                caption = img_md.get('caption', None)
                return str(caption) if caption is not None else None

            def get_metadata_df(img_key, keys):
                if render == 'partial' and ui_data_tab.tab is not metadata_tab:
                    return None
                if img_key is None or img_key == '' or img_key not in univset:
                    return None
                data = [{translate(key.replace('_', ' ').title(), language): univset[img_key].get(key, None) for key in keys}]
                data = [{k: v if v is None or isinstance(v, (int, float, str)) else str(v) for k, v in row.items()} for row in data]
                df = pandas.DataFrame(data=data, columns=data[0].keys())
                return df

            def get_nl_caption(img_key):
                if not img_key:
                    return None
                img_md = univset[img_key]
                return img_md.get('description', None)

            def get_original_size(img_key):
                if not img_key:
                    return None
                img_md = univset[img_key]
                original_size = img_md.get('original_size', None)
                if not original_size:
                    img_path = img_md.get('image_path', None)
                    if not os.path.isfile(img_path):
                        return None
                    try:
                        image = Image.open(img_path)
                    except:
                        return None
                    original_size = image.size
                    original_size = f"{original_size[0]}x{original_size[1]}"
                return original_size

            # TODO
            def get_gen_info(img_key):
                img_md = univset[img_key]
                img_path = img_md['image_path']
                if not os.path.isfile(img_path):
                    return None, None, None
                try:
                    image = Image.open(img_path)
                except:
                    return None, None, None
                metadata_dict = image_utils.parse_gen_info(image.info)
                pos_pmt = metadata_dict.pop('Positive prompt', None)
                neg_pmt = metadata_dict.pop('Negative prompt', None)
                # single row pandas dataframe for params
                params_df = pandas.DataFrame(data=[metadata_dict], columns=list(metadata_dict.keys())) if len(metadata_dict) > 0 else None
                return pos_pmt, neg_pmt, params_df

            def track_img_key(img_key):
                if img_key is None or img_key == '':  # no image key selected
                    return {k: None for k in cur_img_key_change_listeners}
                if img_key != state.selected.key:  # fix
                    state.selected.select((state.selected.index, img_key))
                img_md = univset[img_key]
                img_path = img_md.get('image_path', None)
                reso = get_original_size(img_key)
                if render == 'full':
                    pos_pmt, neg_pmt, param_df = get_gen_info(img_key)
                    res = {
                        image_path: img_path,
                        caption: get_caption(img_key),
                        caption_metadata_df: get_metadata_df(img_key, keys=CAPTION_MD_KEYS),
                        other_metadata_df: get_metadata_df(img_key, keys=OTHER_MD_KEYS),
                        description: get_nl_caption(img_key),
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

            cur_img_key.change(
                fn=track_img_key,
                inputs=[cur_img_key],
                outputs=cur_img_key_change_listeners,
                concurrency_limit=1,
            )

            if render == 'partial':
                caption_tab.select(
                    fn=change_activating_tab(ui_data_tab, caption_tab, get_caption),
                    inputs=[cur_img_key],
                    outputs=[caption],
                    concurrency_limit=1,
                )

                metadata_tab.select(
                    fn=lambda img_key: (
                        change_activating_tab(ui_data_tab, metadata_tab, get_metadata_df)(img_key, keys=CAPTION_MD_KEYS),
                        get_metadata_df(img_key, keys=OTHER_MD_KEYS)
                    ),
                    inputs=[cur_img_key],
                    outputs=[caption_metadata_df, other_metadata_df],
                    trigger_mode='always_last',
                    concurrency_limit=1,
                )

                description_tab.select(
                    fn=change_activating_tab(ui_data_tab, description_tab, get_nl_caption),
                    inputs=[cur_img_key],
                    outputs=[description],
                    concurrency_limit=1,
                )

                gen_info_tab.select(
                    fn=change_activating_tab(ui_data_tab, gen_info_tab, get_gen_info),
                    inputs=[cur_img_key],
                    outputs=[positive_prompt, negative_prompt, gen_params_df],
                    concurrency_limit=1,
                )

            caption.change(
                fn=lambda img_key: get_metadata_df(img_key, keys=CAPTION_MD_KEYS),
                inputs=[cur_img_key],
                outputs=[caption_metadata_df],
                trigger_mode='always_last',
                concurrency_limit=1,
            )

            # ========================================= Below showcase ========================================= #

            # def remove_image(img_key, chunk_index):
            #     if img_key is None or img_key == '':
            #         return {log_box: f"empty image key"}
            #     uiset.remove(img_key)
            #     if subset is not uiset and img_key in subset:
            #         del subset[img_key]  # remove from current dataset
            #     return change_to_dataset(new_page_index=chunk_index)

            # remove_image_btn.click(
            #     fn=remove_image,
            #     inputs=[cur_img_key, cur_page_number],
            #     outputs=dataset_change_listeners,
            #     concurrency_limit=1,
            # )

            # def show_i_th_sample(index):
            #     if len(sample_history) == 0:
            #         return {
            #             log_box: f"empty sample history",
            #         }
            #     new_img_key = sample_history.select(index)
            #     return {
            #         showcase: dataset_to_gallery(Dataset(uiset[new_img_key])),
            #         cur_img_key: new_img_key,
            #     }

            # load_pre_hist_btn.click(
            #     fn=lambda: show_i_th_sample(sample_history.index - 1) if sample_history.index is not None else {log_box: f"empty sample history"},
            #     inputs=[],
            #     outputs=dataset_change_listeners,
            #     concurrency_limit=1,
            # )

            # load_next_hist_btn.click(
            #     fn=lambda: show_i_th_sample(sample_history.index + 1) if sample_history.index is not None else {log_box: f"empty sample history"},
            #     inputs=[],
            #     outputs=dataset_change_listeners,
            #     concurrency_limit=1,
            # )

            # ========================================= Base Tagging Buttons ========================================= #

            # def random_sample(n=1):
            #     sampleset = subset  # sample from current dataset
            #     if len(sampleset) == 0:
            #         return {log_box: f"empty subset"}
            #     sampleset = sampleset.make_subset(condition=lambda img_info: img_info.key not in sample_history)
            #     if len(sampleset) == 0:
            #         return {log_box: f"no more image to sample"}
            #     samples: Dataset = sampleset.sample(n=n, randomly=True)
            #     for sample in samples.keys():
            #         sample_history.add(sample)
            #     new_img_key = sample_history.select(len(sample_history) - 1)
            #     return {
            #         showcase: dataset_to_gallery(Dataset(uiset[new_img_key])),
            #         cur_img_key: new_img_key,
            #         cur_page_number: gr.update(value=1, label=f"{translate('Chunk', language)} 1/{subset.num_chunks}"),
            #     }

            # random_btn.click(
            #     fn=random_sample,
            #     inputs=[],
            #     outputs=dataset_change_listeners,
            #     concurrency_limit=1,
            # )

            def data_edition_handler(func: Callable[[Dict, Tuple[Any, ...], Dict[str, Any]], Caption]) -> Tuple[str, str]:
                funcname = func.__name__

                def wrapper(img_key, opts, *args, progress: gr.Progress = gr.Progress(track_tqdm=True), **kwargs):
                    nonlocal funcname, cpu_max_workers
                    proc_func_log_name = funcname.replace('_', ' ')

                    if language != 'en':
                        opts = translate(opts, 'en')
                    opts = [pm.lower() for pm in opts]
                    do_batch = ('batch' in opts) and (func is not write_caption)
                    do_progress = 'progress' in opts
                    extra_kwargs = dict(
                        do_append='append' in opts,
                        do_regex='regex' in opts,
                    )
                    # filter out extra kwargs
                    funcparams = list(inspect.signature(func).parameters.keys())
                    extra_kwargs = {k: v for k, v in extra_kwargs.items() if k in funcparams}

                    if img_key is None or img_key == '':
                        if not do_batch:
                            return {log_box: f"{proc_func_log_name}: empty image key"}
                    else:
                        img_key = Path(img_key).stem if Path(img_key).suffix in IMAGE_EXTS else img_key

                    def edit_batch(batch, *args, **kwargs):
                        batch = deepcopy(batch)
                        if not isinstance(batch, list):  # single image
                            img_md = batch
                            if not os.path.isfile(img_md['image_path']):
                                return []
                            new_img_md = func(img_md, *args, **extra_kwargs, **kwargs)
                            return new_img_md if isinstance(new_img_md, Iterable) else [new_img_md]
                        else:
                            batch = [img_md for img_md in batch if os.path.isfile(img_md['image_path'])]
                            new_batch = func([img_md for img_md in batch], *args, **extra_kwargs, **kwargs)
                            return new_batch

                    results = []
                    if do_batch:
                        logger.print(f"batch processing: {proc_func_log_name}")
                        editset = univset.curset
                        if func in (ws_scoring, wd_tagging):
                            batch_size, args = args[0], args[1:]  # first arg is batch size
                            values = list(editset.values())
                            batches = [values[i:i + batch_size] for i in range(0, len(editset), batch_size)]
                        else:
                            batch_size = 1
                            batches = list(editset.values())
                        desc = f'[{proc_func_log_name}] batch processing'
                        pbar = logger.tqdm(total=len(batches), desc=desc)
                        edit_batch = logging.track_tqdm(pbar)(edit_batch)
                        if do_progress:
                            edit_batch = track_progress(progress, desc=desc, total=len(batches))(edit_batch)

                        if cpu_max_workers == 1:
                            for batch in batches:
                                res = edit_batch(batch, *args, **kwargs)
                                if isinstance(res, dict):
                                    results.append(res)
                                elif isinstance(res, list):
                                    results.extend(res)
                        else:
                            with ThreadPoolExecutor(max_workers=cpu_max_workers) as executor:
                                futures = [executor.submit(edit_batch, batch, *args, **kwargs) for batch in batches]
                                try:
                                    wait(futures)
                                    for future in futures:
                                        res = future.result()
                                        if isinstance(res, dict):
                                            results.append(res)
                                        elif isinstance(res, list):
                                            results.extend(res)
                                except (gr.CancelledError, KeyboardInterrupt):
                                    for future in futures:
                                        future.cancel()
                                    raise
                        pbar.close()

                    else:
                        if func in (ws_scoring, wd_tagging):
                            args = args[1:]
                            batch = [univset[img_key]]
                        else:
                            batch = univset[img_key]
                        res = edit_batch(batch, *args, **kwargs)
                        if isinstance(res, dict):
                            results.append(res)
                        elif isinstance(res, list):
                            results.extend(res)
                        else:
                            raise ValueError(f"invalid return type: {type(res)}")

                    # write to dataset
                    for res in results:
                        if res is not None:
                            imk = res.get('image_key', os.path.basename(os.path.splitext(res['image_path'])[0]))
                            if func not in (undo, redo):
                                if func == write_caption and res['caption'] == univset[img_key]['caption']:
                                    continue
                                if imk not in buffer:
                                    buffer.do(imk, univset[imk])
                                buffer.do(imk, res)
                            univset[imk] = res
                            # univset.set(img_key, res)

                    if any(results):
                        ret = track_img_key(img_key)
                        if not img_key:
                            ret.update({log_box: f"{proc_func_log_name}: batch"})
                        else:
                            ret.update({log_box: f"{proc_func_log_name}: `{img_key}`"})
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

            def write_caption(img_md, caption: str):
                if caption:
                    img_md['caption'] = str(caption)
                return img_md

            caption.blur(
                fn=data_edition_handler(write_caption),
                inputs=[image_path, general_edit_opts, caption],
                outputs=cur_img_key_change_listeners,
                cancels=cancel_event,
                concurrency_limit=1,
            )

            def undo(img_md):
                return buffer.undo(img_md['image_key']) or img_md

            undo_btn.click(
                fn=data_edition_handler(undo),
                inputs=[cur_img_key, general_edit_opts],
                outputs=cur_img_key_change_listeners,
                concurrency_limit=1,
            )

            def redo(img_md):
                return buffer.redo(img_md['image_key']) or img_md

            redo_btn.click(
                fn=data_edition_handler(redo),
                inputs=[cur_img_key, general_edit_opts],
                outputs=cur_img_key_change_listeners,
                concurrency_limit=1,
            )

            def save_to_disk(fp, progress: gr.Progress = gr.Progress(track_tqdm=True)):
                rootset = univset.root

                # patch for DirectoryDataset
                if isinstance(rootset, ToDiskMixin):
                    os.makedirs(os.path.dirname(fp), exist_ok=True)
                    rootset.commit() if fp else rootset.dump(fp)
                    return f"saved: {rootset.path}"
                elif fp:
                    ext = os.path.splitext(fp)[1]
                    if ext not in ('.json', '.csv', '.sqlite3', '.db'):
                        raise gr.Error(f"unsupported extension: {ext}")
                        return {log_box: f"unsupported extension: {ext}"}
                    os.makedirs(os.path.dirname(fp), exist_ok=True)
                    AutoDataset.dump(rootset, fp)
                    return f"saved: {fp}"
                elif rootset.__class__.__name__ == "DirectoryDataset":
                    def save_one(img_md):
                        img_path = Path(img_md['image_path'])
                        txt_path = img_path.with_suffix('.txt')
                        txt_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(txt_path, 'w', encoding='utf-8') as f:
                            f.write(str(img_md['caption']))
                    save_one = track_progress(progress, desc=f"[{save_to_disk.__name__}]", total=len(buffer))(save_one)
                    for img_md in buffer.latests().values():
                        save_one(img_md)
                    return f"saved"
                else:
                    raise gr.Error(f"failed to save dataset: Dataset type not supported or invalid save path")
                    return {log_box: f"failed to save dataset: Dataset type not supported or invalid save path"}

            save_btn.click(
                fn=save_to_disk,
                inputs=[save_path],
                outputs=[log_box],
                concurrency_limit=1,
            )

            # ========================================= Quick Tagging ========================================= #

            def format_tag(img_md, tag):
                r"""
                %dir%: category (directory) name
                %cat%: same as %dirname%
                %stem%: filename
                """
                img_path = img_md['image_path']
                category = os.path.basename(os.path.dirname(img_path))
                stem = os.path.basename(os.path.splitext(img_path)[0])
                tag = tag.replace('%dir%', category)
                tag = tag.replace('%dirname%', category)
                tag = tag.replace('%cat%', category)
                tag = tag.replace('%category%', category)
                tag = tag.replace('%stem%', stem)
                tag = tag.replace('%filename%', stem)
                return tag

            def contains_fmt_tag(tags):
                return any(re.search(r'%.*%', tag) for tag in tags)

            def add_tags(img_md, tags, do_append):
                caption = Caption(img_md.get('caption', None))
                if isinstance(tags, str):
                    tags = [tags]
                tags = [format_tag(img_md, tag) for tag in tags]
                if contains_fmt_tag(tags):
                    raise gr.Error(f"invalid tag format: {tags}")
                if do_append:
                    caption += tags
                else:
                    caption = tags + caption
                img_md['caption'] = caption.text
                return img_md

            def remove_tags(img_md, tags, do_regex):
                caption = img_md.get('caption', None)
                if caption is None:
                    return img_md
                caption = Caption(caption)
                if isinstance(tags, str):
                    tags = [tags]

                tags = [format_tag(img_md, tag) for tag in tags]

                if contains_fmt_tag(tags):
                    raise gr.Error(f"invalid tag format: {tags}")
                if do_regex:
                    try:
                        tags = [re.compile(tag) for tag in tags]
                    except re.error as e:
                        raise gr.Error(f"invalid regex: {e}")
                caption -= tags
                img_md['caption'] = caption.text
                return img_md

            # ========================================= Custom Tagging ========================================= #

            for add_tag_btn, tag_selector, remove_tag_btn in zip(add_tag_btns, tag_selectors, remove_tag_btns):
                add_tag_btn.click(
                    fn=data_edition_handler(add_tags),
                    inputs=[image_path, general_edit_opts, tag_selector],
                    outputs=cur_img_key_change_listeners,
                    concurrency_limit=1,
                )
                remove_tag_btn.click(
                    fn=data_edition_handler(remove_tags),
                    inputs=[image_path, general_edit_opts, tag_selector],
                    outputs=cur_img_key_change_listeners,
                    concurrency_limit=1,
                )

            def replace_tag(img_md, old, new, match_tag, do_regex):
                caption = img_md.get('caption', None)
                if caption is None:
                    return img_md
                caption = Caption(caption)
                if do_regex:
                    try:
                        old = re.compile(old)
                    except re.error as e:
                        raise gr.Error(f"invalid regex `{old}`: {e}")
                    try:
                        caption[old] = new
                    except re.error as e:
                        raise gr.Error(f"regex error: {e}")
                else:
                    if match_tag:
                        caption[old] = new
                    else:
                        caption = Caption(caption.text.replace(old, new))
                img_md['caption'] = caption.text
                return img_md

            for replace_tag_btn, old_tag_selector, new_tag_selector in zip(replace_tag_btns, old_tag_selectors, new_tag_selectors):
                replace_tag_btn.click(
                    fn=data_edition_handler(replace_tag),
                    inputs=[image_path, general_edit_opts, old_tag_selector, new_tag_selector, match_tag_checkbox],
                    outputs=cur_img_key_change_listeners,
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
            def read_attrs(img_md, types):
                if language != 'en':
                    types = [translate(t, 'en') for t in types]
                attrs_dict = data_utils.read_attrs(img_md['image_path'], types=types)
                if not attrs_dict:
                    return None
                img_md.update(attrs_dict)
                return img_md

            read_attrs_btn.click(
                fn=data_edition_handler(read_attrs),
                inputs=[cur_img_key, general_edit_opts, read_attrs_types],
                outputs=cur_img_key_change_listeners,
                concurrency_limit=1,
            )

            def parse_caption_attrs(img_md):
                if img_md.get('caption', None) is None:
                    return img_md
                caption = Caption(img_md['caption']).parsed()
                img_md['caption'] = caption.text
                attrs = caption.attrs
                attrs.pop('tags')
                for attr, value in attrs.items():
                    img_md[attr] = caption.sep.join(value)
                return img_md

            parse_caption_attrs_btn.click(
                fn=data_edition_handler(parse_caption_attrs),
                inputs=[cur_img_key, general_edit_opts],
                outputs=cur_img_key_change_listeners,
                concurrency_limit=1,
            )

            def sort_caption(img_md):
                caption = img_md.get('caption', None)
                if caption is None:
                    return img_md
                img_md['caption'] = Caption(caption).sorted().text
                return img_md

            sort_caption_btn.click(
                fn=data_edition_handler(sort_caption),
                inputs=[cur_img_key, general_edit_opts],
                outputs=cur_img_key_change_listeners,
                concurrency_limit=1,
            )

            def formalize_caption(img_md, formats):
                caption = img_md.get('caption', None)
                if caption is None:
                    return img_md
                if isinstance(formats, str):
                    formats = [formats]
                if language != 'en':
                    formats = [translate(fmt, 'en') for fmt in formats]
                caption = Caption(caption)
                for fmt in formats:
                    caption.format(FORMAT_PRESETS[fmt])
                img_md['caption'] = caption.text
                return img_md

            formalize_caption_btn.click(
                fn=data_edition_handler(formalize_caption),
                inputs=[cur_img_key, general_edit_opts, formalize_caption_dropdown],
                outputs=cur_img_key_change_listeners,
                concurrency_limit=1,
            )

            def deduplicate_caption(img_md):
                caption = img_md.get('caption', None)
                if caption is None:
                    return img_md
                img_md['caption'] = Caption(caption).deduplicated().text
                return img_md

            deduplicate_caption_btn.click(
                fn=data_edition_handler(deduplicate_caption),
                inputs=[cur_img_key, general_edit_opts],
                outputs=cur_img_key_change_listeners,
                concurrency_limit=1,
            )

            def deoverlap_caption(img_md):
                caption = img_md.get('caption', None)
                if caption is None:
                    return img_md
                img_md['caption'] = Caption(caption).deoverlapped().text
                return img_md

            deoverlap_caption_btn.click(
                fn=data_edition_handler(deoverlap_caption),
                inputs=[cur_img_key, general_edit_opts],
                outputs=cur_img_key_change_listeners,
                concurrency_limit=1,
            )

            def decharacterize_caption(img_md, feature_type, freq_thres):
                caption = img_md.get('caption', None)
                if caption is None:
                    return None
                img_md['caption'] = Caption(caption).decharacterized(feature_type=feature_type, freq_thres=freq_thres).text
                return img_md

            decharacterize_caption_btn.click(
                fn=data_edition_handler(decharacterize_caption),
                inputs=[cur_img_key, general_edit_opts, dechar_feature_type, dechar_freq_thres],
                outputs=cur_img_key_change_listeners,
                concurrency_limit=1,
            )

            # ========================================= WD ========================================= #

            def wd_tagging(batch: List, model_repo_or_path, general_threshold, character_threshold, overwrite_mode):
                nonlocal waifu_tagger
                if language != 'en':
                    overwrite_mode = translate(overwrite_mode, 'en')
                if not isinstance(batch, list):
                    batch = [batch]
                if overwrite_mode == 'ignore':
                    batch = [img_md for img_md in batch if img_md['caption'] is None]
                    if len(batch) == 0:
                        return []

                if waifu_tagger is None or (waifu_tagger and waifu_tagger.model_name != model_repo_or_path):
                    from waifuset.components.waifu_tagger.predict import WaifuTagger, repo2path
                    model_path, label_path = repo2path(model_repo_or_path)
                    waifu_tagger = WaifuTagger(model_path=model_path, label_path=label_path, verbose=True)
                    waifu_tagger.model_name = model_repo_or_path

                batch_images = [Image.open(img_md['image_path']) for img_md in batch]
                batch_pred_tags = waifu_tagger(batch_images, general_threshold=general_threshold, character_threshold=character_threshold)
                for img_md, pred_tags in zip(batch, batch_pred_tags):
                    if overwrite_mode == 'overwrite' or overwrite_mode == 'ignore':
                        tags = pred_tags
                    elif overwrite_mode == 'append':
                        tags = img_md.get('caption', '').split(', ') + pred_tags
                    elif overwrite_mode == 'prepend':
                        tags = pred_tags + img_md.get('caption', []).split(', ')
                    else:
                        raise ValueError(f"invalid os_mode: {overwrite_mode}")
                    img_md['caption'] = ', '.join(tags)
                return batch

            wd_run_btn.click(
                fn=data_edition_handler(wd_tagging),
                inputs=[cur_img_key, general_edit_opts, wd_batch_size, wd_model, wd_general_threshold, wd_character_threshold, wd_overwrite_mode],
                outputs=cur_img_key_change_listeners,
                concurrency_limit=1,
            )

            # ========================================= WS ========================================= #
            def ws_scoring(batch, model_repo_or_path, overwrite_mode) -> List[Dict]:
                if language != 'en':
                    overwrite_mode = translate(overwrite_mode, 'en')

                nonlocal waifu_scorer
                if waifu_scorer is None or (waifu_scorer and waifu_scorer.model_name != model_repo_or_path):
                    import torch
                    from waifuset.components.waifu_scorer.predict import WaifuScorer, repo2path
                    model_path = repo2path(model_repo_or_path)
                    waifu_scorer = WaifuScorer(model_path=model_path, device='cuda' if torch.cuda.is_available() else 'cpu', verbose=True)
                    waifu_scorer.model_name = model_repo_or_path

                if not isinstance(batch, list):
                    batch = [batch]
                batch = [img_md for img_md in batch if os.path.isfile(img_md['image_path']) and not (overwrite_mode == 'ignore' and img_md.get('aesthetic_score', None) is not None)]
                if len(batch) == 0:
                    return []
                images = [Image.open(img_md['image_path']) for img_md in batch]
                pred_scores = waifu_scorer(images)
                if isinstance(pred_scores, float):  # single output
                    pred_scores = [pred_scores]
                for i, img_md in enumerate(batch):
                    img_md['aesthetic_score'] = pred_scores[i]
                return batch

            ws_run_btn.click(
                fn=data_edition_handler(ws_scoring),
                inputs=[cur_img_key, general_edit_opts, ws_batch_size, ws_model, ws_overwrite_mode],
                outputs=cur_img_key_change_listeners,
                concurrency_limit=1,
                cancels=cancel_event,
            )

            def set_aesthetic_score(img_md, score):
                img_md['aesthetic_score'] = score
                return img_md

            ws_delabel_btn.click(
                fn=data_edition_handler(partial(set_aesthetic_score, score=None)),
                inputs=[cur_img_key, general_edit_opts],
                outputs=cur_img_key_change_listeners,
                concurrency_limit=1,
            )

            # ========================================= Perceptual Hash ========================================= #
            def get_perceptual_hash(img_md, os_mode):
                if language != 'en':
                    os_mode = translate(os_mode, 'en')
                try:
                    import imagehash
                except ImportError:
                    raise gr.Error("imagehash package is not installed!")
                orig_p_hash = img_md.get('perceptual_hash', None)
                if orig_p_hash is not None and os_mode == 'ignore':
                    return img_md
                image = Image.open(img_md['image_path'])
                p_hash = imagehash.phash(image)
                img_md['perceptual_hash'] = p_hash
                return img_md

            hasher_run_btn.click(
                fn=data_edition_handler(get_perceptual_hash),
                inputs=[cur_img_key, general_edit_opts, hasher_overwrite_mode],
                outputs=cur_img_key_change_listeners,
                concurrency_limit=1,
            )

            def set_perceptual_hash(img_md, p_hash):
                img_md['perceptual_hash'] = p_hash
                return img_md

            hasher_dehash_btn.click(
                fn=data_edition_handler(partial(set_perceptual_hash, p_hash=None)),
                inputs=[cur_img_key, general_edit_opts],
                outputs=cur_img_key_change_listeners,
                concurrency_limit=1,
            )

            # ========================================= Open file folder ========================================= #

            def open_file_folder(path: str):
                print(f"Open {path}")
                if path is None or path == "":
                    return
                command = f'explorer /select,"{path}"'
                os.system(command)

            open_folder_btn.click(
                fn=open_file_folder,
                inputs=[image_path],
                outputs=[],
                concurrency_limit=1,
            )

            # ========================================= Buffer ========================================= #

            def show_buffer():
                bufferset = UISubset(buffer.latests(), host=univset)
                return {
                    buffer_df: bufferset.df(),
                    buffer_metadata_df: convert_dataset_to_statistic_dataframe(bufferset),
                }

            buffer_tab.select(
                fn=show_buffer,
                outputs=[buffer_df, buffer_metadata_df],
            )

            # ========================================= Query ========================================= #

            query_base_inputs = [query_opts, sorting_methods_dropdown, sorting_reverse_checkbox]

            def query_handler(func: Callable[[Tuple[Any, ...], Dict[str, Any]], UISubset]):
                funcname = func.__name__

                def wrapper(opts, sorting_methods=None, reverse=False, *args, **kwargs):
                    # parse opts as extra kwargs
                    if language != 'en':
                        opts = translate(opts, 'en')
                    opts = [opt.lower() for opt in opts]
                    do_subset = 'subset' in opts
                    do_complement = 'complement' in opts
                    extra_kwargs = dict(
                        do_regex='regex' in opts,
                    )
                    # filter out extra kwargs
                    funcparams = list(inspect.signature(func).parameters.keys())
                    extra_kwargs = {k: v for k, v in extra_kwargs.items() if k in funcparams}
                    queryset = univset.curset if do_subset else univset.fullset
                    if queryset is None or len(queryset) == 0:
                        return {log_box: f"empty query range"}

                    # QUERY
                    resset = func(queryset, *args, **extra_kwargs, **kwargs)
                    if resset is None:
                        return {log_box: f"invalid query result"}
                    if do_complement:
                        resset = UISubset([img_key for img_key in queryset.keys() if img_key not in resset], host=univset)
                    logger.print(f"`{funcname}` found: {len(resset)}/{len(queryset)}")
                    result = load_subset_from_dataset(resset, sorting_methods=sorting_methods, reverse=reverse)
                    result.update({log_box: f"[{funcname}] found: {len(resset)}/{len(queryset)}"})
                    return result
                return wrapper

            def query_attr(queryset: UISubset, attr, pattern, do_regex=False):
                if not attr:
                    return None
                if not pattern:
                    if isinstance(rootset := queryset.root, SQLite3Dataset):
                        ressset = [row[0] for row in rootset.table.select_is(attr, None)]
                    else:
                        ressset = rootset.subkeys(lambda img_md: img_md.get(attr, '') is None)
                    return UISubset(ressset, host=univset)
                if do_regex:
                    def match(attr, pattern):
                        return re.match(pattern, attr) is not None
                else:
                    def match(attr, pattern):
                        return attr == pattern
                if isinstance(rootset := queryset.root, SQLite3Dataset):
                    if do_regex:
                        ressset = [row[0] for row in rootset.table.select_func(match, '$' + attr + '$', pattern)]
                    else:
                        ressset = [row[0] for row in rootset.table.select_is(attr, pattern)]
                else:
                    ressset = rootset.subkeys(lambda img_md: match(pattern, img_md.get(attr, '')))
                return UISubset(ressset, host=univset)

            query_attr_btn.click(
                fn=query_handler(query_attr),
                inputs=query_base_inputs + [query_attr_selector, query_attr_pattern],
                outputs=dataset_change_listeners,
                concurrency_limit=1,
            )

            query_attr_selector.focus(
                fn=lambda: gr.update(choices=univset.header),
                outputs=[query_attr_selector],
                concurrency_limit=1,
            )

    return demo
