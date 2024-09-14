import os
import re
import gradio as gr
import inspect
import time
import pandas
from pathlib import Path
from PIL import Image
from functools import wraps
from typing import Callable, Any, Tuple, Dict, List, Iterable, Union, Optional, Literal
from pathlib import Path
from copy import deepcopy
from waifuset.const import IMAGE_EXTS
from waifuset import logging
from waifuset import Dataset, Caption, SQLite3Dataset, AutoDataset, FastDataset
from waifuset.utils import image_utils, class_utils
from waifuset.classes.dataset.dataset_mixin import ToDiskMixin
from waifuset.classes.data import data_utils
from waifuset.components.waifu_tagger.const import WD_REPOS
from waifuset.components.waifu_scorer.const import WS_REPOS
from .emoji import Emoji
from .ui_utils import *
from .ui_dataset import UIDataset, UISubset


class UIManager(class_utils.FromConfigMixin):
    dataset_source: Union[str, List[str], Dict[str, Any], List[Dict[str, Any]]] = None
    share: bool = False
    gradio_sever_port: Optional[int] = None
    gradio_sever_name: Optional[str] = None
    gradio_max_threads: Optional[int] = 40
    ui_language: Literal['en', 'cn'] = 'cn'
    ui_page_size: int = 40
    cpu_max_workers: int = 1
    enable_category: bool = True
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
            with self.logger.timer('load dataset', level='debug'):
                dataset = FastDataset(self.dataset_source, verbose=self.verbose, **self.get_default_kwargs())
            with self.logger.timer('patch dataset', level='debug'):
                if any(col not in dataset.header for col in ('image_key', 'category')):
                    self.logger.print('patching image path base info')
                    dataset.add_columns(['image_path', 'image_key', 'category', 'source', 'caption', 'description'])
                    dataset.apply_map(patch_image_path_base_info)
            dataset = UIDataset.from_dataset(
                dataset,
                host=dataset,
                page_size=self.ui_page_size
            )
        # self.logger.print(dataset, no_prefix=True)
        self.logger.print(f"dataset size: {len(dataset)}x{len(dataset.header)}")
        return dataset

    def setup(self):
        self.logger.print("setting up UI")
        with self.logger.timer('setup', level='debug'):
            self.dataset = self.load_dataset()
            with self.logger.timer('launch ui'):
                self.ui = create_ui(
                    univset=self.dataset,
                    buffer=UIBuffer(),
                    cpu_max_workers=self.cpu_max_workers,
                    language=self.ui_language,
                    render='full',
                    enable_category=self.enable_category,
                )

    def launch(self):
        self.logger.print("launching UI")
        self.ui.queue().launch(
            share=self.share,
            server_port=self.gradio_sever_port,
            server_name=self.gradio_sever_name,
            max_threads=self.gradio_max_threads,
        )


def create_ui(
    univset: UIDataset,
    buffer: UIBuffer,
    cpu_max_workers=1,
    language='en',
    render='full',
    enable_category=True,
):
    # ========================================= UI ========================================= #
    assert isinstance(univset, UIDataset), f"expected `univset` to be an instance of `UIDataset`, but got {type(univset)}"
    assert isinstance(buffer, UIBuffer), f"expected `buffer` to be an instance of `UIBuffer`, but got {type(buffer)}"
    assert language in ('en', 'cn'), f"expected `language` to be one of ('en', 'cn'), but got {language}"
    assert render in ('full', 'demo'), f"expected `render` to be one of ('full', 'demo'), but got {render}"

    logger = logging.get_logger('UI')
    logger.debug(f"initializing UI state")
    state = UIState(
        page_index=0,
        selected=UIGallerySelectData(),
    )
    waifu_tagger = None
    waifu_scorer = None
    # logger.debug(f"initializing custom tags")
    tagging.init_custom_tags()

    def convert_dataset_to_statistic_dataframe(dataset: UISubset):
        num_images = len(dataset)
        data = {
            'Number of images': [num_images],
        }
        if enable_category:
            categories = sorted(dataset.get_categories()) if len(dataset) > 0 else []
            num_cats = len(categories)
            if num_cats > 5:
                categories = categories[:5] + ['...']
            data.update({
                'Number of categories': [num_cats],
                'Categories': [categories],
            })
        # translate keys
        data = {translate(k, language): v for k, v in data.items()}
        df = pandas.DataFrame(data)
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
                        # Disable when `enable_category` is False
                        with gr.Tab(translate('Category', language), visible=enable_category):
                            with gr.Row():
                                # logger.debug(f"initializing categories")
                                category_selector = gr.Dropdown(
                                    label=translate('Category', language),
                                    choices=univset.get_categories() if enable_category else [],
                                    value=None,
                                    container=False,
                                    multiselect=True,
                                    allow_custom_value=False,
                                    min_width=256,
                                )
                                reload_category_btn = EmojiButton(Emoji.anticlockwise)

                        with gr.Tab(translate('Query', language)):
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
                                        placeholder=translate('Value', language),
                                    )
                                    query_attr_btn = EmojiButton(Emoji.right_pointing_magnifying_glass, variant='primary')
                            with gr.Row():
                                query_opts = gr.CheckboxGroup(
                                    choices=[translate('Subset', language), translate('Complement', language), translate('Regex', language)],
                                    value=None,
                                    container=False,
                                    scale=1,
                                )

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
                                    container=True,
                                )

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
                                # logger.debug(f"initializing showcase")
                                showcase = gr.Gallery(
                                    label=translate('Showcase', language),
                                    value=convert_dataset_to_gallery(univset.curset.get_page(0)),
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
                                # logger.debug(f"initializing dataset metadata")
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
                            with gr.Tab(label=translate('Tagging', language)):
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

                            with gr.Tab(label=translate('Toolbox', language)):

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
                                            min_width=96,
                                            precision=0,
                                            scale=0,
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
                                        ws_add_score_btn = EmojiButton(Emoji.black_right_pointing_triangle, variant='primary', min_width=40)
                                        ws_score2quality_btn = EmojiButton(Emoji.label, min_width=40)
                                        ws_del_score_btn = EmojiButton(Emoji.no_entry, variant='stop', min_width=40)

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

                            with gr.Tab(label=translate('Labeling', language)):
                                with gr.Tab(label=translate('Quality Labeling', language)):
                                    with gr.Row(variant='compact'):
                                        quality_label_btns = [
                                            gr.Button(
                                                value=translate(quality, language),
                                                scale=1,
                                                min_width=96,
                                                variant='primary' if i == 0 else 'stop' if i == len(get_quality2score()) - 1 else 'secondary',
                                            ) for i, quality in enumerate(get_quality2score())
                                        ]

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

            # logger.debug(f"initializing functions")

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
                Correct the chunk index of `dataset` to `page_index`
                """
                assert isinstance(dataset, UISubset), f"expected `dataset` to be an instance of `UISubset`, but got {type(dataset)}"
                if page_index is None or page_index == 0:
                    page_index = 1
                else:
                    page_index = min(max(page_index, 1), dataset.num_pages)
                return page_index

            def change_current_subset(newset: UISubset, sorting_methods=None, reverse=False):
                r"""
                Change the current dataset to a new dataset.
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

                univset.set_curset(newset)

                # post-sorting
                # if sorting_methods is not None and len(sorting_methods) > 0:
                #     subset.sort(key=lambda item: tuple(func(item[1], **kwargs) for func, kwargs in sorting_keys), reverse=reverse)

            def update_showcase_from_dataset(showset: UISubset = None, new_page_index=1):
                r"""
                Convert `showset` to gallery value and image key that should be selected
                """
                assert isinstance(showset, UISubset), f"expected `showset` to be an instance of `UISubset`, but got {type(showset)}"
                new_page_index = correct_page_index(showset, new_page_index)
                page = showset.get_page(new_page_index - 1)
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
                page = dataset.get_page(new_page_index - 1) if isinstance(dataset, UISubset) else dataset
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

            def load_subset_from_categories(categories: List[str], sorting_methods=None, reverse=False):
                r"""
                Change current dataset to a new dataset whose images are all belong to one of the specified categories.
                """
                logger.info(f"loading subset from categories: {', '.join([logging.yellow(category) for category in categories])}")
                tic = time.time()

                # If no categories are selected, show the full dataset
                if not categories:
                    catset = univset.fullset
                # If the backbone is SQLite3Dataset, use SQL query to select the subset to improve efficiency
                elif isinstance(rootset := univset.root, SQLite3Dataset):
                    if 'category' in rootset.header:
                        catset = rootset.select_in('category', categories) if len(categories) > 1 else rootset.select_is('category', categories[0])
                    else:
                        catset = rootset.select_like('image_path', f"%{categories[0]}%")
                        catset = catset.subset(condition=lambda img_md: os.path.basename(os.path.dirname(img_md['image_path'])) in set(categories))
                    catset = UISubset.from_dataset(catset, host=univset)
                # Otherwise, directly use Python to select the subset
                else:
                    catset = univset.subset(condition=lambda img_md: (img_md.get('category', None) or os.path.basename(os.path.dirname(img_md['image_path']))) in set(categories))
                subset = load_subset_from_dataset(catset, new_page_index=1, sorting_methods=sorting_methods, reverse=reverse)
                logger.info(f"loaded {logging.yellow(len(catset))} data from {logging.yellow(len(categories))} categories in {time.time() - tic:.3f}s")
                return subset

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

            def get_other_md_keys(img_md=None):
                return [key for key in (img_md.keys() if img_md is not None else univset.header) if key not in (*BASE_MD_KEYS, *CAPTION_MD_KEYS)]

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
                    # logger.warning(f"image key mismatch: {img_key} != {state.selected.key}, fix to {state.selected.key}")
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
                        other_metadata_df: get_metadata_df(img_key, keys=get_other_md_keys(img_md)),
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
                            other_metadata_df: get_metadata_df(img_key, keys=get_other_md_keys(img_md)),
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
                        get_metadata_df(img_key, keys=get_other_md_keys(univset[img_key]))
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

            # ========================================= Base Tagging Buttons ========================================= #

            def data_edition_handler(func: Callable[[DataDict, Tuple[Any, ...], Dict[str, Any]], ResultDict]) -> Tuple[str, str]:
                r"""
                Warp a data edition function to handle the data edition process.
                @param func: A function that can be the following format:
                    - `func(img_md: DataDict, *args: Any, **kwargs: Any) -> ResultDict`
                    - `func(batch: List[DataDict], *args: Any, **kwargs: Any) -> List[ResultDict]`
                @selected_img_key: The image key that is selected by the user
                @opts: A list of strings that contains the options for the function
                @progress: The progress bar
                """
                func_name = func.__name__

                def wrapper(selected_img_key: str, opts: List[str], *args, progress: gr.Progress = gr.Progress(track_tqdm=True), **kwargs):
                    nonlocal func_name, cpu_max_workers
                    formatted_func_name = func_name.replace('_', ' ')

                    # parse options
                    if language != 'en':
                        opts = translate(opts, 'en')
                    opts = [opt.lower() for opt in opts]
                    do_batch = ('batch' in opts) and (func is not write_caption)
                    do_progress = 'progress' in opts
                    extra_kwargs = dict(
                        do_append='append' in opts,
                        do_regex='regex' in opts,
                    )
                    # filter out extra kwargs
                    func_params = list(inspect.signature(func).parameters.keys())
                    extra_kwargs = {k: v for k, v in extra_kwargs.items() if k in func_params}

                    if selected_img_key is None or selected_img_key == '':
                        if not do_batch:
                            return {log_box: f"{formatted_func_name}: empty image key"}
                    else:
                        selected_img_key = Path(selected_img_key).stem if Path(selected_img_key).suffix in IMAGE_EXTS else selected_img_key

                    def edit_batch(batch: Union[DataDict, List[DataDict]], *args, **kwargs) -> Dict[str, ResultDict]:
                        batch = deepcopy(batch)  # avoid modifying the original data
                        if not isinstance(batch, list):  # single img_md
                            img_md = batch
                            if not os.path.isfile(img_md['image_path']):
                                return []
                            res_md = func(img_md, *args, **extra_kwargs, **kwargs)  # single img_md
                            return {img_md['image_key']: res_md}
                        else:  # list of img_md
                            batch = [img_md for img_md in batch if os.path.isfile(img_md['image_path'])]
                            res_batch = func([img_md for img_md in batch], *args, **extra_kwargs, **kwargs)  # list of img_md
                            return {img_md['image_key']: res_md for img_md, res_md in zip(batch, res_batch)}

                    is_func_support_batch = func in (ws_scoring, wd_tagging)
                    res_dict = {}
                    if do_batch:
                        logger.print(f"batch processing: {formatted_func_name}")
                        editset = univset.curset
                        if is_func_support_batch:
                            batch_size, args = args[0], args[1:]  # first arg is batch size
                            values = list(editset.values())
                            batches = [values[i:i + batch_size] for i in range(0, len(editset), batch_size)]
                        else:
                            batch_size = 1  # fix batch size to 1
                            batches = list(editset.values())
                        desc = f'[{formatted_func_name}] batch processing'
                        pbar = logger.tqdm(total=len(batches), desc=desc)
                        edit_batch = logging.track_tqdm(pbar)(edit_batch)
                        if do_progress:
                            edit_batch = track_progress(progress, desc=desc, total=len(batches))(edit_batch)

                        if cpu_max_workers == 1:
                            for batch in batches:
                                res_dict.update(edit_batch(batch, *args, **kwargs))
                        else:
                            from concurrent.futures import ThreadPoolExecutor, wait
                            with ThreadPoolExecutor(max_workers=cpu_max_workers) as executor:
                                futures = [executor.submit(edit_batch, batch, *args, **kwargs) for batch in batches]
                                try:
                                    wait(futures)
                                    for future in futures:
                                        res_dict.update(future.result())
                                except (gr.CancelledError, KeyboardInterrupt):
                                    for future in futures:
                                        future.cancel()
                                    raise
                        pbar.close()

                    else:
                        if is_func_support_batch:  # list in
                            args = args[1:]
                            batch = [univset[selected_img_key]]
                        else:  # single in
                            batch = univset[selected_img_key]
                        res_dict.update(edit_batch(batch, *args, **kwargs))

                    # write to dataset
                    is_updated_at_least_one = False
                    is_undo_redo = func in (undo, redo)
                    is_write_caption = func is write_caption

                    for img_key, res_md in res_dict.items():
                        if res_md:
                            if not is_undo_redo:
                                if is_write_caption and 'caption' in res_md and res_md['caption'] == univset[img_key]['caption']:
                                    continue  # skip if `write_caption` didn't change the caption
                                if img_key not in buffer:
                                    buffer.do(img_key, univset[img_key])  # push the original data into the bottom of the buffer stack
                            univset.set(img_key, res_md)
                            if not is_undo_redo:
                                buffer.do(img_key, univset[img_key])
                            is_updated_at_least_one = True

                    if is_updated_at_least_one:
                        orig_header = univset.header
                        # univset.update_header()
                        if univset.header != orig_header:
                            logger.info(f"add new columns: {', '.join(set(univset.header) - set(orig_header))}")
                        ret = track_img_key(selected_img_key)
                        if do_batch:  # batch processing
                            ret.update({log_box: f"{formatted_func_name}: update {len(res_dict)} over {len(editset)}"})
                        else:  # single processing
                            ret.update({log_box: f"{formatted_func_name}: update `{selected_img_key}`"})
                        return ret
                    else:
                        return {log_box: f"{formatted_func_name}: no change"}
                return wrapper

            def cancel_edition_process():
                return {log_box: "processing is cancelled."}

            cancel_event = cancel_btn.click(
                fn=cancel_edition_process,
                outputs=[log_box],
                concurrency_limit=1,
            )

            def write_caption(img_md, caption: str):
                return {'caption': caption}

            caption.blur(
                fn=data_edition_handler(write_caption),
                inputs=[image_path, general_edit_opts, caption],
                outputs=cur_img_key_change_listeners,
                cancels=cancel_event,
                concurrency_limit=1,
            )

            def undo(img_md):
                return buffer.undo(img_md['image_key'])

            undo_btn.click(
                fn=data_edition_handler(undo),
                inputs=[cur_img_key, general_edit_opts],
                outputs=cur_img_key_change_listeners,
                concurrency_limit=1,
            )

            def redo(img_md):
                return buffer.redo(img_md['image_key'])

            redo_btn.click(
                fn=data_edition_handler(redo),
                inputs=[cur_img_key, general_edit_opts],
                outputs=cur_img_key_change_listeners,
                concurrency_limit=1,
            )

            def save_to_disk(fp, progress: gr.Progress = gr.Progress(track_tqdm=True)):
                rootset = univset.root

                # if rootset is ToDiskMixin, save to disk
                if isinstance(rootset, ToDiskMixin):
                    if fp:
                        os.makedirs(os.path.dirname(fp), exist_ok=True)
                        rootset.dump(fp)
                        logger.info(f"dump dataset to: {fp}")
                    else:
                        fp = rootset.fp
                        rootset.commit()
                        logger.info(f"commit dataset to: {fp}")
                    return f"saved: {fp}"
                # if rootset is not ToDiskMixin but save path is provided, try to dump the dataset to the save path
                elif fp:
                    ext = os.path.splitext(fp)[1]
                    if ext not in ('.json', '.csv', '.sqlite3', '.db'):
                        raise gr.Error(f"unsupported extension: {ext}")
                        return {log_box: f"unsupported extension: {ext}"}
                    os.makedirs(os.path.dirname(fp), exist_ok=True)
                    FastDataset.dump(rootset, fp)
                    logger.info(f"dump dataset to: {fp}")
                    return f"saved: {fp}"
                # if rootset is DirectoryDataset, save the dataset as txt caption files
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
                    logger.info(f"save dataset to txt caption files")
                    return f"saved"
                else:
                    raise gr.Error(f"failed to save dataset: Dataset type not supported or invalid save path")

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
                return {'caption': caption.text}

            def remove_tags(img_md, tags, do_regex):
                caption = img_md.get('caption', None)
                if caption is None:
                    return None
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
                return {'caption': caption.text}

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
                    return
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
                return {'caption': caption.text}

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
                return attrs_dict

            read_attrs_btn.click(
                fn=data_edition_handler(read_attrs),
                inputs=[cur_img_key, general_edit_opts, read_attrs_types],
                outputs=cur_img_key_change_listeners,
                concurrency_limit=1,
            )

            def parse_caption_attrs(img_md):
                caption = img_md.get('caption', None)
                if caption is None:
                    return None
                caption = Caption(caption).parsed()
                attr_dict = {}
                attr_dict['caption'] = caption.text
                attrs = caption.attrs
                attrs.pop('tags')
                for attr, value in attrs.items():
                    attr_dict[attr] = caption.sep.join(value)
                return attr_dict

            parse_caption_attrs_btn.click(
                fn=data_edition_handler(parse_caption_attrs),
                inputs=[cur_img_key, general_edit_opts],
                outputs=cur_img_key_change_listeners,
                concurrency_limit=1,
            )

            def sort_caption(img_md):
                caption = img_md.get('caption', None)
                if caption is None:
                    return None
                return {'caption': Caption(caption).sorted().text}

            sort_caption_btn.click(
                fn=data_edition_handler(sort_caption),
                inputs=[cur_img_key, general_edit_opts],
                outputs=cur_img_key_change_listeners,
                concurrency_limit=1,
            )

            def formalize_caption(img_md, formats):
                caption = img_md.get('caption', None)
                if caption is None:
                    return None
                if isinstance(formats, str):
                    formats = [formats]
                if language != 'en':
                    formats = [translate(fmt, 'en') for fmt in formats]
                caption = Caption(caption)
                for fmt in formats:
                    caption.format(FORMAT_PRESETS[fmt])
                return {'caption': caption.text}

            formalize_caption_btn.click(
                fn=data_edition_handler(formalize_caption),
                inputs=[cur_img_key, general_edit_opts, formalize_caption_dropdown],
                outputs=cur_img_key_change_listeners,
                concurrency_limit=1,
            )

            def deduplicate_caption(img_md):
                caption = img_md.get('caption', None)
                if caption is None:
                    return None
                return {'caption': Caption(caption).deduplicated().text}

            deduplicate_caption_btn.click(
                fn=data_edition_handler(deduplicate_caption),
                inputs=[cur_img_key, general_edit_opts],
                outputs=cur_img_key_change_listeners,
                concurrency_limit=1,
            )

            def deoverlap_caption(img_md):
                caption = img_md.get('caption', None)
                if caption is None:
                    return None
                return {'caption': Caption(caption).deoverlapped().text}

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
                if language != 'en':
                    feature_type = translate(feature_type, 'en')
                return {'caption': Caption(caption).decharacterized(feature_type=feature_type, freq_thres=freq_thres).text}

            decharacterize_caption_btn.click(
                fn=data_edition_handler(decharacterize_caption),
                inputs=[cur_img_key, general_edit_opts, dechar_feature_type, dechar_freq_thres],
                outputs=cur_img_key_change_listeners,
                concurrency_limit=1,
            )

            # ========================================= WD ========================================= #

            def wd_tagging(batch: List[DataDict], model_repo_or_path, general_threshold, character_threshold, overwrite_mode) -> List[ResultDict]:
                nonlocal waifu_tagger
                if language != 'en':  # translate overwrite_mode
                    overwrite_mode = translate(overwrite_mode, 'en')
                if overwrite_mode not in ('ignore', 'overwrite', 'append', 'prepend'):  # check overwrite_mode
                    raise ValueError(f"invalid os_mode: {overwrite_mode}")

                # make batch
                if not isinstance(batch, list):
                    batch = [batch]
                if overwrite_mode == 'ignore':
                    batch = [img_md for img_md in batch if img_md['caption'] is None]
                    if len(batch) == 0:
                        return []

                if waifu_tagger is None or (waifu_tagger and waifu_tagger.model_name != model_repo_or_path):
                    try:
                        from waifuset.components.waifu_tagger.predict import WaifuTagger, repo2path
                    except ModuleNotFoundError as e:
                        missing_package_name = e.name
                        raise gr.Error(f"Missing package {missing_package_name}. Please read README.md for installation instructions.")
                    model_path, label_path = repo2path(model_repo_or_path)
                    waifu_tagger = WaifuTagger(model_path=model_path, label_path=label_path, verbose=True)
                    waifu_tagger.model_name = model_repo_or_path

                batch_images = [Image.open(img_md['image_path']) for img_md in batch]
                batch_pred_tags = waifu_tagger(batch_images, general_threshold=general_threshold, character_threshold=character_threshold)
                batch_results = []
                for img_md, pred_tags in zip(batch, batch_pred_tags):
                    if overwrite_mode == 'overwrite' or overwrite_mode == 'ignore':
                        tags = pred_tags
                    elif overwrite_mode == 'append':
                        tags = img_md.get('caption', '').split(', ') + pred_tags
                    else:  # elif overwrite_mode == 'prepend':
                        tags = pred_tags + img_md.get('caption', []).split(', ')
                    batch_results.append({'caption': ', '.join(tags)})
                return batch

            wd_run_btn.click(
                fn=data_edition_handler(wd_tagging),
                inputs=[cur_img_key, general_edit_opts, wd_batch_size, wd_model, wd_general_threshold, wd_character_threshold, wd_overwrite_mode],
                outputs=cur_img_key_change_listeners,
                concurrency_limit=1,
            )

            # ========================================= WS ========================================= #
            def ws_scoring(batch: List[DataDict], model_repo_or_path: str, overwrite_mode: Literal['ignore', 'overwrite', 'append', 'prepend']) -> List[ResultDict]:
                if language != 'en':
                    overwrite_mode = translate(overwrite_mode, 'en')

                nonlocal waifu_scorer
                if waifu_scorer is None or (waifu_scorer and waifu_scorer.model_name != model_repo_or_path):
                    try:
                        import torch
                        from waifuset.components.waifu_scorer.predict import WaifuScorer, repo2path
                    except ModuleNotFoundError as e:
                        missing_package_name = e.name
                        raise gr.Error(f"Missing package {missing_package_name}. Please read README.md for installation instructions.")
                    model_path = repo2path(model_repo_or_path)
                    waifu_scorer = WaifuScorer(model_path=model_path, device='cuda' if torch.cuda.is_available() else 'cpu', verbose=True)
                    waifu_scorer.model_name = model_repo_or_path

                if not isinstance(batch, list):
                    batch = [batch]
                batch = [img_md for img_md in batch if os.path.isfile(img_md['image_path']) and not (overwrite_mode == 'ignore' and img_md.get('aesthetic_score', None) is not None)]
                if len(batch) == 0:
                    return []
                images = [Image.open(img_md['image_path']) for img_md in batch]
                aesthetic_scores = waifu_scorer(images)
                if isinstance(aesthetic_scores, float):  # single output
                    aesthetic_scores = [aesthetic_scores]
                return [{'aesthetic_score': score} for score in aesthetic_scores]

            ws_add_score_btn.click(
                fn=data_edition_handler(ws_scoring),
                inputs=[cur_img_key, general_edit_opts, ws_batch_size, ws_model, ws_overwrite_mode],
                outputs=cur_img_key_change_listeners,
                concurrency_limit=1,
                cancels=cancel_event,
            )

            def set_aesthetic_score(img_md, score):
                return {'aesthetic_score': score}

            ws_del_score_btn.click(
                fn=data_edition_handler(partial(set_aesthetic_score, score=None)),
                inputs=[cur_img_key, general_edit_opts],
                outputs=cur_img_key_change_listeners,
                concurrency_limit=1,
            )

            def set_quality(img_md, quality):
                img_md['quality'] = quality
                if (caption := img_md.get('caption', None)) is not None and 'quality' in caption:
                    caption = Caption(caption)
                    caption[r"(.+)[\s_]quality"] = rf"{quality}\2quality"
                return {'caption': caption.text, 'quality': quality}

            for quality, quality_label_btn in zip(get_quality2score().keys(), quality_label_btns):
                quality_label_btn.click(
                    fn=data_edition_handler(partial(set_quality, quality=quality)),
                    inputs=[cur_img_key, general_edit_opts],
                    outputs=cur_img_key_change_listeners,
                    concurrency_limit=1,
                )

            def set_score_to_quality(img_md):
                aesthetic_score = img_md.get('aesthetic_score', None)
                if aesthetic_score is None:
                    return img_md
                if not 0 <= aesthetic_score <= 10:
                    raise gr.Error(f"invalid score: {aesthetic_score}")
                quality = convert_score2quality(aesthetic_score)
                return {**set_quality(img_md, quality), 'aesthetic_score': aesthetic_score}

            ws_score2quality_btn.click(
                fn=data_edition_handler(set_score_to_quality),
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
                return {'perceptual_hash': str(p_hash)}

            hasher_run_btn.click(
                fn=data_edition_handler(get_perceptual_hash),
                inputs=[cur_img_key, general_edit_opts, hasher_overwrite_mode],
                outputs=cur_img_key_change_listeners,
                concurrency_limit=1,
            )

            def set_perceptual_hash(img_md, p_hash):
                return {'perceptual_hash': p_hash}

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
                bufferset = UISubset.from_dict(buffer.latests(), host=univset)
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
                        resset = UISubset.from_keys([img_key for img_key in queryset.keys() if img_key not in resset], host=univset)
                    logger.print(f"`{funcname}` found: {len(resset)}/{len(queryset)}")
                    result = load_subset_from_dataset(resset, sorting_methods=sorting_methods, reverse=reverse)
                    result.update({log_box: f"[{funcname}] found: {len(resset)}/{len(queryset)}"})
                    return result
                return wrapper

            def query_attr(queryset: UISubset, attr: str, pattern: str, do_regex: bool = False):
                r"""
                Query the dataset by a specific attribute.
                """
                if not attr:
                    return None
                if not pattern:
                    if isinstance(rootset := queryset.root, SQLite3Dataset):
                        result_keys = [row[0] for row in rootset.table.select_is(attr, None)]
                    else:
                        result_keys = rootset.subkeys(lambda img_md: img_md.get(attr, '') is None)
                    return UISubset.from_keys(result_keys, host=univset)
                if do_regex:
                    def match(attr, pattern):
                        return re.match(pattern, attr) is not None
                else:
                    def match(attr, pattern):
                        return attr == pattern
                if isinstance(rootset := queryset.root, SQLite3Dataset):
                    if do_regex:
                        result_keys = [row[0] for row in rootset.table.select_func(match, '$' + attr + '$', pattern)]
                    else:
                        result_keys = [row[0] for row in rootset.table.select_is(attr, pattern)]
                else:
                    result_keys = rootset.subkeys(lambda img_md: match(pattern, img_md.get(attr, '')))
                return UISubset.from_keys(result_keys, host=univset)

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
