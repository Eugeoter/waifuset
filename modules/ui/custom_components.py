import gradio as gr
import emoji
from typing import Literal


def EmojiButton(value, variant: Literal['primary', 'secondary', 'stop'] = "secondary", *args, **kwargs):
    return gr.Button(value=value, variant=variant, scale=0, min_width=40, *args, **kwargs)


def CaptionPainterPanel(choices):
    add_btn = EmojiButton(emoji.emojize(':plus:'), variant='primary')
    tag_selector = gr.Dropdown(
        label='Tag',
        choices=choices,
        value=None,
        multiselect=True,
        allow_custom_value=True,
    )
    remove_btn = EmojiButton(emoji.emojize(':minus:'), variant='stop')
    return add_btn, tag_selector, remove_btn
