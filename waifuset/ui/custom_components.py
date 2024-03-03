import gradio as gr
from typing import Literal
from .emoji import Emoji


def EmojiButton(value, variant: Literal['primary', 'secondary', 'stop'] = "secondary", scale=0, min_width=40, *args, **kwargs):
    return gr.Button(value=value, variant=variant, scale=scale, min_width=min_width, *args, **kwargs)


def CaptionPainterPanel(choices):
    add_btn = EmojiButton(Emoji.plus_sign, variant='primary')
    tag_selector = gr.Dropdown(
        label='Tag',
        choices=choices,
        value=None,
        multiselect=True,
        allow_custom_value=True,
    )
    remove_btn = EmojiButton(Emoji.minus_sign, variant='stop')
    return add_btn, tag_selector, remove_btn
