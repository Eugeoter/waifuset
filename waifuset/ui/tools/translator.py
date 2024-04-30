import googletrans
import re
import gradio as gr
from ..emoji import Emoji


def create_ui(univargs):
    src_langs = ['auto'] + list(googletrans.LANGUAGES.keys())
    dest_langs = ['auto'] + list(googletrans.LANGUAGES.keys())

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=1):
                src_language = gr.Dropdown(
                    choices=src_langs,
                    show_label=False,
                    value='auto',
                    scale=0,
                )
            with gr.Column(scale=1):
                swap_button = gr.Button(Emoji.clockwise_rightwards_and_leftwards_open_circle_arrows)
            with gr.Column(scale=1):
                dest_language = gr.Dropdown(
                    choices=dest_langs,
                    show_label=False,
                    value='auto',
                    scale=0,
                )

        with gr.Row():
            with gr.Column(scale=1):
                input_text = gr.Textbox(lines=5, show_label=False)
            with gr.Column(scale=1):
                output_text = gr.Textbox(lines=5, show_label=False)

        def google_translate(text, src='auto', dest='auto'):
            if not text:
                return ""
            # if text contains chinese, japanese, korean, translate them into english
            if re.search(r'[\u4e00-\u9fa5]+|[\u3040-\u309f\u30a0-\u30ff]+|[\uac00-\ud7a3]+', text):
                dest = 'en'
            else:  # otherwise, translate into universal language
                dest = 'zh-CN' if univargs.language == 'cn' else 'en'
            text = text.replace('_', ' ').replace('\(', '(').replace('\)', ')')
            return googletrans.Translator().translate(text, src=src, dest=dest).text

        input_text.change(
            fn=google_translate,
            inputs=[input_text, src_language, dest_language],
            outputs=[output_text]
        )

        swap_button.click(
            fn=lambda a, b: (b, a),
            inputs=[src_language, dest_language],
            outputs=[src_language, dest_language]
        )

    return demo
