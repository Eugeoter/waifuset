import gradio as gr
from modules.ui.ui import create_ui
from modules.ui.arg_parser import parse_args


def api(args):
    ui: gr.Blocks = create_ui(args)
    ui.launch(share=args.share)


if __name__ == '__main__':
    args = parse_args()
    api(args)
