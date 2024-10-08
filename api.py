import gradio as gr
from waifuset.ui.ui import create_ui
from waifuset.ui.arg_parser import parse_args


def api(args):
    ui: gr.Blocks = create_ui(args)
    ui.launch(
        share=args.share,
        server_port=args.port,
    )


if __name__ == '__main__':
    args = parse_args()
    api(args)
