from typing import Optional, Literal
from waifuset.utils import config_utils
from waifui.ui import UIManager


def get_config():
    dataset_source = r"/path/to/your/dataset/source"
    gradio_share: bool = False
    gradio_sever_port: Optional[int] = None
    gradio_sever_name: Optional[str] = None
    gradio_max_threads: Optional[int] = 40
    ui_language: Literal['en', 'cn'] = 'cn'
    ui_page_size: int = 40
    cpu_max_workers: int = 1
    verbose: bool = True

    return config_utils.config(
        dataset_source=dataset_source,
        gradio_share=gradio_share,
        gradio_sever_port=gradio_sever_port,
        gradio_sever_name=gradio_sever_name,
        gradio_max_threads=gradio_max_threads,
        ui_language=ui_language,
        ui_page_size=ui_page_size,
        cpu_max_workers=cpu_max_workers,
        verbose=verbose,
    )


def launch(config):
    manager = UIManager.from_config(config)
    manager.launch()


if __name__ == "__main__":
    launch(get_config())
