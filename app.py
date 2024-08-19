from typing import Optional, Literal
from waifuset.utils import config_utils
from waifui.ui import UIManager


def get_config():
    dataset_source = r"/path/to/your/dataset/source"
    share: bool = False
    port: Optional[int] = None
    language: Literal['en', 'cn'] = 'cn'
    page_size: int = 40
    cpu_max_workers: int = 1
    verbose: bool = True

    return config_utils.config(
        dataset_source=dataset_source,
        share=share,
        port=port,
        language=language,
        page_size=page_size,
        cpu_max_workers=cpu_max_workers,
        verbose=verbose,
    )


def launch(config):
    manager = UIManager.from_config(config)
    manager.launch()


if __name__ == "__main__":
    launch(get_config())
