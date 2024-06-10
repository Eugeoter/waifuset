from waifuset.ui.ui import UIManager
from argparse import ArgumentParser


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--source', nargs='+', default=[])
    parser.add_argument('--metadata', type=str, default=None)
    parser.add_argument('--loader', choices=['full', 'metadata'], default='full')
    parser.add_argument('--share', type=bool, default=False)
    parser.add_argument('--port', type=int, default=None)
    parser.add_argument('--language', choices=['en', 'cn'], default='en')
    parser.add_argument('--page_size', type=int, default=40)
    parser.add_argument('--cpu_max_workers', type=int, default=1)
    parser.add_argument('--verbose', type=bool, default=False)
    return parser.parse_args()


def launch(config):
    manager = UIManager(config)
    manager.launch()


if __name__ == "__main__":
    config = parse_arguments()
    launch(config)
