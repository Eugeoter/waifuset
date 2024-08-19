import os
from .. import logging


class OnnxModelLoader:
    def __init__(self, model_path=None, model_url=None, cache_dir=None, *args, device='cuda', verbose=False, **kwargs):
        import torch
        import onnxruntime as rt

        if not hasattr(self, 'verbose'):
            self.verbose = verbose
        if not hasattr(self, 'logger'):
            self.logger = logging.get_logger(self.__class__.__name__)
        self.model_path = os.path.abspath(model_path)
        if not os.path.isfile(self.model_path):
            from ..utils.file_utils import download_from_url
            if model_url:
                self.model_path = download_from_url(model_url, cache_dir=cache_dir)
            else:
                raise FileNotFoundError(f"model file `{self.model_path}` not found.")

        # Load model
        device = torch.device(device)
        os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
        if device.type == 'cuda':
            self.providers = [
                'CUDAExecutionProvider',
                'CPUExecutionProvider',
            ]
            self.device = device if 'CUDAExecutionProvider' in self.providers else torch.device('cpu')
        elif device.type == 'cpu':
            self.providers = ['CPUExecutionProvider']
            self.device = device

        if device != self.device:
            self.logger.print(f"device `{device}` is not available, use `{self.device}` instead.")

        if self.verbose:
            self.logger.print(f"loading pretrained model from `{logging.stylize(self.model_path, logging.ANSI.YELLOW, logging.ANSI.UNDERLINE)}`")
            self.logger.print(f"  providers: {logging.stylize(self.providers, logging.ANSI.GREEN)}")
            if self.device == 'cuda':
                self.logger.print(f"  run on cuda: {logging.stylize(torch.version.cuda, logging.ANSI.GREEN)}")
            elif self.device == 'cpu':
                self.logger.print(f"  run on CPU.")

        with logging.timer("load model", logger=self.logger):
            self.model = rt.InferenceSession(
                self.model_path,
                providers=self.providers
            )
