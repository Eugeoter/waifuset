import os
import time
from ..utils import log_utils


class OnnxModelLoader:
    def __init__(self, model_path=None, model_url=None, cache_dir=None, *args, device='cuda', verbose=False, **kwargs):
        import torch
        import onnxruntime as rt

        if not hasattr(self, 'verbose'):
            self.verbose = verbose
        if not hasattr(self, 'logger'):
            self.logger = log_utils.get_logger(self.__class__.__name__)
        self.model_path = os.path.abspath(model_path)
        if not os.path.isfile(self.model_path):
            from ..utils.file_utils import download_from_url
            if model_url:
                self.model_path = download_from_url(model_url, cache_dir=cache_dir)
            else:
                raise FileNotFoundError(f"model file `{self.model_path}` not found.")

        # Load model
        os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
        if device == 'cuda':
            self.providers = [
                'CUDAExecutionProvider',
                'CPUExecutionProvider',
            ]
            self.device = 'cuda' if device == 'cuda' and 'CUDAExecutionProvider' in self.providers else 'cpu'
        elif device == 'cpu':
            self.providers = ['CPUExecutionProvider']
            self.device = 'cpu'

        if device != self.device:
            self.logger.print(f"device `{device}` is not available, use `{self.device}` instead.")

        if self.verbose:
            self.logger.print(f"loading pretrained model from `{log_utils.stylize(self.model_path, log_utils.ANSI.YELLOW, log_utils.ANSI.UNDERLINE)}`")
            self.logger.print(f"  providers: {log_utils.stylize(self.providers, log_utils.ANSI.GREEN)}")
            if self.device == 'cuda':
                self.logger.print(f"  run on cuda: {log_utils.stylize(torch.version.cuda, log_utils.ANSI.GREEN)}")
            elif self.device == 'cpu':
                self.logger.print(f"  run on CPU.")

        with log_utils.timer("load model", self.logger):
            self.model = rt.InferenceSession(
                self.model_path,
                providers=self.providers
            )

    def __del__(self):
        if self.verbose:
            self.logger.print(f"model unloaded.")
        del self.model
