import os
import torch
import time
import onnxruntime as rt
from ...utils.file_utils import download_from_url
from ...utils import log_utils as logu


class OnnxModelLoader:
    def __init__(self, model_path=None, model_url=None, cache_dir=None, *args, device='cuda', verbose=False, **kwargs):
        self.verbose = verbose
        self.model_path = os.path.abspath(model_path)
        if not os.path.isfile(self.model_path):
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
            logu.warn(f"device `{device}` is not available, use `{self.device}` instead.")

        if self.verbose:
            tic = time.time()
            verbose_info = []
            verbose_info.append(f"loading pretrained model from `{logu.stylize(self.model_path, logu.ANSI.YELLOW, logu.ANSI.UNDERLINE)}`")
            verbose_info.append(f"  providers: {logu.stylize(self.providers, logu.ANSI.GREEN)}")
            if self.device == 'cuda':
                verbose_info.append(f"  run on cuda: {logu.stylize(torch.version.cuda, logu.ANSI.GREEN)}")
            elif self.device == 'cpu':
                verbose_info.append(f"  run on CPU.")
            verbose_info = '\n'.join(verbose_info)
            logu.info(verbose_info)

        self.model = rt.InferenceSession(
            self.model_path,
            providers=self.providers
        )

        if self.verbose:
            toc = time.time()
            logu.info(f"model loaded: time_cost={toc-tic:.2f}")
