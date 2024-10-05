import os
import numpy as np
import torch
import pandas as pd
import onnxruntime as rt
from pathlib import Path
from PIL import Image
from typing import Dict, List, Union
from huggingface_hub import hf_hub_download

from ...utils import image_utils
from ... import logging

logger = logging.get_logger("waifu_tagger")


class WaifuTagger(object):
    def __init__(
        self,
        model: rt.InferenceSession,
        tag_names: List[str],
        general_indexes: List[int],
        character_indexes: List[int],
        input_name: str,
        label_name: str,
        model_target_size: int,
        device: torch.device,
        verbose: bool = True,
    ):
        self.model = model
        self.tag_names = tag_names
        self.general_indexes = general_indexes
        self.character_indexes = character_indexes
        self.input_name = input_name
        self.label_name = label_name
        self.model_target_size = model_target_size
        self.device = device
        self.verbose = verbose

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        cache_dir: str = None,
        use_safetensors: bool = False,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        verbose=True,
    ):
        # Download model and label if not exist
        if not os.path.isdir(pretrained_model_name_or_path):
            model_filename = "model.onnx" if not use_safetensors else "model.safetensors"
            label_filename = "selected_tags.csv"
            model_path = hf_hub_download(
                pretrained_model_name_or_path,
                filename=model_filename,
                cache_dir=cache_dir,
            )
            label_path = hf_hub_download(
                pretrained_model_name_or_path,
                filename=label_filename,
                cache_dir=cache_dir,
            )
        else:
            model_path = os.path.join(pretrained_model_name_or_path, "model.onnx" if not use_safetensors else "model.safetensors")
            label_path = os.path.join(pretrained_model_name_or_path, "selected_tags.csv")

        # Load model
        device = torch.device(device)
        os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
        if device.type == 'cuda':
            providers = [
                'CUDAExecutionProvider',
                'CPUExecutionProvider',
            ]
            device = device if 'CUDAExecutionProvider' in providers else torch.device('cpu')
        elif device.type == 'cpu':
            providers = ['CPUExecutionProvider']
            device = device

        if device != device:
            logger.print(f"device `{device}` is not available, use `{device}` instead.")

        if verbose:
            logger.print(f"loading pretrained model from `{logging.stylize(pretrained_model_name_or_path, logging.ANSI.YELLOW, logging.ANSI.UNDERLINE)}`")
            logger.print(f"  providers: {logging.stylize(providers, logging.ANSI.GREEN)}")
            if str(device) == 'cuda':
                logger.print(f"  run on CUDA: {logging.stylize(torch.version.cuda, logging.ANSI.GREEN)}")
            elif str(device) == 'cpu':
                logger.print(f"  run on CPU.")

        with logging.timer("load WaifuTagger", logger=logger):
            model = rt.InferenceSession(
                model_path,
                providers=providers
            )

        # Load labels
        df = pd.read_csv(label_path)  # Read csv file
        tag_names = df["name"].tolist()
        general_indexes = list(np.where(df["category"] == 0)[0])
        character_indexes = list(np.where(df["category"] == 4)[0])

        # Load other parameters
        input_name = model.get_inputs()[0].name
        label_name = model.get_outputs()[0].name
        model_target_size = model.get_inputs()[0].shape[1]

        return cls(
            model=model,
            tag_names=tag_names,
            general_indexes=general_indexes,
            character_indexes=character_indexes,
            input_name=input_name,
            label_name=label_name,
            model_target_size=model_target_size,
            device=device,
            verbose=verbose,
        )

    @property
    def providers(self):
        return self.model.get_providers()

    def __call__(
        self,
        images: Union[List[Image.Image], Image.Image],
        general_threshold: float = 0.35,
        character_threshold: float = 0.35,
    ) -> List[List[str]]:
        if not isinstance(images, list):
            images = [images]
        batch_inputs = np.concatenate([self.prepare_image(img) for img in images])

        # Run model
        with torch.no_grad():
            batch_probs = self.model.run(
                [self.label_name],
                {self.input_name: batch_inputs}
            )[0]

        batch_captions = []
        batch_labels = [list(zip(self.tag_names, batch_probs[i].astype(float))) for i in range(len(batch_probs))]

        for labels in batch_labels:
            # Get general and character tags
            general_tags = get_tags(
                labels,
                self.general_indexes,
                general_threshold
            )
            character_tags = get_tags(
                labels,
                self.character_indexes,
                character_threshold
            )

            # Get top general tag
            general_tags = postprocess_tags(general_tags)
            character_tags = postprocess_tags(character_tags)
            character_tags = [f"character:{tag}" for tag in character_tags]

            tags = character_tags + general_tags
            batch_captions.append(tags)

        return batch_captions

    def prepare_image(self, image: Image.Image):
        target_size = self.model_target_size

        image = image.convert("RGBA")
        canvas = Image.new("RGBA", image.size, (255, 255, 255))
        canvas.alpha_composite(image)
        image = canvas.convert("RGB")

        # Pad image to square
        image_shape = image.size
        max_dim = max(image_shape)
        pad_left = (max_dim - image_shape[0]) // 2
        pad_top = (max_dim - image_shape[1]) // 2

        padded_image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        padded_image.paste(image, (pad_left, pad_top))

        # Resize
        if max_dim != target_size:
            padded_image = padded_image.resize(
                (target_size, target_size),
                Image.BICUBIC,
            )

        # Convert to numpy array
        image_array = np.asarray(padded_image, dtype=np.float32)

        # Convert PIL-native RGB to BGR
        image_array = image_array[:, :, ::-1]

        return np.expand_dims(image_array, axis=0)


def preprocess_image(image, size: int):
    import cv2
    # 1. load image
    # convert to numpy array
    if isinstance(image, (str, Path)):
        image = image_utils.load_image(image)
    elif isinstance(image, Image.Image):
        image = np.array(image, dtype=np.uint8)

    image = image_utils.cvt2rgb(image)  # convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert to BGR

    # 2. Pad image with white pixels to (target_size, target_size)
    old_size = image.shape[:2]
    desired_size = max(old_size)
    desired_size = max(desired_size, size)

    delta_w = desired_size - old_size[1]
    delta_h = desired_size - old_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [255, 255, 255]
    image = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )

    # 3. Resize image accordingly
    if image.shape[0] > size:
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    elif image.shape[0] < size:
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC)

    # 4. Convert image type to fit model required datatype
    image = image.astype(np.float32)
    image = np.expand_dims(image, 0)

    return image


def get_tags(labels, indexes, threshold) -> Dict[str, float]:
    r"""
    Filter tags whose probability is beyond threshold and convert them to dictionary.
    """
    names = [labels[i] for i in indexes]
    res = [x for x in names if x[1] > threshold]
    res = dict(res)
    return res


def postprocess_tags(tags: Dict[str, float]) -> List[str]:
    r"""
    Post-process tags by:
        (1) Sort tags by probability
        (2) Transform dict tags to string of labels
        (3) Replace underlines with spaces, parentheses with escape parentheses
    :param tags: Dict of tags mapping from tag name to probability.
    :return: Post-processed string of tags
    """
    tags = dict(sorted(tags.items(), key=lambda item: item[1], reverse=True))
    tags = list(tags.keys())
    return tags
