import os
import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from typing import Dict, List, Union
from ..loaders import OnnxModelLoader
from ....utils.file_utils import download_from_url
from ....utils import log_utils as logu, image_utils as imgu
from ....const import StrPath

WD_MODEL_URL = os.getenv(
    "WD_MODEL_URL",
    "https://huggingface.co/SmilingWolf/wd-v1-4-swinv2-tagger-v2/model.onnx"
)

WD_LABEL_URL = os.getenv(
    "WD_LABEL_URL",
    "https://huggingface.co/SmilingWolf/wd-v1-4-swinv2-tagger-v2/selected_tags.csv"
)

WD_CACHE_DIR = "./models/wd/"
WD_MODEL_PATH = "./models/wd/swinv2.onnx"
WD_LABEL_PATH = "./models/wd/selected_tags.csv"


class WaifuTagger(OnnxModelLoader):
    def __init__(self, model_path=WD_MODEL_PATH, label_path=WD_LABEL_PATH, cache_dir=WD_CACHE_DIR, device='cuda', verbose=False):
        if model_path is None:
            logu.info(f"model path is None, switch to default: `{WD_MODEL_PATH}`")
            model_path = WD_MODEL_PATH
        if label_path is None:
            logu.info(f"label path is None, switch to default: `{WD_LABEL_PATH}`")
            label_path = WD_LABEL_PATH

        super().__init__(model_path=model_path, model_url=WD_MODEL_URL, cache_dir=cache_dir, device=device, verbose=verbose)

        self.label_path = Path(label_path)
        if not os.path.isfile(self.label_path):
            self.label_path = download_from_url(WD_LABEL_URL, cache_dir=cache_dir)

        # Load labels
        df = pd.read_csv(self.label_path)  # Read csv file
        self._tag_names = df["name"].tolist()
        self._general_indexes = list(np.where(df["category"] == 0)[0])
        self._character_indexes = list(np.where(df["category"] == 4)[0])

        # Load other parameters
        self._input_name = self.model.get_inputs()[0].name
        self._label_name = self.model.get_outputs()[0].name
        self._size = self.model.get_inputs()[0].shape[1]

        if self.verbose:
            logu.info(f"label loaded.")

    def __call__(
        self,
        image: Union[StrPath, np.ndarray, Image.Image],
        general_threshold: float = 0.35,
        character_threshold: float = 0.35,
    ) -> str:

        batch_inputs = preprocess_image(image, self._size)

        # Run model
        with torch.no_grad():
            probs = self.model.run(
                [self._label_name],
                {self._input_name: batch_inputs}
            )[0]

        labels = list(zip(self._tag_names, probs[0].astype(float)))

        # Get general and character tags
        general_tags = get_tags(
            labels,
            self._general_indexes,
            general_threshold
        )
        character_tags = get_tags(
            labels,
            self._character_indexes,
            character_threshold
        )

        # Get top general tag
        general_tags = postprocess_tags(general_tags)
        character_tags = postprocess_tags(character_tags)

        caption = ', '.join(character_tags + general_tags)

        return caption


def preprocess_image(image, size: int):
    # 1. load image
    # convert to numpy array
    if isinstance(image, (str, Path)):
        image = imgu.load_image(image)
    elif isinstance(image, Image.Image):
        image = np.array(image)

    image = imgu.cvt2rgb(image)  # convert to RGB
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
    r"""Filter tags whose probability is beyond threshold and convert them to dictionary."""
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
    tags = [tag.replace('_', ' ') for tag in tags.keys()]

    return tags
