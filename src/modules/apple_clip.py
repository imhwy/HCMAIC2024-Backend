"""
"""

import torch
from torch import device, Tensor
import torch.nn.functional as F
from PIL import Image
from open_clip.factory import (create_model,
                               image_transform_v2,
                               get_tokenizer)

class AppleCLIP:
    """
    """
    def __init__(
        self,
        model: create_model,
        processor: image_transform_v2,
        tokenizer: get_tokenizer,
        device_type: device
    ) -> None:
        """
        """
        self._model = model
        self._processor = processor
        self._tokenizer = tokenizer
        self._device_type = device_type
        
    def text_embedding(
        self,
        text: str
    ) -> Tensor:
        """
        """
        text = self._tokenizer(
            text,
            context_length=self._model.context_length
        )
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_features = self._model.encode_text(text)
            text_features = F.normalize(text_features, dim=-1)
        return text_features

    def image_embedding(
        self,
        image
    ) -> Tensor:
        """
        """
        image = Image.open(image).convert("RGB")
        image = self._processor(image).unsqueeze(0).to(self._device_type)
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self._model.encode_image(image)
            image_features = F.normalize(image_features, dim=-1)
        return image_features
