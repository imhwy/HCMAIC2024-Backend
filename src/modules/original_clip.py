"""
Implements CLIP model-based text and image embedding.
"""

from PIL import Image
from torch import device, Tensor
from transformers import AutoTokenizer, AutoProcessor, CLIPModel


class OriginalCLIP:
    """
    Handles text and image embeddings using the CLIP model.
    """

    def __init__(
        self,
        model: CLIPModel,
        processor: AutoProcessor,
        tokenizer: AutoTokenizer,
        device_type: device
    ) -> None:
        """
        Initializes the OriginalClip with the provided model, processor, tokenizer, and device.

        Args:
            model (CLIPModel): The CLIP model for embedding.
            processor (AutoProcessor): Processor for image data.
            tokenizer (AutoTokenizer): Tokenizer for text data.
            device_type (device): The device type (e.g., 'cpu' or 'cuda').
        """
        self._model = model
        self._processor = processor
        self._tokenizer = tokenizer
        self._device_type = device_type

    async def text_embedding(
        self,
        text: str
    ) -> Tensor:
        """
        Generates a text embedding for the given input text.

        Args:
            text (str): The input text to embed.

        Returns:
            Tensor: The text embedding as a tensor.
        """
        inputs = self._tokenizer(
            text,
            padding=True,
            return_tensors="pt"
        ).to(self._device_type)
        text_features = self._model.get_text_features(**inputs)
        return text_features

    async def image_embedding(
        self,
        image
    ) -> Tensor:
        """
        Generates an image embedding for the given input image.

        Args:
            image: The input image to embed.

        Returns:
            Tensor: The image embedding as a tensor.
        """
        image = Image.open(image).convert("RGB")
        inputs = self._processor(
            images=image,
            return_tensors="pt"
        ).to(self._device_type)
        image_features = self._model.get_image_features(**inputs)
        return image_features
