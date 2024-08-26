"""
Service class for initializing and managing the CLIP retrieval system.
"""

import os
from dotenv import load_dotenv
import torch
from open_clip import (create_model_from_pretrained,
                       get_tokenizer)
from transformers import (CLIPProcessor,
                          AutoTokenizer,
                          CLIPModel)

from src.utils.utility import convert_value
from src.modules.original_clip import OriginalCLIP
from src.modules.apple_clip import AppleCLIP
from src.modules.laion_clip import LaionCLIP
from src.repositories.load_faiss import ClipFaiss
from src.repositories.load_json import LoadJson
from src.services.text_clip_retrieval import TextClipRetrieval
from src.services.image_clip_retrieval import ImageClipRetrieval

load_dotenv()

ORIGINAL_CLIP_MODEL = convert_value(os.environ.get("ORIGINAL_CLIP_MODEL"))
APPLE_CLIP_MODEL = convert_value(os.environ.get("APPLE_CLIP_MODEL"))
APPLE_CLIP_TOKENIZER = convert_value(os.environ.get("APPLE_CLIP_TOKENIZER"))
LAION_CLIP_MODEL = convert_value(os.environ.get("LAION_CLIP_MODEL"))
LAION_CLIP_TOKENIZER = convert_value(os.environ.get("LAION_CLIP_TOKENIZER"))
ORIGINAL_FAISS = convert_value(os.environ.get("ORIGINAL_FAISS"))
APPLE_FAISS = convert_value(os.environ.get("APPLE_FAISS"))
LAION_FAISS = convert_value(os.environ.get("LAION_FAISS"))
TOP_K = convert_value(os.environ.get("TOP_K"))
JSON_CLIP = convert_value(os.environ.get("JSON_CLIP"))


class Service:
    """
    Initializes and provides access to the CLIP retrieval service.
    """

    def __init__(
        self,
        original_clip_model=ORIGINAL_CLIP_MODEL,
        apple_clip_model=APPLE_CLIP_MODEL,
        apple_clip_tokenizer=APPLE_CLIP_TOKENIZER,
        laion_clip_model=LAION_CLIP_MODEL,
        laion_clip_tokenizer=LAION_CLIP_TOKENIZER,
        original_clip_faiss=ORIGINAL_FAISS,
        apple_clip_faiss=APPLE_FAISS,
        laion_clip_faiss=LAION_FAISS,
        json_clip=JSON_CLIP,
        top_k=TOP_K
    ) -> None:
        """
        Sets up the necessary components for the CLIP retrieval service.

        Args:
            original_clip_model (str): The path or identifier for the CLIP model.
            original_clip_faiss (str): The path to the FAISS index file.
            top_k (int): The number of top results to return during retrieval.
        """
        self._data = LoadJson(
            json_url=json_clip
        )
        # self._device = torch.device(
        #     "cuda" if torch.cuda.is_available() else "cpu"
        # )
        self._device = torch.device("cpu")
        self._oc_model = CLIPModel.from_pretrained(
            original_clip_model
        ).to(self._device)
        self._oc_processor = CLIPProcessor.from_pretrained(original_clip_model)
        self._oc_tokenizer = AutoTokenizer.from_pretrained(original_clip_model)
        self._apple_model, self._apple_processor = create_model_from_pretrained(
            apple_clip_model
        )
        self._apple_model.to(self._device)
        self._apple_model = get_tokenizer(apple_clip_tokenizer)
        self._laion_model, self._laion_processor = create_model_from_pretrained(
            laion_clip_model
        )
        self._laion_modelbranch
        .to(self._device)
        self._laion_tokenizer = get_tokenizer(laion_clip_tokenizer)
        self._original_clip = OriginalCLIP(
            model=self._oc_model,
            processor=self._oc_processor,
            tokenizer=self._oc_tokenizer,
            device_type=self._device
        )
        self._apple_clip = AppleCLIP(
            model=self._apple_model,
            processor=self._apple_processor,
            tokenizer=self._apple_model,
            device_type=self._device
        )
        self._laion_clip = LaionCLIP(
            model=self._laion_model,
            processor=self._laion_processor,
            tokenizer=self._laion_tokenizer,
            device_type=self._device
        )
        self._faiss = ClipFaiss(
            original_faiss_url=original_clip_faiss,
            apple_faiss_url=apple_clip_faiss,
            laion_faiss_url=laion_clip_faiss
        )
        self._text_clip_retrieval = TextClipRetrieval(
            top_k=top_k,
            original_clip=self._original_clip,
            apple_clip=self._apple_clip,
            laion_clip=self._laion_clip,
            faiss=self._faiss,
            data=self._data
        )
        self._image_clip_retrieval = ImageClipRetrieval(
            top_k=top_k,
            original_clip=self._original_clip,
            apple_clip=self._apple_clip,
            laion_clip=self._laion_clip,
            faiss=self._faiss,
            data=self._data
        )

    @property
    def text_clip_retrieval(self):
        """
        Provides access to the initialized CLIP retrieval service.

        Returns:
            ClipRetrieval: The CLIP retrieval service instance.
        """
        return self._text_clip_retrieval

    @property
    def image_clip_retrieval(self):
        """
        """
        return self._image_clip_retrieval
