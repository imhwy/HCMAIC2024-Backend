"""
Service class for initializing and managing the CLIP retrieval system.
"""

import os
from dotenv import load_dotenv
import torch
from transformers import CLIPProcessor, AutoTokenizer, CLIPModel

from src.utils.utility import convert_value
from src.modules.original_clip import OriginalClip
from backend.src.repositories.load_faiss import OriginalClipFaiss
from src.repositories.load_json import LoadJson
from src.services.clip_retrieval import ClipRetrieval

load_dotenv()

ORIGINAL_CLIP_MODEL = convert_value(os.environ.get("ORIGINAL_CLIP_MODEL"))
ORIGINAL_CLIP_FAISS = convert_value(os.environ.get("ORIGINAL_CLIP_FAISS"))
TOP_K = convert_value(os.environ.get("TOP_K"))
JSON_CLIP = convert_value(os.environ.get("JSON_CLIP"))


class Service:
    """
    Initializes and provides access to the CLIP retrieval service.
    """

    def __init__(
        self,
        original_clip_model=ORIGINAL_CLIP_MODEL,
        original_clip_faiss=ORIGINAL_CLIP_FAISS,
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
            son_url=JSON_CLIP
        )
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._oc_model = CLIPModel.from_pretrained(
            original_clip_model
        ).to(self._device)
        self._oc_processor = CLIPProcessor.from_pretrained(original_clip_model)
        self._oc_tokenizer = AutoTokenizer.from_pretrained(original_clip_model)

        self._original_clip = OriginalClip(
            model=self._oc_model,
            processor=self._oc_processor,
            tokenizer=self._oc_tokenizer,
            device_type=self._device
        )
        self._oc_faiss = OriginalClipFaiss(
            faiss_url=original_clip_faiss
        )
        self._clip_retrieval = ClipRetrieval(
            top_k=top_k,
            oc_clip=self._original_clip,
            oc_faiss=self._oc_faiss,
            oc_data=self._oc_data.oc_data
        )

    @property
    def clip_retrieval(self):
        """
        Provides access to the initialized CLIP retrieval service.

        Returns:
            ClipRetrieval: The CLIP retrieval service instance.
        """
        return self._clip_retrieval
