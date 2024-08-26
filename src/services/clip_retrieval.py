"""
Implements text retrieval using CLIP embeddings and FAISS index.
"""

from typing import List, Dict
from src.modules.original_clip import OriginalCLIP
from src.modules.apple_clip import AppleCLIP
from src.modules.laion_clip import LaionCLIP
from src.repositories.load_faiss import ClipFaiss


class ClipRetrieval:
    """
    Handles retrieval of text data using CLIP embeddings and FAISS index.
    """

    def __init__(
        self,
        top_k: int,
        original_clip: OriginalCLIP,
        apple_clip: AppleCLIP,
        laion_clip: LaionCLIP,
        faiss: ClipFaiss,
        data: Dict
    ) -> None:
        """
        """
        self._top_k = top_k
        self._original_clip = original_clip
        self._apple_clip = apple_clip
        self._laion_clip = laion_clip
        self._faiss = faiss
        self._data = data

    async def mapping_results(
        self,
        data: Dict,
        indices: List[int]
    ) -> List:
        """
        """
        filtered_list = [data[indice] for indice in indices if indice in data]
        return filtered_list

    async def original_text_retrieval(
        self,
        text: str
    ) -> List[Dict]:
        """
        Retrieves text data using the original CLIP model.

        Args:
            text (str): The input text to retrieve data for.

        Returns:
            List[Dict]: A list of dictionaries containing the retrieval results.
        """
        vector_embedding = await self._original_clip.text_embedding(
            text=text
        )
        indices = await self._faiss.original_search(
            top_k=self._top_k,
            query_vectors=vector_embedding
        )
        result = await self.mapping_results(
            data=self._data,
            indices=indices[0]
        )
        return result

    async def apple_text_retrieval(
        self,
        text: str
    ) -> List[Dict]:
        """
        Retrieves text data using the apple CLIP model.

        Args:
            text (str): The input text to retrieve data for.

        Returns:
            List[Dict]: A list of dictionaries containing the retrieval results.
        """
        vector_embedding = await self._apple_clip.text_embedding(
            text=text
        )
        indices = await self._faiss.apple_search(
            top_k=self._top_k,
            query_vectors=vector_embedding
        )
        result = await self.mapping_results(
            data=self._data,
            indices=indices[0]
        )
        return result

    async def laion_text_retrieval(
        self,
        text: str
    ) -> List[Dict]:
        """
        Retrieves text data using the laion CLIP model.

        Args:
            text (str): The input text to retrieve data for.

        Returns:
            List[Dict]: A list of dictionaries containing the retrieval results.
        """
        vector_embedding = await self._laion_clip.text_embedding(
            text=text
        )
        indices = await self._faiss.laion_search(
            top_k=self._top_k,
            query_vectors=vector_embedding
        )
        result = await self.mapping_results(
            data=self._data,
            indices=indices[0]
        )
        return result

    async def text_retrieval(
        self,
        model_type: str,
        text: str
    ) -> List[Dict]:
        """
        Retrieves text data based on the specified model type.

        Args:
            model_type (str): The type of model to use for retrieval.
            text (str): The input text to retrieve data for.

        Returns:
            List[Dict]: A list of dictionaries containing the retrieval results.
        """
        if model_type == "original_clip":
            return await self.original_text_retrieval(text=text)
        elif model_type == "apple_clip":
            return await self.apple_text_retrieval(text=text)
        elif model_type == "laion_clip":
            return await self.laion_text_retrieval(text=text)
        else:
            return {
                "error": "Model type not supported"
            }
