"""
Implements text retrieval using CLIP embeddings and FAISS index.
"""

from io import BytesIO
from typing import List, Dict

from src.modules.original_clip import OriginalCLIP
from src.modules.apple_clip import AppleCLIP
from src.modules.laion_clip import LaionCLIP
from src.repositories.load_faiss import ClipFaiss


class ImageClipRetrieval:
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

    async def original_image_retrieval(
        self,
        image: BytesIO
    ) -> List[Dict]:
        """
        Retrieves text data using the original CLIP model.

        Args:
            text (str): The input text to retrieve data for.

        Returns:
            List[Dict]: A list of dictionaries containing the retrieval results.
        """
        vector_embedding = await self._original_clip.image_embedding(
            image=image
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

    async def apple_image_retrieval(
        self,
        image: BytesIO
    ) -> List[Dict]:
        """
        Retrieves text data using the apple CLIP model.

        Args:
            text (str): The input text to retrieve data for.

        Returns:
            List[Dict]: A list of dictionaries containing the retrieval results.
        """
        vector_embedding = await self._apple_clip.image_embedding(
            image=image
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

    async def laion_image_retrieval(
        self,
        image: BytesIO
    ) -> List[Dict]:
        """
        Retrieves text data using the laion CLIP model.

        Args:
            text (str): The input text to retrieve data for.

        Returns:
            List[Dict]: A list of dictionaries containing the retrieval results.
        """
        vector_embedding = await self._laion_clip.image_embedding(
            image=image
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
        image: BytesIO
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
            return await self.original_image_retrieval(
                image=image
            )
        if model_type == "apple_clip":
            return await self.apple_image_retrieval(
                image=image
            )
        if model_type == "laion_clip":
            return await self.laion_image_retrieval(
                image=image
            )
        return {
            "error": "Model type not supported"
        }
