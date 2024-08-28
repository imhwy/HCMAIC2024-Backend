"""
Implements a FAISS-based search for CLIP embeddings.
"""

import faiss
from typing import List
from torch import Tensor


class ClipFaiss:
    """
    Handles FAISS-based search operations for CLIP embeddings.
    """

    def __init__(
        self,
        original_faiss_url: str,
        apple_faiss_url: str,
        laion_faiss_url: str
    ) -> None:
        """
        Initializes the FAISS index and loads it onto a GPU.

        Args:
            faiss_url (str): The path to the FAISS index file.
            device_type (device): The device type (e.g., 'cpu' or 'cuda').

        """
        self._original_index = faiss.read_index(original_faiss_url)
        self._original_res = faiss.StandardGpuResources()
        self._original_gpu_index = faiss.index_cpu_to_gpu(
            provider=self._original_res,
            device=0,
            index=self._original_index
        )
        self._apple_index = faiss.read_index(apple_faiss_url)
        self._apple_res = faiss.StandardGpuResources()
        self._apple_gpu_index = faiss.index_cpu_to_gpu(
            provider=self._apple_res,
            device=0,
            index=self._apple_index
        )
        self._laion_index = faiss.read_index(laion_faiss_url)
        self._laion_res = faiss.StandardGpuResources()
        self._laion_gpu_index = faiss.index_cpu_to_gpu(
            provider=self._laion_res,
            device=0,
            index=self._laion_index
        )

    async def original_search(
        self,
        top_k: int,
        query_vectors: Tensor
    ) -> List[int]:
        """
        """
        query_vectors = query_vectors.cpu().detach().numpy()
        _, indices = self._original_gpu_index.search(query_vectors, top_k)
        return indices

    async def apple_search(
        self,
        top_k: int,
        query_vectors: Tensor
    ) -> List[int]:
        """
        """
        query_vectors = query_vectors.cpu().detach().numpy()
        _, indices = self._apple_gpu_index.search(query_vectors, top_k)
        return indices

    async def laion_search(
        self,
        top_k: int,
        query_vectors: Tensor
    ) -> List[int]:
        """
        """
        query_vectors = query_vectors.cpu().detach().numpy()
        _, indices = self._laion_gpu_index.search(query_vectors, top_k)
        return indices
