"""
"""

from typing import List, Dict
from src.modules.original_clip import OriginalCLIP
from src.modules.apple_clip import AppleCLIP
from src.modules.laion_clip import LaionCLIP
from src.repositories.load_faiss import ClipFaiss


class MultiEventRetrieval:
    """
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
        """
        if model_type == "original_clip":
            return await self.original_text_retrieval(
                text=text
            )
        elif model_type == "apple_clip":
            return await self.apple_text_retrieval(
                text=text
            )
        elif model_type == "laion_clip":
            return await self.laion_text_retrieval(
                text=text
            )
        else:
            return {
                "error": "Model type not supported"
            }

    async def find_common_elements_by_field(
        self,
        list_event: List[Dict],
        field: str = "video_id",
        frame_field: str = "frame_id"
    ) -> List[Dict]:
        """
        Find objects with the same video_id and ensure the frame_id (as a string) 
        increases across lists.
        """
        base_list = list_event[0]
        common_elements = []

        def extract_frame_number(frame_id: str) -> int:
            return int(frame_id.split('.')[0])
        for item in base_list:
            video_id_value = item[field]
            frame_id_value = extract_frame_number(item[frame_field])
            if all(
                any(
                    d[field] == video_id_value and extract_frame_number(
                        d[frame_field]) > frame_id_value
                    for d in lst
                ) for lst in list_event[1:]
            ):
                for _, lst in enumerate(list_event):
                    for d in lst:
                        if d[field] == video_id_value:
                            common_elements.append(d)
                            frame_id_value = extract_frame_number(
                                d[frame_field])

        half_size = len(common_elements) // 2
        return common_elements[:half_size]

    async def multi_event_search(
        self,
        model_type: str,
        list_event: List[str]
    ) -> List[Dict]:
        list_result = []
        for event in list_event:
            result = await self.text_retrieval(
                model_type=model_type,
                text=event
            )
            list_result.append(result)
        result = await self.find_common_elements_by_field(
            list_event=list_result,
            field="video_id"
        )
        return result
