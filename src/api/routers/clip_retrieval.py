"""
This module defines a FastAPI router for handling clip text retrieval requests.
"""

import time
from fastapi import (status,
                     Depends,
                     APIRouter,
                     HTTPException)

from src.api.schemas.clip import (RequestClipText,
                                  ResponseClipText,
                                  ListResponseClipText)
from src.services.service import Service
from src.api.dependencies.dependency import get_service


clip_router = APIRouter(
    tags=["Clip"],
    prefix="/clip",
)


@clip_router.post(
    '/clipTextRetrieval',
    status_code=status.HTTP_200_OK,
    response_model=ListResponseClipText
)
async def clip_text_retrieval(
    request: RequestClipText,
    service: Service = Depends(get_service)
) -> ListResponseClipText:
    """
    Retrieves relevant text clips based on the provided query.

    Args:
        request (RequestClipText): The input data containing the text query and model type.
        service (Service): The service instance to handle the clip retrieval logic.

    Returns:
        ListResponseClipText: A list of text clips relevant to the input query.

    Raises:
        HTTPException: If the input text query is missing or an error occurs during processing.
    """
    if not request.text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query is required"
        )
    try:
        a = time.time()
        result = await service.clip_retrieval.text_retrieval(
            model_type=request.model_type,
            text=request.text
        )
        b = time.time()
        print(b-a)
        return ListResponseClipText(
            data=[
                ResponseClipText(**record) for record in result
            ]
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)) from e
