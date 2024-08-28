"""
This module defines a FastAPI router for handling clip text retrieval requests.
"""

import io
import time
from fastapi import (status,
                     Depends,
                     APIRouter,
                     HTTPException,
                     UploadFile,
                     File)

from src.api.schemas.clip import (RequestClipText,
                                  ResponseClip,
                                  ListResponseClip)
from src.services.service import Service
from src.api.dependencies.dependency import get_service


clip_router = APIRouter(
    tags=["Clip"],
    prefix="/clip",
)


@clip_router.post(
    '/clipTextRetrieval',
    status_code=status.HTTP_200_OK,
    response_model=ListResponseClip
)
async def clip_text_retrieval(
    request: RequestClipText,
    service: Service = Depends(get_service)
) -> ListResponseClip:
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
        result = await service.text_clip_retrieval.text_retrieval(
            model_type=request.model_type,
            text=request.text
        )
        b = time.time()
        print(b-a)
        return ListResponseClip(
            data=[
                ResponseClip(**record) for record in result
            ]
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)) from e


@clip_router.post(
    "/searchByImage",
    status_code=status.HTTP_200_OK,
    response_model=ListResponseClip)
async def search_by_image(
    model_type: str,
    file: UploadFile = File(...),
    service: Service = Depends(get_service)
) -> ListResponseClip:
    """
    Perform a search using an uploaded image.

    Args:
        file (UploadFile): The image file to search with.
        service (Service): The service instance used for performing the search.

    Returns:
        ResponseResult: An object containing the search results.

    Raises:
        HTTPException: If no file is provided or if an error occurs during processing.
    """
    if not file.file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Image is required"
        )
    try:
        contents = await file.read()
        image_stream = io.BytesIO(contents)
        result = await service.image_clip_retrieval.image_retrieval(
            model_type=model_type,
            image=image_stream
        )
        return ListResponseClip(
            data=[
                ResponseClip(**record)for record in result
            ]
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        ) from e
