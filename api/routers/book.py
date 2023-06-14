# FastAPI.
from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

# General.
from typing import Annotated
from pydantic import BaseModel

from .auth import get_current_user

# Services.
from api.services.book_recognition_service import BookRecognitionService

# Instantiate services.
book_recognition_service = BookRecognitionService()

router = APIRouter(prefix="/books", tags=["books"])

user_dependency = Annotated[dict, Depends(get_current_user)]


class ImageInput(BaseModel):
    image: str


@router.post("/recognize", status_code=status.HTTP_200_OK)
async def recognize_books(image_input: ImageInput):
    image = image_input.image
    book_texts = await book_recognition_service.recognize(image)
    return JSONResponse(status_code=status.HTTP_200_OK, content=book_texts)
