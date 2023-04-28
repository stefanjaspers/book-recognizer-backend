# FastAPI
from fastapi import APIRouter, Body, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

# General
from typing import Annotated

# Models
from models.book import AddBookToUser
from .auth import get_current_user


router = APIRouter(prefix="/books", tags=["books"])

user_dependency = Annotated[dict, Depends(get_current_user)]


@router.post("/add", status_code=status.HTTP_201_CREATED)
async def add_book_to_user(
    user: user_dependency, request: Request, add_book: AddBookToUser = Body(...)
):
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication failed.")

    add_book = jsonable_encoder(add_book)

    new_book = await request.app.mongodb["books"].insert_one(add_book)
    added_book = await request.app.mongodb["books"].find_one(
        {"_id": new_book.inserted_id}
    )

    return JSONResponse(status_code=status.HTTP_201_CREATED, content=added_book)
