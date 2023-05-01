# FastAPI
from fastapi import APIRouter, Body, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

# General
from typing import Annotated

# Models
from models.book import AddBookToUserModel
from models.book import RemoveBookFromUserModel
from .auth import get_current_user


router = APIRouter(prefix="/books", tags=["books"])

user_dependency = Annotated[dict, Depends(get_current_user)]


@router.post("/add", status_code=status.HTTP_201_CREATED)
async def add_book_to_user(
    user: user_dependency, request: Request, add_book: AddBookToUserModel = Body(...)
):
    add_book = jsonable_encoder(add_book)

    # Check if user is authenticated
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication failed.")

    add_book["user"] = user["_id"]

    # Insert new book into books collection
    new_book = await request.app.mongodb["books"].insert_one(add_book)

    # Fetch id of newly inserted book
    added_book = await request.app.mongodb["books"].find_one(
        {"_id": new_book.inserted_id}
    )

    # Return status code 201 and the id of the newly inserted book
    return JSONResponse(status_code=status.HTTP_201_CREATED, content=added_book)


@router.delete("/delete", status_code=status.HTTP_204_NO_CONTENT)
async def remove_book_from_user(
    user: user_dependency,
    request: Request,
    remove_book: RemoveBookFromUserModel = Body(...),
):
    remove_book = jsonable_encoder(remove_book)

    # Check if user is authenticated
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication failed."
        )

    # Find the book by its id
    delete_book = await request.app.mongodb["books"].find_one(
        {"_id": remove_book["_id"]}
    )

    # Raise 404 if book doesn't exist
    if not delete_book:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Unable to find book."
        )

    # Make sure only the owner of the book can delete it
    if user["_id"] != delete_book["user"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You are not allowed to delete this book.",
        )

    await request.app.mongodb["books"].delete_one(delete_book)

    return None
