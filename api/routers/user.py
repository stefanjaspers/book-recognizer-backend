from fastapi import APIRouter, Body, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse

from api.models.user import AddBookPreferences, UpdateBookPreferences
from .auth import get_current_user

router = APIRouter(prefix="/user", tags=["user"])


@router.post("/book_preferences", status_code=status.HTTP_200_OK)
async def add_book_preferences(request: Request, add_preferences: AddBookPreferences = Body(...), user: dict = Depends(get_current_user)):
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication failed.")

    user_id = user["_id"]
    preferences_to_add = add_preferences.preferences

    await request.app.mongodb["users"].update_one(
        {"_id": user_id},
        {"$addToSet": {"book_preferences": {"$each": preferences_to_add}}},
    )

    updated_user = await request.app.mongodb["users"].find_one({"_id": user_id})
    return JSONResponse(
        status_code=status.HTTP_200_OK, content=updated_user["book_preferences"]
    )


@router.put("/book_preferences", status_code=status.HTTP_200_OK)
async def update_book_preferences(request: Request, update_preferences: UpdateBookPreferences = Body(...), user: dict = Depends(get_current_user)):
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication failed.")

    user_id = user["_id"]
    new_preferences = update_preferences.preferences

    await request.app.mongodb["users"].update_one(
        {"_id": user_id}, {"$set": {"book_preferences": new_preferences}}
    )

    updated_user = await request.app.mongodb["users"].find_one({"_id": user_id})
    return JSONResponse(
        status_code=status.HTTP_200_OK, content=updated_user["book_preferences"]
    )
