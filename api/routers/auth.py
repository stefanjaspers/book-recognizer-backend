from fastapi import APIRouter, Body, Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from passlib.context import CryptContext

from models.user import CreateUser, UpdateUser

router = APIRouter(prefix="/auth", tags=["auth"])

bcrypt_context = CryptContext(schemes=["bcrypt"])


@router.post("/", response_description="Create new user.")
async def create_user(request: Request, create_user: CreateUser = Body(...)):
    create_user = jsonable_encoder(create_user)
    
    # Hash password
    create_user["hashed_password"] = bcrypt_context.hash(create_user["hashed_password"])

    new_user = await request.app.mongodb["users"].insert_one(create_user)
    created_user = await request.app.mongodb["users"].find_one(
        {"_id": new_user.inserted_id}
    )

    return JSONResponse(status_code=status.HTTP_201_CREATED, content=created_user)
