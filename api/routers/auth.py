# FastAPI
from fastapi import APIRouter, Body, Depends, Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi.security import OAuth2PasswordRequestForm

# Cryptography
from passlib.context import CryptContext

# Models
from models.user import CreateUser, UpdateUser

# General
from typing import Annotated

router = APIRouter(prefix="/auth", tags=["auth"])

bcrypt_context = CryptContext(schemes=["bcrypt"])


async def authenticate_user(username: str, password: str, request: Request):
    user = await request.app.mongodb["users"].find_one({"username": username})

    print(user)

    if not user:
        return False
    if not bcrypt_context.verify(password, user["password"]):
        return False
    return True


@router.post("/", response_description="Create new user.")
async def create_user(request: Request, create_user: CreateUser = Body(...)):
    create_user = jsonable_encoder(create_user)

    # Hash password
    create_user["password"] = bcrypt_context.hash(create_user["password"])

    new_user = await request.app.mongodb["users"].insert_one(create_user)
    created_user = await request.app.mongodb["users"].find_one(
        {"_id": new_user.inserted_id}
    )

    return JSONResponse(status_code=status.HTTP_201_CREATED, content=created_user)


@router.post("/token")
async def login_for_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()], request: Request
):
    user = await authenticate_user(form_data.username, form_data.password, request)

    if not user:
        return "Failed authentication."
    return "Successful authentication."
