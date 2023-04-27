# FastAPI
from fastapi import APIRouter, Body, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer

# Cryptography
from passlib.context import CryptContext
from jose import jwt, JWTError

# Models
from models.user import CreateUser

# General
import os
from typing import Annotated
from datetime import timedelta, datetime
from dotenv import load_dotenv

router = APIRouter(prefix="/auth", tags=["auth"])

bcrypt_context = CryptContext(schemes=["bcrypt"])

oauth2_bearer = OAuth2PasswordBearer(tokenUrl="auth/token")

load_dotenv()
JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY")
JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM")


async def authenticate_user(username: str, password: str, request: Request):
    user = await request.app.mongodb["users"].find_one({"username": username})

    if not user:
        return False
    if not bcrypt_context.verify(password, user["password"]):
        return False
    return user


def create_access_token(username: str, _id: int, expires_delta: timedelta):
    encode = {"sub": username, "_id": _id}
    expires = datetime.utcnow() + expires_delta
    encode.update({"exp": expires})

    return jwt.encode(encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


async def get_current_user(token: Annotated[str, Depends(oauth2_bearer)]):
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        username: str = payload.get("sub")
        user_id: int = payload.get("_id")

        if username is None or user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate user.",
            )
        return {"username": username, "_id": user_id}
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate user."
        )


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
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate user."
        )

    token = create_access_token(user["username"], user["_id"], timedelta(minutes=20))

    return {"access_token": token, "token_type": "bearer"}
