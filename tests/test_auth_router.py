import datetime
import sys
import os
import uuid
import httpx
import pytest
from fastapi import HTTPException, status
from datetime import timedelta, datetime
from jose import jwt

from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.models.user import CreateUser
from api.routers.auth import get_current_user

from main import app

client = TestClient(app)


@pytest.mark.asyncio
async def test_authenticate_user_returns_user_when_password_correct():
    # Test user data
    username = "testuser"
    password = "testpassword"
    first_name = "testfirstname"
    last_name = "testlastname"
    book_preferences = []

    # Create a test user
    user_data = CreateUser(
        username=username,
        password=password,
        first_name=first_name,
        last_name=last_name,
        book_preferences=book_preferences,
    )
    user_data_dict = {
        k: str(v) if isinstance(v, uuid.UUID) else v
        for k, v in user_data.dict().items()
    }

    # Manually trigger the startup event
    await app.router.startup()

    async with httpx.AsyncClient(app=app, base_url="http://test") as client:
        user_response = await client.post("/auth/", json=user_data_dict)
        assert user_response.status_code == 201

        # Attempt to get an access token
        token_response = await client.post(
            "/auth/token",
            data={
                "username": username,
                "password": password,
                "grant_type": "password",
                "scope": "",
                "client_id": "",
                "client_secret": "",
            },
        )

        assert token_response.status_code == 200
        assert "access_token" in token_response.json()
        assert token_response.json()["token_type"] == "bearer"

        # Cleanup: delete the test user
        user_id = user_response.json()["_id"]
        await app.mongodb["users"].delete_one({"_id": user_id})

    # Manually trigger the shutdown event
    await app.router.shutdown()


@pytest.mark.asyncio
async def test_get_current_user_returns_username_and_id_when_jwt_token_valid():
    # Test user data
    username = "testuser"
    password = "testpassword"
    first_name = "testfirstname"
    last_name = "testlastname"
    book_preferences = []

    # Create a test user
    user_data = CreateUser(
        username=username,
        password=password,
        first_name=first_name,
        last_name=last_name,
        book_preferences=book_preferences,
    )
    user_data_dict = {
        k: str(v) if isinstance(v, uuid.UUID) else v
        for k, v in user_data.dict().items()
    }

    # Manually trigger the startup event
    await app.router.startup()

    async with httpx.AsyncClient(app=app, base_url="http://test") as client:
        user_response = await client.post("/auth/", json=user_data_dict)

        # Attempt to get an access token
        response = await client.post(
            "/auth/token",
            data={
                "username": username,
                "password": password,
                "grant_type": "password",
                "scope": "",
                "client_id": "",
                "client_secret": "",
            },
        )

        assert response.status_code == 200
        access_token = response.json()["access_token"]

        # Call the get_current_user function
        user = await get_current_user(access_token)

        # Verify the returned user data
        assert user["username"] == username

        # Cleanup: delete the test user
        user_id = user_response.json()["_id"]
        await app.mongodb["users"].delete_one({"_id": user_id})

    # Manually trigger the shutdown event
    await app.router.shutdown()


def create_invalid_access_token(expires_delta: timedelta):
    encode = {}
    expires = datetime.utcnow() + expires_delta
    encode.update({"exp": expires})

    return jwt.encode(encode, "invalid", algorithm="HS256")


@pytest.mark.asyncio
async def test_get_current_user_missing_payload():
    # Create a token with missing payload
    token = create_invalid_access_token(timedelta(minutes=20))

    # Call the get_current_user function and expect an HTTPException
    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(token)

    # Verify the exception details
    assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
    assert exc_info.value.detail == "Could not validate user."


def create_partial_access_token(missing_field: str, expires_delta: timedelta):
    encode = {"sub": "testuser", "_id": "testuserid"}
    expires = datetime.utcnow() + expires_delta
    encode.update({"exp": expires})

    if missing_field in encode:
        del encode[missing_field]

    return jwt.encode(encode, "invalid", algorithm="HS256")


@pytest.mark.asyncio
async def test_get_current_user_missing_username_or_user_id():
    # Create a token with missing username
    token_missing_username = create_partial_access_token("sub", timedelta(minutes=20))

    # Call the get_current_user function and expect an HTTPException
    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(token_missing_username)

    # Verify the exception details
    assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
    assert exc_info.value.detail == "Could not validate user."

    # Create a token with missing user_id
    token_missing_user_id = create_partial_access_token("_id", timedelta(minutes=20))

    # Call the get_current_user function and expect an HTTPException
    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(token_missing_user_id)

    # Verify the exception details
    assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
    assert exc_info.value.detail == "Could not validate user."
