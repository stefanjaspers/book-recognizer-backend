import sys
import os
import uuid
import httpx
import pytest

from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.models.user import CreateUser

from main import app

client = TestClient(app)


@pytest.mark.asyncio
async def test_authenticate_user_integration():
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
