import boto3
import os
from dotenv import load_dotenv

# FastAPI
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

# MongoDB driver
from motor.motor_asyncio import AsyncIOMotorClient
from api.config import mongo_config

# Router imports
from api.routers import auth
from api.routers import book
from api.routers import user

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()

boto3.setup_default_session(
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    region_name=os.environ["AWS_REGION"],
)


@app.on_event("startup")
async def startup_mongo_client():
    print("Instantiating MongoDB connection...")
    app.mongodb_client = AsyncIOMotorClient(mongo_config.MONGO_URI)
    app.mongodb = app.mongodb_client[mongo_config.MONGO_DB_NAME]
    print("MongoDB connection established successfully.")


@app.on_event("shutdown")
async def shutdown_mongo_client():
    print("Closing MongoDB connection...")
    app.mongodb_client.close()
    print("MongoDB connection closed successfully.")


app.include_router(auth.router)
app.include_router(book.router)
app.include_router(user.router)


@app.get("/")
async def index():
    return {"Text": "Welcome to the Book Recognizer Backend!"}
