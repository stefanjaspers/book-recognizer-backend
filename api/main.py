# FastAPI
from fastapi import FastAPI
import uvicorn

# MongoDB driver
from motor.motor_asyncio import AsyncIOMotorClient
from config import mongo_config

# Router imports
from routers import auth, user


app = FastAPI()


@app.on_event("startup")
async def startup_mongo_client():
    app.mongodb_client = AsyncIOMotorClient(mongo_config.MONGO_URI)
    app.mongodb = app.mongodb_client[mongo_config.MONGO_DB_NAME]


@app.on_event("shutdown")
async def shutdown_mongo_client():
    app.mongodb_client.close()


app.include_router(auth.router)
app.include_router(user.router)
