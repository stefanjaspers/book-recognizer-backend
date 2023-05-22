# FastAPI
from fastapi import FastAPI

# MongoDB driver
from motor.motor_asyncio import AsyncIOMotorClient
from api.config import mongo_config

# Router imports
from api.routers import auth 
from api.routers import book

app = FastAPI()


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
