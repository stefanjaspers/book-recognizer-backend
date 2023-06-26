from typing import Optional, List
from pydantic import BaseModel, Field
import uuid


class CreateUser(BaseModel):
    id: str = Field(default_factory=uuid.uuid4, alias="_id")
    username: str = Field(...)
    password: str = Field(...)
    first_name: str = Field(...)
    last_name: str = Field(...)
    book_preferences: List[str]

    class Config:
        allow_population_by_field_name = True
        schema_extra = {
            "example": {
                "username": "stefanjaspers",
                "password": "guitarhero99",
                "first_name": "Stefan",
                "last_name": "Jaspers",
                "book_preferences": [],
            }
        }


class UpdateUser(BaseModel):
    password: Optional[str]
    first_name: Optional[str]
    last_name: Optional[str]
    book_preferences: Optional[List[str]]

    class Config:
        schema_extra = {
            "example": {
                "password": "Pancakes",
                "first_name": "Dohn",
                "last_name": "Joe",
                "book_preferences": ["Family", "Comedy", "Drama"],
            }
        }


class GetBookPreferences(BaseModel):
    preferences: List[str]

    class Config:
        schema_extra = {"example": {"preferences": ["Mystery", "Science Fiction"]}}


class AddBookPreferences(BaseModel):
    preferences: List[str]

    class Config:
        schema_extra = {"example": {"preferences": ["Mystery", "Science Fiction"]}}


class UpdateBookPreferences(BaseModel):
    preferences: List[str]

    class Config:
        schema_extra = {"example": {"preferences": ["Comedy", "Drama"]}}
