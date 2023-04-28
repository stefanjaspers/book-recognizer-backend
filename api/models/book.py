from typing import List
from pydantic import BaseModel, Field
import uuid


"""
Represents the model for adding a book to a user.
"""


class AddBookToUserModel(BaseModel):
    id: str = Field(default_factory=uuid.uuid4, alias="_id")
    isbn: str = Field(...)
    title: str = Field(...)
    author: str = Field(...)
    genres: List[str] = Field(...)
    user: str

    class Config:
        allow_population_by_field_name = True
        schema_extra = {
            "example": {
                "isbn": "978-3-16-148410-0",
                "title": "Ruud in de Mini-Disco",
                "author": "Ruud Hermans",
                "genres": ["Family", "Comedy", "Drama"],
                "user": "",
            }
        }


"""
Represents the model for removing a book from a user.
"""


class RemoveBookFromUserModel(BaseModel):
    id: str = Field(alias="_id")
