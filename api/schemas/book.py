from pydantic import BaseModel, Field
from typing import List


"""
Represents how book documents will be stored in the MongoDB database.
"""


class BookSchema(BaseModel):
    id: str = Field(alias="_id")
    isbn: str
    title: str
    author: str
    genres: List[str]
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
