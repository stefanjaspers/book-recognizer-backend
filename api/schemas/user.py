from pydantic import BaseModel, Field
from typing import List


"""
Represents how book documents will be stored in the MongoDB database.
"""


class UserSchema(BaseModel):
    id: str = Field(alias="_id")
    username: str
    password: str
    first_name: str
    last_name: str
    book_preferences: List[str]

    class Config:
        allow_population_by_field_name = True
        schema_extra = {
            "example": {
                "_id": "8c566338-92e7-4847-84ea-4cc21f6118ec",
                "username": "stefanjaspers",
                "password": "guitarhero99",
                "first_name": "Stefan",
                "last_name": "Jaspers",
                "book_preferences": []
            }
        }