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
    subtitle: str = Field(...)
    authors: List[str] = Field(...)
    publisher: str = Field(...)
    published_date: str = Field(...)
    description: str = Field(...)
    genres: List[str] = Field(...)
    thumbnail: str = Field(...)
    user_id: str

    class Config:
        allow_population_by_field_name = True
        schema_extra = {
            "example": {
                "isbn": "9780393059311",
                "title": "Lend Me Your Ears",
                "subtitle": "Great Speeches In History",
                "authors": ["William Safire"],
                "publisher": "W. W. Norton & Company",
                "published_date": "2004-10-05",
                "description": "A compendium of more than two hundred classic and modern speeches includes Orson Welles eulogizing Darryl F. Zanuck, George Patton exhorting his D-Day troops, etc.",
                "genres": ["Reference"],
                "thumbnail": "http://books.google.com/books/content?id=EKkO4JBxtVkC&printsec=frontcover&img=1&zoom=1&source=gbs_api",
                "user_id": "",
            }
        }


"""
Represents the model for removing a book from a user.
"""


class RemoveBookFromUserModel(BaseModel):
    id: str = Field(alias="_id")
