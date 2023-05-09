from pydantic import BaseModel, Field
from typing import List


"""
Represents how book documents will be stored in the MongoDB database.
"""


class BookSchema(BaseModel):
    id: str = Field(alias="_id")
    isbn: str
    title: str
    subtitle: str
    authors: List[str]
    publisher: str
    published_date: str
    description: str
    genres: List[str]
    thumbnail: str
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
