from typing import Optional
from pydantic import BaseModel, Field, EmailStr
import uuid


class User(BaseModel):
    id: str = Field(default_factory=uuid.uuid4, alias="_id")
    email: EmailStr = Field(...)
    first_name: str = Field(...)
    last_name: str = Field(...)

    class Config:
        allow_population_by_field_name = True
        schema_extra = {
            "example": {
                "id": "00010203-0405-0607-0809-0a0b0c0d0e0f",
                "email": "johndoe@email.com",
                "first_name": "John",
                "last_name": "Doe",
            }
        }


class UpdateUser(BaseModel):
    email: Optional[EmailStr]
    first_name: Optional[str]
    last_name: Optional[str]

    class Config:
        schema_extra = {
            "example": {
                "email": "dohnjoe@email.com",
                "first_name": "Dohn",
                "last_name": "Joe",
            }
        }
