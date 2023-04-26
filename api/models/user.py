from typing import Optional
from pydantic import BaseModel, Field, EmailStr
import uuid


class CreateUser(BaseModel):
    id: str = Field(default_factory=uuid.uuid4, alias="_id")
    username: str = Field(...)
    password: str = Field(...)
    first_name: str = Field(...)
    last_name: str = Field(...)

    class Config:
        allow_population_by_field_name = True
        schema_extra = {
            "example": {
                "username": "johndoe",
                "password": "test1234",
                "first_name": "John",
                "last_name": "Doe",
            }
        }


class UpdateUser(BaseModel):
    password: Optional[str]
    first_name: Optional[str]
    last_name: Optional[str]

    class Config:
        schema_extra = {
            "example": {
                "password": "Pancakes",
                "first_name": "Dohn",
                "last_name": "Joe",
            }
        }
