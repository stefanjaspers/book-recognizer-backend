from typing import Optional
from pydantic import BaseModel, Field, EmailStr
import uuid


class CreateUser(BaseModel):
    id: str = Field(default_factory=uuid.uuid4, alias="_id")
    email: EmailStr = Field(...)
    hashed_password: str = Field(...)
    first_name: str = Field(...)
    last_name: str = Field(...)

    class Config:
        allow_population_by_field_name = True
        schema_extra = {
            "example": {
                "id": "00010203-0405-0607-0809-0a0b0c0d0e0f",
                "email": "johndoe@email.com",
                "hashed_password": "$2y$04$.8qp4YMYpv.DPz/JBONAweC5cRn9JG/pqlIreJUVwqsnmoZVswOAa",
                "first_name": "John",
                "last_name": "Doe",
            }
        }


class UpdateUser(BaseModel):
    email: Optional[EmailStr]
    password: Optional[str]
    first_name: Optional[str]
    last_name: Optional[str]

    class Config:
        schema_extra = {
            "example": {
                "email": "dohnjoe@email.com",
                "password": "Pancakes",
                "first_name": "Dohn",
                "last_name": "Joe",
            }
        }
