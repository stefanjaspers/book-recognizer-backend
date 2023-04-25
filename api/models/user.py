from pydantic import BaseModel
from uuid import uuid4

class User(BaseModel):
    email: str
    first_name: str
    last_name: str