from pydantic import BaseModel, Field

class Token(BaseModel):
    access_token: str = Field(...)
    token_type: str = Field(...)