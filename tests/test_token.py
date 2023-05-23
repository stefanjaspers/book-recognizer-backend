import os
import sys
import pytest
from pydantic import ValidationError

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.models.token import Token

def test_token_creation():
    token = Token(access_token="my_access_token", token_type="bearer")
    assert token.access_token == "my_access_token"
    assert token.token_type == "bearer"

def test_token_missing_access_token():
    with pytest.raises(ValidationError):
        Token(token_type="bearer")

def test_token_missing_token_type():
    with pytest.raises(ValidationError):
        Token(access_token="my_access_token")

def test_token_missing_both_fields():
    with pytest.raises(ValidationError):
        Token()