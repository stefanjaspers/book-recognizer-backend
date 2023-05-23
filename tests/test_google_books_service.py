import os
import sys
import pytest
import httpx

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.services.google_books_service import GoogleBooksService

# Sample data for testing
book_texts = ["The Catcher in the Rye", "To Kill a Mockingbird", "Pride and Prejudice"]
processed_book_texts = [
    "The Catcher in the Rye",
    "To Kill a Mockingbird",
    "Pride and Prejudice",
]
encoded_book_texts = [
    "The+Catcher+in+the+Rye",
    "To+Kill+a+Mockingbird",
    "Pride+and+Prejudice",
]

@pytest.fixture
def google_books_service():
    return GoogleBooksService()

def test_remove_non_english_chars(google_books_service):
    text = "Hello, World! こんにちは 12345"
    expected_output = "Hello, World! 12345"
    assert google_books_service.remove_non_english_chars(text) == expected_output

def test_process_book_texts(google_books_service):
    assert google_books_service.process_book_texts(book_texts) == processed_book_texts

def test_url_encode_strings(google_books_service):
    assert google_books_service.url_encode_strings(processed_book_texts) == encoded_book_texts

def test_create_api_urls(google_books_service):
    base_url = google_books_service.GOOGLE_BOOKS_BASE_URL
    url_suffix = google_books_service.GOOGLE_BOOKS_URL_SUFFIX
    expected_output = [
        f"{base_url}{encoded_book_text}{url_suffix}" for encoded_book_text in encoded_book_texts
    ]
    assert google_books_service.create_api_urls(encoded_book_texts) == expected_output

@pytest.mark.asyncio
async def test_get_book_info(google_books_service):
    # Replace with a valid API URL for testing
    api_url = "https://www.googleapis.com/books/v1/volumes?q=Patterson+How+AMERICA+LOST+ITS+MIND+OKLAHOMA"
    async with httpx.AsyncClient() as client:
        book_info = await google_books_service.get_book_info(client, api_url)
        assert isinstance(book_info, dict)

@pytest.mark.asyncio
async def test_get_book_list(google_books_service):
    # Replace with a list of valid API URLs for testing
    api_urls = [
        "https://www.googleapis.com/books/v1/volumes?q=Patterson+How+AMERICA+LOST+ITS+MIND+OKLAHOMA",
        "https://www.googleapis.com/books/v1/volumes?q=Patterson+How+AMERICA+LOST+ITS+MIND+OKLAHOMA",
        "https://www.googleapis.com/books/v1/volumes?q=Patterson+How+AMERICA+LOST+ITS+MIND+OKLAHOMA",
    ]
    book_list = await google_books_service.get_book_list(api_urls)
    assert isinstance(book_list, list)
    assert len(book_list) == len(api_urls)