import os
import httpx
import asyncio
import re

from urllib.parse import quote_plus
from dotenv import load_dotenv


class GoogleBooksService:
    def __init__(self):
        load_dotenv()

        self.GOOGLE_BOOKS_BASE_URL: str = os.getenv("GOOGLE_BOOKS_BASE_URL")
        self.GOOGLE_BOOKS_URL_SUFFIX: str = os.getenv("GOOGLE_BOOKS_URL_SUFFIX")
        self.GOOGLE_BOOKS_API_KEY: str = os.getenv("GOOGLE_BOOKS_API_KEY")

    """
    Uses regex to pattern match any character that is not an English letter (upper or lowercase),
    a digit, a whitespace character, or common punctuation.
    """

    def remove_non_english_chars(self, text):
        pattern = re.compile(r"[^a-zA-Z0-9\s.,!?;:'\"]|\-|\*")
        text = pattern.sub("", text)

        # Replace multiple spaces with a single space.
        text = re.sub(" +", " ", text)
        return text

    """
    Performs the operations from `remove_non_english_chars` on each string in the list
    and returns a list with newly processed strings according to the regex.
    """

    def process_book_texts(self, book_texts):
        processed_book_texts = []

        for text in book_texts:
            processed_book_texts.append(self.remove_non_english_chars(text))

        return processed_book_texts

    """
    URL-encodes all strings in a list, so that they are ready to be used in HTTP calls.
    Periods (.) are considered safe by default, so they are manually replaced by "%2E".
    
    Returns a list of URL-encoded strings.
    """

    def url_encode_strings(self, processed_book_texts):
        return [
            quote_plus(book, safe="").replace(".", "%2E")
            for book in processed_book_texts
        ]

    """
    Takes a list of URL-encoded strings and turns them into API-URLS that are ready to be called. 
    Prepends the URL-encoded strings with the Google Books base URL and appends the string with
    a suffix that contains printType, startIndex and maxResults filters.

    Returns a list of API-URLS ready to be called by a HTTP client.
    """

    def create_api_urls(self, encoded_book_texts):
        return [
            self.GOOGLE_BOOKS_BASE_URL
            + encoded_book_text
            + self.GOOGLE_BOOKS_URL_SUFFIX
            for encoded_book_text in encoded_book_texts
        ]

    """
    Takes the HTTPX async client and an API URL as arguments, and performs a GET request for that
    specific book.

    Returns a JSON representation of the result.
    """

    async def get_book_info(self, client, url):
        response = await client.get(url)
        book = response.json()

        return book

    """
    Takes a list of all API URLS.
    Uses asyncio to create a list of asynchronous tasks to perform 
    (retrieve info for all books in this example).

    Returns a JSON object containing info for all books.
    """

    async def get_book_list(self, encoded_urls):
        async with httpx.AsyncClient() as client:
            tasks = []

            for url in encoded_urls:
                tasks.append(asyncio.ensure_future(self.get_book_info(client, url)))

            book_list = await asyncio.gather(*tasks)

            return book_list
