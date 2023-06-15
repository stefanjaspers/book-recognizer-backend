from locust import HttpUser, task, between
from random import randint, choice


class UserBehavior(HttpUser):
    wait_time = between(1, 2.5)

    @task
    def on_start(self):
        """
        Actions to be performed when a User starts.
        """
        user_credentials = {"username": "stefanjaspers", "password": "guitarhero99"}
        response = self.client.post("/auth/token", data=user_credentials)

        # Store token if login is successful
        if response.status_code == 200:
            self.token = response.json().get("access_token")

    @task
    def index_page(self):
        """
        A task that redirects the user to the index page.
        """
        self.client.get("/")

    @task
    def register_user(self):
        """
        A task to register a new user.
        """
        new_user = {
            "first_name": f"User{randint(1, 100000)}",
            "last_name": f"User{randint(1, 100000)}",
            "username": f"User{randint(1, 100000)}",
            "password": f"Password{randint(1, 100000)}",
            "book_preferences": [],
        }
        self.client.post("/auth/", json=new_user)

    @task
    def add_preferences(self):
        """
        A task to add book genre preferences to a user.
        """
        if hasattr(self, "token"):
            headers = {"Authorization": f"Bearer {self.token}"}

            preferences = {"preferences": ["Fantasy", "Adventure"]}

            response = self.client.post(
                "/user/book_preferences", json=preferences, headers=headers
            )

            # check if the response status is not 200, then print the response content for debug
            if response.status_code != 200:
                print(f"Update preferences failed: {response.content}")

    @task
    def update_preferences(self):
        """
        A task to update the user's book genre preferences.
        """
        if hasattr(self, "token"):
            headers = {"Authorization": f"Bearer {self.token}"}

            preferences = {"preferences": ["Fantasy", "Adventure"]}

            response = self.client.put(
                "/user/book_preferences", json=preferences, headers=headers
            )

            # check if the response status is not 200, then print the response content for debug
            if response.status_code != 200:
                print(f"Update preferences failed: {response.content}")
