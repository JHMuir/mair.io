from google import genai


class GoogleClient:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)

    def create_response(self, query: str):
        response = self.client.models.generate_content(
            model="gemini-2.0-flash", contents=f"{query}"
        )
        return response.text
