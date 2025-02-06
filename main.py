import os
from dotenv import load_dotenv
from src.client import GoogleClient

load_dotenv()

api_key = os.environ["GOOGLE_API_KEY"]

if __name__ == "__main__":
    client = GoogleClient(api_key=api_key)
    print(
        client.create_response(
            query="Explain to me who Mario is in a few short sentences"
        )
    )
    print(client.create_response_with_audio(query="Describe this audio clip"))
