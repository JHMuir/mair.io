import os
from dotenv import load_dotenv
from src.client import GoogleClient
from src.process import MusicProcessor

load_dotenv()

api_key = os.environ["GOOGLE_API_KEY"]


def test_client():
    client = GoogleClient(api_key=api_key)
    print(
        client.create_response(
            query="Explain to me who Mario is in a few short sentences"
        )
    )
    print(
        client.create_response_with_audio(query="Is this audio clip from Super Mario?")
    )
    print(client.return_song())


def test_processor():
    processor = MusicProcessor()
    processor.process()


# testing the client
if __name__ == "__main__":
    # test_client()
    test_processor()
