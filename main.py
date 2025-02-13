import os
from dotenv import load_dotenv
from src.client import GoogleClient
from src.process import AudioProcessor
from src.classify import AudioClassifier

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
    processor = AudioProcessor()
    classifier = AudioClassifier(audio_metadata=processor.audio_metadata)
    classifier.do_a_thing()
    processor.print_features()


# testing the client
if __name__ == "__main__":
    # test_client()
    test_processor()
