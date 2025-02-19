import os
from dotenv import load_dotenv
from src.client import GoogleClient
from src.process import AudioProcessor
from src.classify import AudioClassifier
from src.utils import setup_logging

load_dotenv()

api_key = os.environ["GOOGLE_API_KEY"]

if __name__ == "__main__":
    setup_logging()
    audio_files = [audio_file for audio_file in os.listdir(r"data\music")]
    processor = AudioProcessor(audio_files=audio_files)
    classifier = AudioClassifier(audio_metadata=processor.get_audio_metadata().copy())
    client = GoogleClient(api_key=api_key, audio_files=audio_files)

    processor.print_metadata()
    classifier.print_features()
    # print(
    #     client.create_response(
    #         query="Explain to me who Mario is in a few short sentences"
    #     )
    # )
