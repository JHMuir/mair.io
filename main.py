import os
from dotenv import load_dotenv
from src.client import GoogleClient
from src.process import AudioProcessor
from src.classify import AudioClassifier
from src.utils import Utilities

load_dotenv()

api_key = os.environ["GOOGLE_API_KEY"]


def test():
    logger = Utilities.create_logger()
    audio_files = [audio_file for audio_file in os.listdir(r"data\music")]
    processor = AudioProcessor(audio_files=audio_files, logger=logger)
    classifier = AudioClassifier(
        audio_metadata=processor.get_audio_metadata(), logger=logger
    )
    client = GoogleClient(api_key=api_key, audio_files=audio_files, logger=logger)

    processor.print_metadata()
    classifier.classify_mood()
    print(
        client.create_response(
            query="Explain to me who Mario is in a few short sentences"
        )
    )


if __name__ == "__main__":
    test()
