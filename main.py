import os
from dotenv import load_dotenv
from src.client import GoogleClient
from src.utils import setup_logging

load_dotenv()

api_key = os.environ["GOOGLE_API_KEY"]

if __name__ == "__main__":
    setup_logging()
    audio_files = [audio_file for audio_file in os.listdir(r"data\music")]
    # audio_pipeline = AudioPipeline(audio_files=audio_files)
    client = GoogleClient(
        api_key=api_key, audio_files=audio_files, document_path="audio_metadata.json"
    )
    client.load_documents()
