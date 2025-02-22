import os
from dotenv import load_dotenv
from src.client import GoogleClient
from src.pipeline import AudioPipeline
from src.utils import setup_logging

load_dotenv()

api_key = os.environ["GOOGLE_API_KEY"]

if __name__ == "__main__":
    setup_logging()
    audio_files = [audio_file for audio_file in os.listdir(r"data\music")]
    audio_pipeline = AudioPipeline(audio_files=audio_files)
    client = GoogleClient(api_key=api_key, audio_files=audio_files)

    # audio_pipeline.processor.print_metadata()
    # audio_pipeline.classifier.print_features()
    audio_pipeline.print_descriptions()
    # print(
    #     client.create_response(
    #         query="Explain to me who Mario is in a few short sentences"
    #     )
    # )
