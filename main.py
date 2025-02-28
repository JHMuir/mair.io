import os
import logging
from dotenv import load_dotenv
from llm.client import GoogleClient
from mir.pipeline import AudioPipeline

load_dotenv()

api_key = os.environ["GOOGLE_API_KEY"]


def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


if __name__ == "__main__":
    setup_logging()
    audio_files = [audio_file for audio_file in os.listdir(r"data\music")]
    # AudioPipeline extracts metadata from our audio files and creates a json for our client to parse
    audio_pipeline = AudioPipeline(audio_files=audio_files)
    audio_metadata_path = audio_pipeline.create_metadata_json()

    # Our GoogleClient takes the json and sets up a vector store so we can search throughout it

    client = GoogleClient(
        api_key=api_key, audio_files=audio_files, document_path=audio_metadata_path
    )

    result1 = client.invoke("Return all the songs that are C Major.")
    print(f"Context: {result1['context']}\n\n")
    print(f"Answer: {result1['response']}")
    result2 = client.invoke(
        "Which songs are played during the Underground levels of the game?"
    )
    print(f"Context: {result2['context']}\n\n")
    print(f"Answer: {result2['response']}")
    result3 = client.invoke("Which song is the most menacing sounding?")
    print(f"Context: {result3['context']}\n\n")
    print(f"Answer: {result3['response']}")
