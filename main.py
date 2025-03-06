import os
import logging
from dotenv import load_dotenv
from app.web import GeminiApp
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

    # GeminiApp initializes both our FastAPI endpoint and our GeminiClient (our llm class)
    # GeminiClient (accessed through GeminiApp.client) takes the json and sets up a vector store so we can search throughout it
    app = GeminiApp(api_key=api_key, audio_metadata_path=audio_metadata_path)

    result1 = app.client.invoke("Return all the songs that are C Major.")
    # print(f"Context: {result1['context']}\n\n")
    print(f"Answer: {result1['response']}")

    result2 = app.client.invoke(
        "Which songs are played during the Underground levels of the game?"
    )
    # print(f"Context: {result2['context']}\n\n")
    print(f"Answer: {result2['response']}")

    result3 = app.client.invoke("Which song is the most menacing sounding?")
    # print(f"Context: {result3['context']}\n\n")
    print(f"Answer: {result3['response']}")

    result4 = app.client.invoke(
        "Return a list of every theme you are currently loaded with."
    )
    # print(f"Context: {result4['context']}\n\n")
    print(f"Answer: {result4['response']}")

    app.run()
