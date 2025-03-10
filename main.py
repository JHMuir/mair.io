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
    logger = logging.getLogger(__name__)
    logger.info("Starting main process")
    audio_files = [audio_file for audio_file in os.listdir(r"data\music")]

    # pprint(get_schema_descriptions())

    # AudioPipeline extracts metadata from our audio files and creates a json for our client to parse
    audio_pipeline = AudioPipeline(audio_files=audio_files)
    audio_metadata_path = audio_pipeline.create_metadata_json()

    # GeminiApp initializes both our FastAPI endpoint and our GeminiClient (our llm class)
    # GeminiClient (accessed through GeminiApp.client) takes the json and sets up a vector store so we can search throughout it
    app = GeminiApp(api_key=api_key, audio_metadata_path=audio_metadata_path)

    question1 = "Return all the songs that are C Major."
    result1 = app.client.invoke(question1)
    # print(f"Context: {result1['context']}\n\n")
    print(f"Question: {question1}\nAnswer: {result1['response']}\n")

    question2 = "Which songs are played during the Underground levels of the game?"
    result2 = app.client.invoke(question2)
    # print(f"Context: {result2['context']}\n\n")
    print(f"Question: {question2}\nAnswer: {result2['response']}\n")

    question3 = "Which song is the most menacing sounding?"
    result3 = app.client.invoke(question3)
    # print(f"Context: {result3['context']}\n\n")
    print(f"Question: {question3}\nAnswer: {result3['response']}\n")

    question4 = "What are all the background themes you are loaded with?"
    result4 = app.client.invoke(question4)
    # print(f"Context: {result4['context']}\n\n")
    print(f"Question: {question4}\nAnswer: {result4['response']}\n")

    question5 = "Which song is the bassiest and why?"
    result5 = app.client.invoke(question5)
    # print(f"Context: {result4['context']}\n\n")
    print(f"Question: {question5}\nAnswer: {result5['response']}\n")

    app.run()
