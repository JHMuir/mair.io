import os
from os.path import isfile, join
from tqdm import tqdm
from google import genai
from google.genai import types

from .utils import create_logger


class GoogleClient:
    def __init__(self, api_key: str, model="gemini-2.0-flash"):
        self.logger = create_logger()
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.music_files = self.parse_music_data()
        self.system_instruction = """
            You are an AI Super Mario Bros soundtrack retrieval bot.
            You are loaded with each Super Mario Bros soundtrack file.
            You will be asked to retrieve or describe certain songs based on the user's query.
            """

    def parse_music_data(self) -> list:
        if not self.client.files.list():
            self.logger.info("File upload not found, uploading files. ")
            music_files = [
                self.client.files.upload(file=join(r"data\music", file))
                for file in tqdm(os.listdir(r"data\music"))
                if isfile(join(r"data\music", file))
            ]
            self.logger.info("Files uploaded.")
        else:
            self.logger.info("File upload found, connecting cached files.")
            music_files = self.client.files.list()
            self.logger.info("Cached files connected.")
        return music_files

    def create_response(self, query: str) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            config=types.GenerateContentConfig(
                system_instruction=self.system_instruction
            ),
            contents=[f"{query}"],
        )
        self.logger.info("Completed standard text response.")
        return response.text

    def create_response_with_audio(self, query: str) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            config=types.GenerateContentConfig(
                system_instruction=self.system_instruction
            ),
            contents=[f"{query}", self.music_files[0]],
        )
        myfile = self.music_files[0]
        file_name = myfile.name
        myfile = self.client.files.get(name=file_name)
        print(myfile)
        self.logger.info("Completed standard response with audio content.")
        return response.text
