import os
from os.path import isfile, join
from tqdm import tqdm
from google import genai

from .utils import create_logger


class GoogleClient:
    def __init__(self, api_key: str):
        self.logger = create_logger()
        self.logger.propagate = False
        self.client = genai.Client(api_key=api_key)
        self.music_files = self.parse_music_data()

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
            model="gemini-2.0-flash", contents=f"{query}"
        )
        self.logger.info("Completed standard text response.")
        return response.text

    def create_response_with_audio(self, query: str) -> str:
        response = self.client.models.generate_content(
            model="gemini-2.0-flash", contents=[f"{query}", self.music_files[0]]
        )
        self.logger.info("Completed standard response with audio content.")
        return response.text
