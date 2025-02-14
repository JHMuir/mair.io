from google import genai
from google.genai import types


class GoogleClient:
    def __init__(
        self, api_key: str, audio_files: list[str], logger, model="gemini-2.0-flash"
    ):
        self._logger = logger
        self._client = genai.Client(api_key=api_key)
        self.model = model
        self.audio_files = audio_files
        self.system_instruction = """
            You are an AI Super Mario Bros soundtrack retrieval bot.
            You are loaded with each Super Mario Bros soundtrack file.
            You will be asked to retrieve or describe certain songs based on the user's query.
            """

    # def parse_music_data(self) -> list:
    #     if not self.client.files.list():
    #         self.logger.info("File upload not found, uploading files. ")
    #         music_files = [
    #             self.client.files.upload(file=join(r"data\music", file))
    #             for file in tqdm(os.listdir(r"data\music"))
    #             if isfile(join(r"data\music", file))
    #         ]
    #         self.logger.info("Files uploaded.")
    #     else:
    #         self.logger.info("File upload found, connecting cached files.")
    #         music_files = self.client.files.list()
    #         self.logger.info("Cached files connected.")
    #     music_dict = {
    #         name.name: file
    #         for name, file in zip(music_files, os.listdir(r"data\music"))
    #     }
    #     # print(music_dict)
    #     return music_files, music_dict

    def create_response(self, query: str) -> str:
        response = self._client.models.generate_content(
            model=self.model,
            config=types.GenerateContentConfig(
                system_instruction=self.system_instruction
            ),
            contents=[f"{query}"],
        )
        self._logger.info("Completed standard text response.")
        return response.text

    def create_response_with_audio(self, query: str) -> str:
        response = self._client.models.generate_content(
            model=self.model,
            config=types.GenerateContentConfig(
                system_instruction=self.system_instruction
            ),
            contents=[f"{query}", self.music_files[0]],
        )
        # myfile = self.music_files[0]
        # file_name = myfile.name
        # myfile = self.client.files.get(name=file_name)
        # print(myfile)
        self._logger.info("Completed standard response with audio content.")
        return response.text

    # def return_song(self):
    #     self.logger.info("Playing music.")
    #     playsound(sound=rf"data\music\{self.music_dict[self.music_files[0].name]}")
    #     self.logger.info("Music stopped.")
    #     return None
    #     # return self.music_files[0]
