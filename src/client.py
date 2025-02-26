import logging
from google import genai
from google.genai import types
from langchain_core.documents import Document
from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import FAISS
from typing_extensions import List, TypedDict
import faiss

logger = logging.getLogger(__name__)


class ClientState(TypedDict):
    query: str
    context: List[Document]
    response: str


class GoogleClient:
    def __init__(
        self,
        api_key: str,
        audio_files: List[str] = None,
        model="gemini-2.0-flash",
    ):
        logger.info("Initializing Gemini client.")
        self._client = genai.Client(api_key=api_key)
        self.model = init_chat_model(model=model, model_provider="google_genai")
        self.audio_files = audio_files
        self.system_instruction = """
            You are an AI Super Mario Bros soundtrack retrieval bot.
            You are loaded with each Super Mario Bros soundtrack file.
            You will be asked to retrieve or describe certain songs based on the user's query.
            """
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004"
        )
        self.vector_store = FAISS(
            embedding_function=self.embeddings,
            index=faiss.IndexFlatL2(len(self.embeddings.embed_query("hello world"))),
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

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

    def load_documents(
        self, document_path: str | List[str]
    ) -> List[Document] | List[List[Document]]:
        logger.info("Loading audio_metadata into client.")
        loader = JSONLoader(
            file_path=document_path, jq_schema=".[]", text_content=False
        )
        # Currently, our audio_metadata is not large enough to warrant using a text splitter (I think)
        # Our JSONLoader already splits the json into smaller documents.
        # text_splitter = RecursiveJsonSplitter(max_chunk_size=1000)
        if isinstance(document_path, str):
            docs = loader.load()
            self.vector_store.add_documents(documents=docs)
            return docs
        elif isinstance(document_path, List):
            docs_list = []
            for doc_path in document_path:
                docs_list.append(loader.load(doc_path))
            self.vector_store.add_documents(documents=docs_list)
            return docs_list

    def create_response(self, query: str) -> str:
        response = self._client.models.generate_content(
            model=self.model,
            config=types.GenerateContentConfig(
                system_instruction=self.system_instruction
            ),
            contents=[f"{query}"],
        )
        logger.info("Completed standard text response.")
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
        logger.info("Completed standard response with audio content.")
        return response.text

    # def return_song(self):
    #     self.logger.info("Playing music.")
    #     playsound(sound=rf"data\music\{self.music_dict[self.music_files[0].name]}")
    #     self.logger.info("Music stopped.")
    #     return None
    #     # return self.music_files[0]
