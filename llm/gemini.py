import logging
from google import genai
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import FAISS
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import TypedDict
import faiss
from mir.metadata_model import get_schema_descriptions

logger = logging.getLogger(__name__)


class ClientState(TypedDict):
    query: str
    context: list[Document]
    response: str


class GeminiClient:
    def __init__(
        self,
        api_key: str,
        audio_metadata_path: str,
        audio_files: list[str] = None,
        model: str = "gemini-2.0-flash",
    ):
        self._client = genai.Client(api_key=api_key)
        self.model = init_chat_model(model=model, model_provider="google_genai")
        self.audio_files = audio_files
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004"
        )
        self.vector_store = FAISS(
            embedding_function=self.embeddings,
            index=faiss.IndexFlatL2(len(self.embeddings.embed_query("hello world"))),
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        self.documents = self._store_documents(document_path=audio_metadata_path)
        if self.documents:
            self.prompt = self._create_prompt()
            self.graph = self._compile()
        else:
            logger.error("Documents not stored correctly")

    async def invoke(self, query) -> dict:
        result = await self.graph.ainvoke({"query": query})
        return result

    # @tool(response_format="content_and_artifact")
    def _retrieve(self, state: ClientState) -> dict:
        # Here if you want to change the number of retrieved docs
        retrieved_docs = self.vector_store.similarity_search(state["query"])
        return {"context": retrieved_docs}

    def _generate(self, state: ClientState) -> dict:
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = self.prompt.invoke(
            {"query": state["query"], "context": docs_content}
        )
        response = self.model.invoke(messages)
        return {"response": response.content}

    def _compile(self) -> CompiledStateGraph:
        logger.info("Building GeminiClient graph")
        graph_builder = StateGraph(ClientState).add_sequence(
            [self._retrieve, self._generate]
        )
        graph_builder.add_edge(START, "_retrieve")
        graph = graph_builder.compile()
        return graph

    def _store_documents(
        self, document_path: str | list[str]
    ) -> list[Document] | list[list[Document]]:
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
        elif isinstance(document_path, list):
            docs_list = []
            for doc_path in document_path:
                docs_list.append(loader.load(doc_path))
            self.vector_store.add_documents(documents=docs_list)
            return docs_list

    def _create_prompt(self) -> PromptTemplate:
        schema_descriptions = get_schema_descriptions()

        text_descriptions = "\n".join(
            [
                f"- {field}: {schema_descriptions.get(field, ' ')}"
                for field in schema_descriptions
            ]
        )
        prompt = f"""
            You are Mairio, an AI assistant built to answer questions about video game soundtracks.
            You will both intelligently answer questions about the soundtrack, or retrieve the soundtrack file if the user asks.
            You are currently loaded with the original Super Mario Bros (1985) soundtrack.
            Always answer in human-readable text and language. Never return in another format.
            Use the following pieces of context to answer the question at the end.
            The context is audio metadata derived and extracted from each audio file in the Super Mario Bros Soundtrack.
            The audio metadata you are loaded with have descriptions that correspond to what each value represents.
            Those descriptions are as follows:
            {text_descriptions}
            Always start the conversation with "It's-a-me, Mairio!"
            If the user asks a question about a song you are not loaded with, or asks a question you cannot reasonably answer, say that you do not know the answer.

            The context for the question is as follows:
            {{context}}

            Question: {{query}}

            Answer:
        """
        return PromptTemplate.from_template(prompt)

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

    # def create_response(self, query: str) -> str:
    #     response = self._client.models.generate_content(
    #         model=self.model,
    #         config=types.GenerateContentConfig(
    #             system_instruction=self.system_instruction
    #         ),
    #         contents=[f"{query}"],
    #     )
    #     logger.info("Completed standard text response.")
    #     return response.text

    # def create_response_with_audio(self, query: str) -> str:
    #     response = self._client.models.generate_content(
    #         model=self.model,
    #         config=types.GenerateContentConfig(
    #             system_instruction=self.system_instruction
    #         ),
    #         contents=[f"{query}", self.music_files[0]],
    #     )
    #     # myfile = self.music_files[0]
    #     # file_name = myfile.name
    #     # myfile = self.client.files.get(name=file_name)
    #     # print(myfile)
    #     logger.info("Completed standard response with audio content.")
    #     return response.text

    # def return_song(self):
    #     self.logger.info("Playing music.")
    #     playsound(sound=rf"data\music\{self.music_dict[self.music_files[0].name]}")
    #     self.logger.info("Music stopped.")
    #     return None
    #     # return self.music_files[0]
