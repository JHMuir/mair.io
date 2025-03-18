import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from llm.mairio_gemini import GeminiClient

logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    query: str


class GeminiApp:
    def __init__(
        self, api_key: str, audio_metadata_path: str, model: str = "gemini-2.0-flash"
    ):
        logger.info("Initializing GeminiClient.")
        self.client = GeminiClient(
            api_key=api_key, audio_metadata_path=audio_metadata_path, model=model
        )
        logger.info("Initializing MAIR.IO API/App")
        self.app = FastAPI(
            title="MAIR.IO API", summary="Endpoint for MAIR.IO's backend"
        )
        self._configure_cors()
        self._setup_routes()

    def _configure_cors(self):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Allows all origins
            allow_credentials=True,
            allow_methods=["*"],  # Allows all methods
            allow_headers=["*"],  # Allows all headers
        )

    def _setup_routes(self) -> None:
        # self.app.get("/")(self.hello)
        self.app.post("/chat")(self.chat_query)

    async def chat_query(self, request: QueryRequest) -> dict:
        result = await self.client.invoke(request.query)
        return {"response": result["response"], "context": result["context"]}

    async def hello(self) -> dict:
        return {"message": "Hello World"}

    def run(self, host: str = "127.0.0.1", port: int = 8000) -> None:
        uvicorn.run(self.app, host=host, port=port)
