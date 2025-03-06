import logging
from fastapi import FastAPI
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
        self.app = FastAPI(
            title="MAIR.IO API", summary="Endpoint for MAIR.IO's backend"
        )
        self.client = GeminiClient(
            api_key=api_key, audio_metadata_path=audio_metadata_path, model=model
        )
        self.setup_routes()

    def setup_routes(self):
        self.app.get("/")(self.hello)
        self.app.post("/chat")(self.chat_query)

    async def chat_query(self, request: QueryRequest):
        result = self.client.invoke(request.query)
        return {"response": result["response"], "context": result["context"]}

    async def hello(self):
        return {"message": "Hello World"}

    def run(self, host: str = "127.0.0.1", port: int = 8000):
        uvicorn.run(self.app, host=host, port=port)
