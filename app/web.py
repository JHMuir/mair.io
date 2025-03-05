from fastapi import FastAPI


class GeminiAPI:
    def __init__(self):
        self.app = FastAPI(
            title="MAIR.IO App", summary="Endpoint for MAIR.IO's backend"
        )
        self.setup_routes()

    def setup_routes(self):
        self.app.route

    def hello(self):
        return "HELLO"
