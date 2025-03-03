from flask import Flask


class FlaskApp:
    def __init__(self):
        self.app = Flask(__name__)
        self.setup_routes()

    def setup_routes(self):
        self.app.route

    def hello(self):
        return "HELLO"
