from fastapi import FastAPI
import threading
from fastapi.responses import JSONResponse
import uvicorn
class HTTPLogger:
    def __init__(self, host="127.0.0.1", port="8080"):
        self._data = {
            "episode": [],
            "reward": []
        }
        self.host = host
        self.port = port
        self._data = {"episode": [], "reward": [], "loss": []}

        # Create FastAPI app
        self.app = FastAPI()

        @self.app.get("/metrics")
        def get_metrics():
            return JSONResponse(self._data)

        # Run HTTP server in background thread
        self._server_thread = threading.Thread(
            target=lambda: uvicorn.run(self.app, host=self.host, port=self.port, log_level="error"),
            daemon=True
        )
        self._server_thread.start()
    def log_episode_metrics(self, log_data):
        for key, value in log_data.items():
            if key in self._data:
                self._data[key].append(value)
    
