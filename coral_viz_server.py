import threading
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import json
import os

# Shared State (Thread-safe enough for this simple use case)
# Structure:
# {
#   "step": 0,
#   "loss": 0.0,
#   "tier": 1,
#   "desc": "...",
#   "layers": [
#       { "id": 0, "heads": 1, "ffn_chunks": 1, "head_mask": [1,0...], "ffn_mask": [1,0...] }
#   ]
# }
GLOBAL_STATE = {
    "step": 0,
    "loss": 0.0,
    "tier": 1,
    "desc": "Initializing...",
    "layers": [],
    "events": [], # List of {step, type, desc}
    "loss_history": [],
    "hippocampus_tier_counts": {}
}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def get_viz():
    return FileResponse(os.path.join(os.path.dirname(__file__), "viz/index.html"))

@app.get("/data")
def get_data():
    return GLOBAL_STATE

def run_server():
    # Try ports 8000-8010
    port = 8000
    max_retries = 10
    for i in range(max_retries):
        try:
            port = 8000 + i
            uvicorn.run(app, host="0.0.0.0", port=port, log_level="error")
            break
        except OSError:
            continue

def start_background_server():
    # We can't easily get the port back from the thread with this simple setup,
    # so we'll just hardcode a safer default or print a generic message.
    # Better: Let's just use 8081 which is less likely to be taken by VS Code.
    pass

def run_server_safe():
    try:
        uvicorn.run(app, host="0.0.0.0", port=8081, log_level="error")
    except Exception as e:
        print(f">>> [VIZ] Error starting server: {e}")

def start_background_server():
    t = threading.Thread(target=run_server_safe, daemon=True)
    t.start()
    print(">>> [VIZ] Server started at http://localhost:8081/data")
    print(">>> [VIZ] View the dashboard at http://localhost:8081/")
