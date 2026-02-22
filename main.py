from fastapi import FastAPI
from datetime import datetime, timezone

app = FastAPI()

@app.get("/health")
def health():
    return {"ok": True, "time": datetime.now(timezone.utc).isoformat()}
