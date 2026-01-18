# start fast_api cli
# uvicorn open_server.fast_api:app --reload --host 0.0.0.0 --port 8000

from fastapi import FastAPI

from core.engine import Engine
from schemas import http


app = FastAPI()
engine_instance = Engine()


@app.post("/v1/embeddings")
async def v1_embeddings(request: http.EmbeddingRequest):
    return await engine_instance.v1_embeddings(request)