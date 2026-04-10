"""
FastAPI application for the persistent worker node.

Receives push commands from the orchestrator:
- POST /init     — start new session with block_hash + first public_key
- POST /add_key  — add public_key to priority queue
- POST /stop     — stop computation, clear all state
- GET  /health   — health check (includes vLLM status)
- GET  /status   — current worker state

Receives callbacks from co-located vLLM:
- POST /generated — artifact batch (buffered and forwarded to orchestrator)
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, Header, Request, Response

from config import get_settings, Settings
from models import (
    InitRequest,
    AddKeyRequest,
    InitResponse,
    AddKeyResponse,
    StopResponse,
    HealthResponse,
    StatusResponse,
)
from vllm_client import VLLMClient
from worker import WorkerEngine
from artifact_buffer import ArtifactBuffer

# Logging
settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(filename)-16s:%(lineno)-4d %(asctime)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Global instances
engine: WorkerEngine | None = None
buffer: ArtifactBuffer | None = None


def verify_api_key(authorization: str = Header(default="")):
    """Validate API key from Authorization header."""
    s = get_settings()
    if s.api_key and not authorization.endswith(s.api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, buffer
    s = get_settings()

    vllm = VLLMClient(s.vllm_host, s.vllm_port)
    buffer = ArtifactBuffer()
    engine = WorkerEngine(vllm, s, buffer)

    healthy = await vllm.health_check()
    if healthy:
        logger.info(f"vLLM healthy at {vllm.base_url}")
    else:
        logger.warning(f"vLLM not healthy at {vllm.base_url} — will retry on first request")

    yield

    # Shutdown: stop computation if active
    if engine._state == "computing":
        await engine.stop()
    await vllm.close()


app = FastAPI(title="PoC v2 Worker Node", lifespan=lifespan)


@app.post("/init", response_model=InitResponse)
async def init(req: InitRequest, _=Depends(verify_api_key)):
    """Start a new computation session. Aborts any active inference first."""
    result = await engine.init_session(
        block_hash=req.block_hash,
        block_height=req.block_height,
        node_id=req.node_id,
        node_count=req.node_count,
        batch_size=req.batch_size,
        target=req.target,
        callback_url=req.callback_url,
        public_key=req.public_key,
        priority=req.priority,
    )
    return InitResponse(**result)


@app.post("/add_key", response_model=AddKeyResponse)
async def add_key(req: AddKeyRequest, _=Depends(verify_api_key)):
    """Add a public key to the computation queue."""
    result = await engine.add_key(
        public_key=req.public_key,
        priority=req.priority,
    )
    return AddKeyResponse(**result)


@app.post("/stop", response_model=StopResponse)
async def stop(_=Depends(verify_api_key)):
    """Stop all computation and clear state."""
    result = await engine.stop()
    return StopResponse(**result)


@app.post("/generated")
async def generated(request: Request):
    """
    Receive artifact batch from co-located vLLM.

    vLLM sends callbacks here (localhost:9000/generated).
    The buffer stores them and forwards to orchestrator in background.
    No data loss even if orchestrator is temporarily unreachable.
    """
    body = await request.body()
    buffer.receive(body.decode())
    return {"status": "ok"}


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check including vLLM status."""
    vllm_ok = await engine.vllm.health_check()
    return HealthResponse(
        status="healthy" if vllm_ok else "unhealthy",
        vllm_healthy=vllm_ok,
    )


@app.get("/status", response_model=StatusResponse)
async def status():
    """Current worker state."""
    return StatusResponse(**engine.get_status())


@app.post("/v1/chat/completions")
async def chat_completions_proxy(request: Request):
    """
    Proxy to vLLM /v1/chat/completions.
    Only available when worker is idle (no PoC session).
    """
    if engine._state != "idle":
        raise HTTPException(status_code=503, detail="PoC computation in progress")

    body = await request.body()
    resp = await engine.vllm.client.post(
        f"{engine.vllm.base_url}/v1/chat/completions",
        content=body,
        headers={"Content-Type": "application/json"},
    )
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        media_type="application/json",
    )


if __name__ == "__main__":
    import uvicorn

    s = get_settings()
    uvicorn.run("main:app", host=s.host, port=s.port)
