"""
Pydantic models for the worker node API.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


# =============================================================================
# Requests (from orchestrator)
# =============================================================================


class InitRequest(BaseModel):
    """POST /init - Start a new session."""

    block_hash: str = Field(..., description="Block hash for model generation")
    block_height: int = Field(..., description="Block height")
    node_id: int = Field(..., description="This worker's node_id for nonce interleaving")
    node_count: int = Field(..., description="Total node count for nonce interleaving")
    batch_size: int = Field(..., description="Nonces per vLLM batch")
    target: int = Field(..., description="Nonces to compute per key before auto-switching")
    callback_url: str = Field(..., description="Orchestrator URL for result callbacks")
    public_key: str = Field(..., description="First public key to compute")
    priority: int = Field(..., description="Priority of the first key (lower = higher)")


class AddKeyRequest(BaseModel):
    """POST /add_key - Add a key to the priority queue."""

    public_key: str = Field(..., description="Public key to add")
    priority: int = Field(..., description="Priority (lower = higher)")


# =============================================================================
# Responses
# =============================================================================


class InitResponse(BaseModel):
    """Response to POST /init."""

    status: str = Field(..., description="ok or error")
    message: str = Field(default="", description="Additional info")


class AddKeyResponse(BaseModel):
    """Response to POST /add_key."""

    status: str = Field(..., description="ok or error")
    queue_size: int = Field(default=0, description="Queue depth after adding")
    message: str = Field(default="", description="Additional info")


class StopResponse(BaseModel):
    """Response to POST /stop."""

    status: str = Field(default="ok")
    was_computing: bool = Field(default=False, description="Was actively computing when stopped")


class HealthResponse(BaseModel):
    """Response to GET /health."""

    status: str = Field(..., description="healthy or unhealthy")
    vllm_healthy: bool = Field(default=False)


class StatusResponse(BaseModel):
    """Response to GET /status."""

    state: str = Field(..., description="idle or computing")
    current_key: Optional[str] = Field(None, description="Current public key being computed")
    current_key_nonces: int = Field(default=0, description="Nonces computed for current key")
    current_key_target: int = Field(default=0, description="Target nonces per key")
    queue_size: int = Field(default=0, description="Keys waiting in queue")
    queue_keys: List[str] = Field(default_factory=list, description="Public keys in queue (truncated)")
    session_block_hash: Optional[str] = Field(None, description="Current block hash")
    uptime_seconds: int = Field(default=0, description="Worker uptime")
