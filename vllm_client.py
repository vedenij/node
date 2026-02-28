"""
Async HTTP client for the co-located vLLM PoC v2 endpoints.

Ported from runpod2/handler.py VLLMPoCClient (sync requests → async httpx).
"""

import logging
from typing import Dict, Optional

import httpx

from config import get_settings

logger = logging.getLogger(__name__)


class VLLMClient:
    """Async client for vLLM PoC v2 endpoints."""

    def __init__(self, host: str | None = None, port: int | None = None):
        settings = get_settings()
        h = host or settings.vllm_host
        p = port or settings.vllm_port
        self.base_url = f"http://{h}:{p}"
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0))
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def health_check(self) -> bool:
        """Check if vLLM is healthy."""
        try:
            resp = await self.client.get(f"{self.base_url}/health", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False

    async def get_status(self) -> Dict:
        """Get PoC generation status."""
        try:
            resp = await self.client.get(
                f"{self.base_url}/api/v1/pow/status",
                timeout=10.0,
            )
            if resp.status_code == 200:
                return resp.json()
            return {"status": "error", "detail": resp.text}
        except Exception as e:
            return {"status": "error", "detail": str(e)}

    async def init_generate(
        self,
        block_hash: str,
        block_height: int,
        public_key: str,
        node_id: int,
        node_count: int,
        batch_size: int = 32,
        callback_url: Optional[str] = None,
        group_id: int = 0,
        n_groups: int = 1,
    ) -> Dict:
        """
        Start continuous artifact generation.

        vLLM sends results to callback_url (orchestrator) directly.
        """
        settings = get_settings()
        payload = {
            "block_hash": block_hash,
            "block_height": block_height,
            "public_key": public_key,
            "node_id": node_id,
            "node_count": node_count,
            "batch_size": batch_size,
            "group_id": group_id,
            "n_groups": n_groups,
            "params": {
                "model": settings.model_name,
                "seq_len": settings.seq_len,
                "k_dim": settings.k_dim,
            },
        }

        if callback_url:
            payload["url"] = callback_url

        logger.info(
            f"init_generate: public_key={public_key[:16]}... "
            f"node_id={node_id} node_count={node_count} batch_size={batch_size}"
        )

        resp = await self.client.post(
            f"{self.base_url}/api/v1/pow/init/generate",
            json=payload,
        )

        if resp.status_code != 200:
            raise RuntimeError(f"init_generate failed: {resp.status_code} {resp.text}")

        return resp.json()

    async def stop(self) -> Dict:
        """Stop artifact generation."""
        try:
            resp = await self.client.post(
                f"{self.base_url}/api/v1/pow/stop",
                json={},
                timeout=30.0,
            )
            if resp.status_code != 200:
                logger.warning(f"vLLM stop failed: {resp.status_code} {resp.text}")
            return resp.json() if resp.status_code == 200 else {"status": "error"}
        except Exception as e:
            logger.warning(f"vLLM stop error: {e}")
            return {"status": "error", "detail": str(e)}
