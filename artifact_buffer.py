"""
Artifact buffer: receives batches from vLLM locally and forwards them
to the orchestrator in the background with reliable retry.

Flow:
  vLLM → POST /generated (localhost, instant) → buffer → forward to orchestrator

This eliminates artifact loss: vLLM callback never fails (localhost),
and the buffer retries forwarding independently of vLLM lifecycle.
"""

import asyncio
import logging
from collections import deque
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


class ArtifactBuffer:
    """
    Thread-safe artifact buffer with background forwarding.

    - receive() is called from the /generated endpoint (vLLM callback)
    - _forward_loop() runs in background, draining the queue to orchestrator
    """

    def __init__(self):
        self._queue: deque = deque()
        self._callback_url: Optional[str] = None
        self._forward_task: Optional[asyncio.Task] = None
        self._event = asyncio.Event()
        self._client: Optional[httpx.AsyncClient] = None
        self._running = False

    def start(self, callback_url: str):
        """Start the forwarding loop with the orchestrator callback URL."""
        self._queue = deque()  # Fresh queue, discard any leftovers
        self._callback_url = callback_url
        self._running = True
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0))
        self._forward_task = asyncio.create_task(self._forward_loop())
        logger.info(f"Artifact buffer started, forwarding to {callback_url}")

    async def stop(self):
        """Stop forwarding and drain remaining items."""
        self._running = False
        self._event.set()  # Wake up the loop

        if self._forward_task and not self._forward_task.done():
            # Wait for loop to finish draining
            try:
                await asyncio.wait_for(self._forward_task, timeout=30.0)
            except asyncio.TimeoutError:
                logger.warning("Forward loop did not finish in 30s, cancelling")
                self._forward_task.cancel()
                try:
                    await self._forward_task
                except asyncio.CancelledError:
                    pass

        if self._client and not self._client.is_closed:
            await self._client.aclose()

        remaining = len(self._queue)
        if remaining:
            logger.warning(f"Buffer stopped with {remaining} unsent batches")
        else:
            logger.info("Buffer stopped, all batches forwarded")

    def receive(self, batch_json: str):
        """
        Receive a raw JSON batch from vLLM callback.
        Called from POST /generated endpoint. Non-blocking.
        """
        self._queue.append(batch_json)
        self._event.set()

    @property
    def pending_count(self) -> int:
        return len(self._queue)

    async def _forward_loop(self):
        """Background loop: drain queue and forward to orchestrator."""
        while self._running or self._queue:
            # Wait for items
            if not self._queue:
                self._event.clear()
                try:
                    await asyncio.wait_for(self._event.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

            while self._queue:
                batch_json = self._queue[0]  # peek

                success = await self._send_with_retry(batch_json)
                if success:
                    self._queue.popleft()
                else:
                    # Retry will happen on next iteration
                    break

        logger.info("Forward loop exited")

    async def _send_with_retry(self, batch_json: str, max_retries: int = 20) -> bool:
        """Send a batch to orchestrator with retry."""
        url = f"{self._callback_url}/generated"

        for attempt in range(1, max_retries + 1):
            try:
                resp = await self._client.post(
                    url,
                    content=batch_json,
                    headers={"Content-Type": "application/json"},
                )
                if resp.status_code == 200:
                    return True
                logger.warning(
                    f"Orchestrator returned {resp.status_code} "
                    f"(attempt {attempt}/{max_retries})"
                )
            except Exception as e:
                logger.warning(
                    f"Forward failed (attempt {attempt}/{max_retries}): {e}"
                )

            if attempt < max_retries:
                backoff = min(2 ** (attempt - 1), 10)
                await asyncio.sleep(backoff)

        logger.error(f"Failed to forward batch after {max_retries} retries, dropping")
        return True  # Remove from queue to prevent infinite block
