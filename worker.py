"""
Worker engine: state machine with priority queue for PoC v2 computation.

Manages the lifecycle of continuous nonce generation:
- Receives session init with block_hash and first public_key
- Maintains a priority queue of public_keys
- Auto-switches to next key after reaching target nonces
- Accepts new keys mid-computation without interruption
- Cleans up on stop command
"""

import asyncio
import heapq
import logging
import time
from typing import Optional

from vllm_client import VLLMClient
from config import Settings

logger = logging.getLogger(__name__)


class WorkerEngine:
    """
    Core worker state machine.

    States:
        idle      - No active session, waiting for /init
        computing - Actively generating nonces, processing priority queue
    """

    def __init__(self, vllm: VLLMClient, settings: Settings):
        self.vllm = vllm
        self.settings = settings
        self._start_time = time.time()

        # Session params (set by /init, cleared by /stop)
        self.block_hash: Optional[str] = None
        self.block_height: Optional[int] = None
        self.node_id: Optional[int] = None
        self.node_count: Optional[int] = None
        self.batch_size: Optional[int] = None
        self.target: Optional[int] = None
        self.callback_url: Optional[str] = None

        # Priority queue: heapq of (priority, insertion_order, public_key)
        self._queue: list = []
        self._insertion_counter: int = 0

        # Current computation
        self._current_key: Optional[str] = None
        self._current_nonces: int = 0
        self._state: str = "idle"

        # Background task
        self._compute_task: Optional[asyncio.Task] = None

        # Lock for state mutations
        self._lock = asyncio.Lock()

    def _add_to_queue(self, public_key: str, priority: int):
        """Add a key to the priority queue. NOT thread-safe, caller must hold lock."""
        heapq.heappush(self._queue, (priority, self._insertion_counter, public_key))
        self._insertion_counter += 1

    def _pop_next_key(self) -> Optional[str]:
        """Pop the highest-priority key from queue. Returns None if empty."""
        if not self._queue:
            return None
        _, _, public_key = heapq.heappop(self._queue)
        return public_key

    async def init_session(self, block_hash: str, block_height: int,
                           node_id: int, node_count: int, batch_size: int,
                           target: int, callback_url: str,
                           public_key: str, priority: int) -> dict:
        """
        Handle POST /init. Start a new session.

        If already computing, stops current work first.
        """
        # Cancel compute task WITHOUT holding lock (avoids deadlock)
        await self._cancel_compute_task()

        async with self._lock:
            # Stop vLLM and clear state
            await self._cleanup_state()

            # Set session params
            self.block_hash = block_hash
            self.block_height = block_height
            self.node_id = node_id
            self.node_count = node_count
            self.batch_size = batch_size
            self.target = target
            self.callback_url = callback_url

            # Clear queue and add first key
            self._queue = []
            self._insertion_counter = 0
            self._add_to_queue(public_key, priority)

            # Start computation loop
            self._state = "computing"
            self._compute_task = asyncio.create_task(self._compute_loop())

            logger.info(
                f"Session started: block_hash={block_hash[:16]}... "
                f"first_key={public_key[:16]}... target={target}"
            )

        return {"status": "ok"}

    async def add_key(self, public_key: str, priority: int) -> dict:
        """
        Handle POST /add_key. Add key to priority queue.

        Does not interrupt current computation.
        """
        async with self._lock:
            if self.block_hash is None:
                return {
                    "status": "error",
                    "queue_size": 0,
                    "message": "No active session. Send /init first.",
                }

            self._add_to_queue(public_key, priority)
            queue_size = len(self._queue) + (1 if self._current_key else 0)

            logger.info(
                f"Key added: {public_key[:16]}... priority={priority} "
                f"queue_size={queue_size}"
            )

        return {"status": "ok", "queue_size": queue_size}

    async def stop(self) -> dict:
        """
        Handle POST /stop. Stop everything, clear all state.
        """
        was_computing = self._state == "computing"

        # Cancel compute task WITHOUT holding lock (avoids deadlock)
        await self._cancel_compute_task()

        async with self._lock:
            await self._cleanup_state()
            logger.info(f"Stopped. was_computing={was_computing}")

        return {"status": "ok", "was_computing": was_computing}

    async def _cancel_compute_task(self):
        """Cancel the background compute task. Must be called WITHOUT lock."""
        if self._compute_task and not self._compute_task.done():
            self._state = "idle"  # Signal loop to exit
            self._compute_task.cancel()
            try:
                await self._compute_task
            except asyncio.CancelledError:
                pass
            self._compute_task = None

    async def _cleanup_state(self):
        """Stop vLLM and clear all state. Caller must hold lock."""
        # Stop vLLM generation
        try:
            await self.vllm.stop()
        except Exception as e:
            logger.warning(f"vLLM stop error during cleanup: {e}")

        # Clear all state
        self._queue = []
        self._insertion_counter = 0
        self._current_key = None
        self._current_nonces = 0
        self._state = "idle"
        self.block_hash = None
        self.block_height = None
        self.node_id = None
        self.node_count = None
        self.batch_size = None
        self.target = None
        self.callback_url = None

    def get_status(self) -> dict:
        """Get current worker status for GET /status."""
        queue_keys = [pk[:16] + "..." for _, _, pk in sorted(self._queue)]
        return {
            "state": self._state,
            "current_key": self._current_key[:16] + "..." if self._current_key else None,
            "current_key_nonces": self._current_nonces,
            "current_key_target": self.target or 0,
            "queue_size": len(self._queue),
            "queue_keys": queue_keys,
            "session_block_hash": self.block_hash[:16] + "..." if self.block_hash else None,
            "uptime_seconds": int(time.time() - self._start_time),
        }

    async def _compute_loop(self):
        """
        Background task: process keys from priority queue.

        Runs until queue is empty or stopped.
        """
        try:
            while self._state == "computing":
                # Pop next key (under lock to avoid race with add_key)
                async with self._lock:
                    key = self._pop_next_key()

                if key is None:
                    # Queue empty — go idle
                    self._state = "idle"
                    self._current_key = None
                    self._current_nonces = 0
                    logger.info("Queue empty, returning to idle")
                    return

                self._current_key = key
                self._current_nonces = 0

                logger.info(f"Starting generation for {key[:16]}... target={self.target}")

                # Tell vLLM to start generating for this key
                try:
                    await self.vllm.init_generate(
                        block_hash=self.block_hash,
                        block_height=self.block_height,
                        public_key=key,
                        node_id=self.node_id,
                        node_count=self.node_count,
                        batch_size=self.batch_size,
                        callback_url=self.callback_url,
                    )
                except Exception as e:
                    logger.error(f"vLLM init_generate failed for {key[:16]}...: {e}")
                    # Skip this key, try next
                    continue

                # Poll vLLM status until target reached
                retry_count = 0
                while self._state == "computing":
                    await asyncio.sleep(2)

                    try:
                        status = await self.vllm.get_status()
                        retry_count = 0  # Reset on success
                    except Exception as e:
                        retry_count += 1
                        if retry_count >= 5:
                            logger.error(f"vLLM status unreachable after 5 retries: {e}")
                            break
                        logger.warning(f"vLLM status error (retry {retry_count}/5): {e}")
                        continue

                    total_valid = status.get("total_valid", 0)
                    self._current_nonces = total_valid

                    if total_valid >= self.target:
                        logger.info(
                            f"Target reached for {key[:16]}...: "
                            f"{total_valid}/{self.target}"
                        )
                        # Stop current generation before switching
                        await self.vllm.stop()
                        break

                    # Check if vLLM stopped unexpectedly
                    vllm_state = status.get("status", "")
                    if vllm_state in ("IDLE", "STOPPED", "ERROR"):
                        logger.warning(
                            f"vLLM stopped unexpectedly ({vllm_state}) "
                            f"for {key[:16]}... at {total_valid} nonces"
                        )
                        break

                # Loop back to pop next key

        except asyncio.CancelledError:
            logger.info("Compute loop cancelled")
        except Exception as e:
            logger.error(f"Compute loop error: {e}", exc_info=True)
            self._state = "idle"
            self._current_key = None
