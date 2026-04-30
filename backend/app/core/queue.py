"""Queue abstraction. InProcess now, Celery later — same interface."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Callable, Awaitable
import asyncio
import uuid
from .config import settings


class TaskQueue(ABC):
    @abstractmethod
    async def enqueue(self, fn: Callable[..., Awaitable[Any]], *args, **kwargs) -> str: ...


class InProcessQueue(TaskQueue):
    """Runs the coroutine on the current event loop. Good enough for dev / SQLite."""

    async def enqueue(self, fn, *args, **kwargs) -> str:
        task_id = str(uuid.uuid4())
        asyncio.create_task(fn(*args, **kwargs), name=f"task-{task_id}")
        return task_id


class CeleryQueue(TaskQueue):
    """Stub. Implement when QUEUE_BACKEND=celery and Redis is available."""

    async def enqueue(self, fn, *args, **kwargs) -> str:
        raise NotImplementedError("Celery backend not yet wired. Set QUEUE_BACKEND=inprocess.")


def get_queue() -> TaskQueue:
    backend = settings.QUEUE_BACKEND.lower()
    if backend == "celery":
        return CeleryQueue()
    return InProcessQueue()


queue = get_queue()
