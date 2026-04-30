from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentMeta:
    code: str
    name: str
    description: str
    version: str = "0.1.0"
    requires_human_review_below: float = 0.70
    consumes_from: str | None = None  # code of upstream agent in pipeline


@dataclass
class AgentResult:
    output: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    needs_human: bool = False
    notes: str = ""


class BaseAgent(ABC):
    meta: AgentMeta

    @abstractmethod
    async def run(self, payload: dict[str, Any], *, context: dict[str, Any] | None = None) -> AgentResult:
        """Process the input payload and return an AgentResult."""
        ...

    async def validate_input(self, payload: dict[str, Any]) -> None:
        """Override to enforce input contract. Raise ValueError on invalid input."""
        return None
