from __future__ import annotations
from typing import Type, Dict
from .base import BaseAgent

AGENTS: Dict[str, Type[BaseAgent]] = {}


def register(cls: Type[BaseAgent]) -> Type[BaseAgent]:
    if not hasattr(cls, "meta") or cls.meta is None:
        raise TypeError(f"{cls.__name__} must define a class-level `meta: AgentMeta`")
    code = cls.meta.code
    if code in AGENTS:
        raise ValueError(f"Agent code '{code}' already registered")
    AGENTS[code] = cls
    return cls


def get_agent(code: str) -> BaseAgent:
    if code not in AGENTS:
        raise KeyError(f"Unknown agent: {code}")
    return AGENTS[code]()


def list_agents() -> list[dict]:
    out = []
    for code, cls in AGENTS.items():
        m = cls.meta
        out.append({
            "code": m.code,
            "name": m.name,
            "description": m.description,
            "version": m.version,
            "consumes_from": m.consumes_from,
            "requires_human_review_below": m.requires_human_review_below,
        })
    return out
