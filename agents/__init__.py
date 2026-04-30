from .base import BaseAgent, AgentResult, AgentMeta
from .registry import register, get_agent, list_agents, AGENTS

# import each module so the @register decorator runs
from . import (  # noqa: F401
    kmz_analyzer,
    description_filler,
    report_generator,
    anti_reprova,
    pipeline_supervisor,
    utm_converter,
    rpa_screen_learner,
    adherence_tester,
    master_supervisor,
)

__all__ = ["BaseAgent", "AgentResult", "AgentMeta", "register", "get_agent", "list_agents", "AGENTS"]
