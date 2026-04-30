from agents.base import BaseAgent, AgentMeta, AgentResult
from agents.registry import register


@register
class AdherenceTesterAgent(BaseAgent):
    meta = AgentMeta(
        code="adherence_tester",
        name="Adherence Tester",
        description=(
            "Cruza material utilizado x servico executado x valor cobrado. "
            "Identifica divergencias e inconsistencias. Aprende padroes de erro/fraude."
        ),
    )

    async def run(self, payload, *, context=None):
        return AgentResult(
            output={"divergences": [], "fraud_signals": [], "stub": True},
            confidence=0.0,
            needs_human=True,
            notes="adherence_tester not yet implemented",
        )
