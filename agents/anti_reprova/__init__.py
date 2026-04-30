from agents.base import BaseAgent, AgentMeta, AgentResult
from agents.registry import register


@register
class AntiReprovaAgent(BaseAgent):
    meta = AgentMeta(
        code="anti_reprova",
        name="Anti-Reprova",
        description=(
            "Recebe saida do report_generator. Verifica todas as pendencias antes "
            "do envio, bloqueia se encontrar problemas, lista o que falta resolver."
        ),
        consumes_from="report_generator",
    )

    async def run(self, payload, *, context=None):
        return AgentResult(
            output={"blocking_issues": [], "warnings": [], "ready_to_send": False, "stub": True},
            confidence=0.0,
            needs_human=True,
            notes="anti_reprova not yet implemented",
        )
