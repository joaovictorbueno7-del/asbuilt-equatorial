from agents.base import BaseAgent, AgentMeta, AgentResult
from agents.registry import register


@register
class DescriptionFillerAgent(BaseAgent):
    meta = AgentMeta(
        code="description_filler",
        name="Description Filler",
        description=(
            "Recebe saida do kmz_analyzer e completa descricoes faltantes "
            "nas estruturas usando contexto tecnico do setor eletrico."
        ),
        consumes_from="kmz_analyzer",
    )

    async def run(self, payload, *, context=None):
        return AgentResult(
            output={"structures": payload.get("structures", []), "filled": 0, "stub": True},
            confidence=0.0,
            needs_human=True,
            notes="description_filler not yet implemented",
        )
