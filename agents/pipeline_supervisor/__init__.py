from agents.base import BaseAgent, AgentMeta, AgentResult
from agents.registry import register


@register
class PipelineSupervisorAgent(BaseAgent):
    meta = AgentMeta(
        code="pipeline_supervisor",
        name="Pipeline Supervisor",
        description=(
            "Monitora os agentes 01-04 em tempo real. Controla score de confianca "
            "de cada um, decide se caso vai para humano ou segue autonomo. "
            "Alerta quando score cai abaixo de 70%."
        ),
        requires_human_review_below=0.70,
    )

    async def run(self, payload, *, context=None):
        runs = payload.get("runs", [])
        scores = [r.get("confidence", 0.0) for r in runs] or [0.0]
        avg = sum(scores) / len(scores)
        return AgentResult(
            output={"avg_confidence": avg, "decision": "human" if avg < 0.70 else "autonomous", "stub": True},
            confidence=avg,
            needs_human=avg < 0.70,
            notes="pipeline_supervisor stub: heuristic average only",
        )
