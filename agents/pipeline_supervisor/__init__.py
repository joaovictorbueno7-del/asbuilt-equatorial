from agents.base import BaseAgent, AgentMeta, AgentResult
from agents.registry import register


@register
class PipelineSupervisorAgent(BaseAgent):
    meta = AgentMeta(
        code="pipeline_supervisor",
        name="Pipeline Supervisor",
        description="Monitora os agentes 01-04 em tempo real. Score de confianca, "
                    "decide se caso vai para humano ou autonomo. Alerta < 70%.",
        requires_human_review_below=0.70,
    )

    async def run(self, payload, *, context=None):
        upstream = payload.get("upstream_runs", [])
        scores = []
        flags = []
        for u in upstream:
            scores.append(float(u.get("confidence_score", 0.0)))
            if u.get("status") == "failed":
                flags.append(f"{u.get('agent_code')}: falhou")
            if u.get("confidence_score", 0.0) < 0.50:
                flags.append(f"{u.get('agent_code')}: confianca baixa")
        avg = sum(scores) / len(scores) if scores else 0.0
        decision = "autonomous" if avg >= 0.70 and not flags else "human"
        return AgentResult(
            output={
                "avg_confidence": round(avg, 3),
                "decision": decision,
                "flags": flags,
                "agents_evaluated": [u.get("agent_code") for u in upstream],
                "stub": True,
            },
            confidence=avg,
            needs_human=decision == "human",
            notes=f"stub: avg={avg:.2f}, flags={len(flags)}",
        )
