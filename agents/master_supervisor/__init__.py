from agents.base import BaseAgent, AgentMeta, AgentResult
from agents.registry import register


@register
class MasterSupervisorAgent(BaseAgent):
    meta = AgentMeta(
        code="master_supervisor",
        name="Master Supervisor",
        description=(
            "Valida saida de todos os 8 agentes. Gera score geral do processo (0-100%). "
            "Decide se pacote esta pronto para envio. Gera relatorio executivo "
            "com ROI e horas economizadas."
        ),
    )

    async def run(self, payload, *, context=None):
        runs = payload.get("runs", [])
        if not runs:
            return AgentResult(output={"score": 0, "ready": False}, confidence=0.0, needs_human=True,
                               notes="no upstream runs to evaluate")
        scores = [r.get("confidence", 0.0) for r in runs]
        overall = sum(scores) / len(scores) * 100
        return AgentResult(
            output={
                "overall_score": round(overall, 2),
                "ready_to_send": overall >= 85,
                "executive_summary": "stub",
                "stub": True,
            },
            confidence=overall / 100,
            needs_human=overall < 85,
            notes="master_supervisor stub: heuristic average only",
        )
