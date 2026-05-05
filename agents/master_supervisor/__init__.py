from agents.base import BaseAgent, AgentMeta, AgentResult
from agents.registry import register


@register
class MasterSupervisorAgent(BaseAgent):
    meta = AgentMeta(
        code="master_supervisor",
        name="Master Supervisor",
        description="Valida saida dos 8 agentes. Score 0-100. Decide se pacote esta pronto. "
                    "Gera relatorio executivo com ROI e horas economizadas.",
    )

    async def run(self, payload, *, context=None):
        upstream = payload.get("upstream_runs", [])
        if not upstream:
            return AgentResult(
                output={"overall_score": 0, "ready_to_send": False, "executive_summary": "sem dados"},
                confidence=0.0, needs_human=True, notes="no upstream runs",
            )
        scores = [float(u.get("confidence_score", 0.0)) for u in upstream]
        completed = sum(1 for u in upstream if u.get("status") == "completed")
        failed = sum(1 for u in upstream if u.get("status") == "failed")
        overall = round(sum(scores) / len(scores) * 100, 1)
        ready = overall >= 75 and failed == 0

        per_agent = [{
            "agent": u.get("agent_code"),
            "status": u.get("status"),
            "confidence": round(u.get("confidence_score", 0.0), 2),
        } for u in upstream]

        # Stub ROI calculation
        n_estruturas = sum(
            len((u.get("output_payload") or {}).get("structures", []))
            for u in upstream if u.get("agent_code") == "kmz_analyzer"
        )
        horas_economizadas = round(n_estruturas * 0.15, 1)  # 9min/estrutura

        return AgentResult(
            output={
                "overall_score": overall,
                "ready_to_send": ready,
                "agents_completed": completed,
                "agents_failed": failed,
                "agents_total": len(upstream),
                "per_agent": per_agent,
                "executive_summary": (
                    f"{n_estruturas} estruturas processadas. "
                    f"{completed}/{len(upstream)} agentes concluidos. "
                    f"Score geral {overall}/100. "
                    f"{'PRONTO PARA ENVIO' if ready else 'NECESSITA REVISAO HUMANA'}."
                ),
                "roi": {
                    "horas_economizadas_estimado": horas_economizadas,
                    "estruturas_processadas": n_estruturas,
                    "minutos_por_estrutura_manual": 9,
                },
                "stub": True,
            },
            confidence=overall / 100,
            needs_human=not ready,
            notes=f"stub: score {overall}, ready={ready}",
        )
