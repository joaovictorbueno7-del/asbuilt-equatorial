from agents.base import BaseAgent, AgentMeta, AgentResult
from agents.registry import register


@register
class AntiReprovaAgent(BaseAgent):
    meta = AgentMeta(
        code="anti_reprova",
        name="Anti-Reprova",
        description="Recebe saida do report_generator. Verifica pendencias antes do envio, "
                    "bloqueia se encontrar problemas, lista o que falta resolver.",
        consumes_from="report_generator",
    )

    async def run(self, payload, *, context=None):
        structures = payload.get("structures", [])
        blocking, warnings = [], []
        for s in structures:
            a = s.get("analysis", {})
            pm = s.get("placemark") or {}
            if not pm.get("name"):
                blocking.append({"image": s.get("image"), "issue": "placemark sem nome"})
            if a.get("structure_type") == "outro":
                warnings.append({"image": s.get("image"), "issue": "estrutura nao identificada com clareza"})
            for nc in a.get("non_conformities", []):
                warnings.append({"image": s.get("image"), "issue": nc})
            if a.get("description_filled_by_stub"):
                warnings.append({"image": s.get("image"), "issue": "descricao gerada por template"})
        ready = len(blocking) == 0 and len(warnings) <= max(2, len(structures) // 4)
        return AgentResult(
            output={
                "blocking_issues": blocking,
                "warnings": warnings,
                "ready_to_send": ready,
                "checklist": {
                    "fotos_com_coordenadas": sum(1 for s in structures if s.get("placemark")),
                    "descricoes_completas": sum(1 for s in structures
                                                if (s.get("analysis", {}).get("details") or "").strip()),
                    "total_estruturas": len(structures),
                },
                "stub": True,
            },
            confidence=0.7 if ready else 0.4,
            needs_human=not ready,
            notes=f"stub: {len(blocking)} bloqueios, {len(warnings)} avisos. ready={ready}",
        )
