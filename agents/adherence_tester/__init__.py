from agents.base import BaseAgent, AgentMeta, AgentResult
from agents.registry import register


@register
class AdherenceTesterAgent(BaseAgent):
    meta = AgentMeta(
        code="adherence_tester",
        name="Adherence Tester",
        description="Cruza material x servico executado x valor cobrado. "
                    "Identifica divergencias e padroes de fraude.",
    )

    async def run(self, payload, *, context=None):
        structures = payload.get("structures", [])
        from collections import Counter
        types = Counter(s.get("analysis", {}).get("structure_type", "outro") for s in structures)
        divergences = []
        if types.get("transformador", 0) > 0 and types.get("para_raios", 0) == 0:
            divergences.append({
                "type": "missing_companion",
                "issue": "Transformador identificado sem para-raios proximos (verificar)",
            })
        if types.get("poste", 0) > 0 and types.get("cruzeta", 0) == 0:
            divergences.append({
                "type": "missing_companion",
                "issue": "Postes sem cruzeta visivel — checar se cruzeta esta presente",
            })
        return AgentResult(
            output={
                "estrutura_counts": dict(types),
                "divergences": divergences,
                "fraud_signals": [],
                "rules_base_loaded": False,
                "stub": True,
            },
            confidence=0.5,
            needs_human=len(divergences) > 0,
            notes=f"stub: heuristica simples. {len(divergences)} divergencias. base UP x Servico nao carregada.",
        )
