from agents.base import BaseAgent, AgentMeta, AgentResult
from agents.registry import register

DEFAULT_DESC = {
    "poste": "Poste de concreto duplo T, padrao distribuicao aerea.",
    "transformador": "Transformador de distribuicao a oleo, montagem em poste.",
    "chave": "Chave seccionadora unipolar, operacao manual com vara.",
    "cruzeta": "Cruzeta de madeira tratada, fixacao a poste por parafuso passante.",
    "isolador": "Isolador de porcelana tipo pino, classe 15kV.",
    "cabo": "Cabo de aluminio nu CAA, secao conforme projeto.",
    "para_raios": "Para-raios de oxido de zinco (ZnO), classe de tensao 12kV.",
    "medidor": "Medidor eletronico monofasico, padrao concessionaria.",
    "outro": "Estrutura nao identificada automaticamente.",
}


@register
class DescriptionFillerAgent(BaseAgent):
    meta = AgentMeta(
        code="description_filler",
        name="Description Filler",
        description="Recebe saida do kmz_analyzer e completa descricoes faltantes "
                    "com contexto tecnico do setor eletrico.",
        consumes_from="kmz_analyzer",
    )

    async def run(self, payload, *, context=None):
        structures = payload.get("structures", [])
        filled = 0
        for s in structures:
            a = s.get("analysis", {})
            details = (a.get("details") or "").strip()
            if not details or len(details) < 30:
                stype = a.get("structure_type", "outro")
                a["details"] = DEFAULT_DESC.get(stype, DEFAULT_DESC["outro"])
                a["description_filled_by_stub"] = True
                filled += 1
        return AgentResult(
            output={
                "structures": structures,
                "filled_count": filled,
                "total": len(structures),
                "stub": True,
            },
            confidence=0.6,
            needs_human=False,
            notes=f"stub: preencheu {filled}/{len(structures)} descricoes com base de templates",
        )
