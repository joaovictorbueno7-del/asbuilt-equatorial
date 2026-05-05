from datetime import datetime
from agents.base import BaseAgent, AgentMeta, AgentResult
from agents.registry import register


@register
class ReportGeneratorAgent(BaseAgent):
    meta = AgentMeta(
        code="report_generator",
        name="Report Generator",
        description="Recebe saida do description_filler. Gera relatorio fotografico "
                    "em Word/PDF com fotos, coordenadas e descricoes.",
        consumes_from="description_filler",
    )

    async def run(self, payload, *, context=None):
        structures = payload.get("structures", [])
        return AgentResult(
            output={
                "structures": structures,
                "report_pages": max(1, len(structures) // 2 + 1),
                "report_path_docx": None,
                "report_path_pdf": None,
                "generated_at": datetime.utcnow().isoformat(),
                "stub": True,
            },
            confidence=0.5,
            needs_human=True,
            notes=f"stub: relatorio para {len(structures)} estruturas (geracao real pendente)",
        )
