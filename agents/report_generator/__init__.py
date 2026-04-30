from agents.base import BaseAgent, AgentMeta, AgentResult
from agents.registry import register


@register
class ReportGeneratorAgent(BaseAgent):
    meta = AgentMeta(
        code="report_generator",
        name="Report Generator",
        description=(
            "Recebe saida do description_filler. Gera relatorio fotografico "
            "em Word/PDF no padrao da empresa, com fotos, coordenadas e descricoes."
        ),
        consumes_from="description_filler",
    )

    async def run(self, payload, *, context=None):
        return AgentResult(
            output={"report_path": None, "format": "docx", "stub": True},
            confidence=0.0,
            needs_human=True,
            notes="report_generator not yet implemented",
        )
