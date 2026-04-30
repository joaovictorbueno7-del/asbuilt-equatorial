from agents.base import BaseAgent, AgentMeta, AgentResult
from agents.registry import register


@register
class UTMConverterAgent(BaseAgent):
    meta = AgentMeta(
        code="utm_converter",
        name="UTM Converter",
        description=(
            "Extrai coordenadas UTM do KMZ e converte para sistema Policonico brasileiro. "
            "Exporta planilha Excel. Valida se pontos estao dentro da area de concessao."
        ),
    )

    async def run(self, payload, *, context=None):
        return AgentResult(
            output={"converted_points": [], "out_of_area": [], "excel_path": None, "stub": True},
            confidence=0.0,
            needs_human=True,
            notes="utm_converter not yet implemented",
        )
