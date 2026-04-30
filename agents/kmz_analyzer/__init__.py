from agents.base import BaseAgent, AgentMeta, AgentResult
from agents.registry import register


@register
class KMZAnalyzerAgent(BaseAgent):
    meta = AgentMeta(
        code="kmz_analyzer",
        name="KMZ Analyzer",
        description=(
            "Le arquivo KMZ, extrai fotos e coordenadas, analisa cada foto "
            "com Claude Vision, identifica estruturas eletricas e valida padroes tecnicos. "
            "Saida: JSON com estruturas + nao-conformidades."
        ),
    )

    async def run(self, payload, *, context=None):
        # TODO: parse KMZ, extract images + KML, call Claude Vision per image
        return AgentResult(
            output={"structures": [], "non_conformities": [], "stub": True},
            confidence=0.0,
            needs_human=True,
            notes="kmz_analyzer not yet implemented",
        )
