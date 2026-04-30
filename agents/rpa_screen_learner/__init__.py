from agents.base import BaseAgent, AgentMeta, AgentResult
from agents.registry import register


@register
class RPAScreenLearnerAgent(BaseAgent):
    meta = AgentMeta(
        code="rpa_screen_learner",
        name="RPA Screen Learner",
        description=(
            "Automatiza preenchimento no sistema PROJ via Citrix. Aprende observando "
            "o operador (10 projetos), analisa com Claude Vision. Apos 10 exemplos entra "
            "em shadow mode. So assume autonomia quando acerto > 95%. Rollback automatico."
        ),
    )

    async def run(self, payload, *, context=None):
        # mode: "observe" | "shadow" | "autonomous"
        mode = payload.get("mode", "observe")
        return AgentResult(
            output={"mode": mode, "examples_learned": 0, "accuracy": 0.0, "stub": True},
            confidence=0.0,
            needs_human=True,
            notes="rpa_screen_learner not yet implemented; computer-use integration pending",
        )
