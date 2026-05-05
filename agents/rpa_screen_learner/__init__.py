from agents.base import BaseAgent, AgentMeta, AgentResult
from agents.registry import register


@register
class RPAScreenLearnerAgent(BaseAgent):
    meta = AgentMeta(
        code="rpa_screen_learner",
        name="RPA Screen Learner",
        description="Automatiza preenchimento no PROJ via Citrix. Aprende observando 10 projetos. "
                    "Shadow mode > 95% antes de autonomia. Rollback automatico.",
        consumes_from="report_generator",
    )

    async def run(self, payload, *, context=None):
        structures = payload.get("structures", [])
        return AgentResult(
            output={
                "mode": "observe",
                "examples_seen": 0,
                "examples_required_before_shadow": 10,
                "accuracy_threshold_for_autonomy": 0.95,
                "current_accuracy": 0.0,
                "would_fill_in_proj": {
                    "n_estruturas": len(structures),
                    "fields_pendentes": ["UP", "tipo_servico", "valor", "data_execucao"],
                },
                "stub": True,
            },
            confidence=0.0,
            needs_human=True,
            notes="stub: RPA Citrix requer integracao real com PROJ + computer-use",
        )
