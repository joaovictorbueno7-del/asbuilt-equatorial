from sqlalchemy import String, ForeignKey, Float, Boolean, JSON, Text, Integer
from sqlalchemy.orm import Mapped, mapped_column
from app.core.database import Base
from .base import UUIDMixin, TimestampMixin


class LearningCase(UUIDMixin, TimestampMixin, Base):
    """One learned example per agent. Used for shadow mode training, feedback loops,
    and confidence calibration. Each agent owns its own pool of cases."""
    __tablename__ = "learning_cases"

    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    agent_code: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    source_run_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("agent_runs.id", ondelete="SET NULL"), nullable=True
    )

    input_payload: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    expected_output: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    observed_output: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)

    is_correct: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    feedback_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    human_notes: Mapped[str] = mapped_column(Text, default="", nullable=False)

    times_used: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
