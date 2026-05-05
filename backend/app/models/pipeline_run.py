import enum
from datetime import datetime
from sqlalchemy import String, ForeignKey, DateTime, Float, Text, JSON, Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column
from app.core.database import Base
from .base import UUIDMixin, TimestampMixin


class PipelineStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    NEEDS_HUMAN = "needs_human"


class PipelineRun(UUIDMixin, TimestampMixin, Base):
    """A pipeline groups multiple AgentRuns into one logical 'work' (e.g., a KMZ analysis)."""
    __tablename__ = "pipeline_runs"

    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    work_name: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    concessionaria: Mapped[str] = mapped_column(String(80), default="", nullable=False)
    tipo: Mapped[str] = mapped_column(String(40), default="", nullable=False)
    status: Mapped[PipelineStatus] = mapped_column(
        SAEnum(PipelineStatus, native_enum=False, length=20),
        default=PipelineStatus.PENDING, nullable=False, index=True,
    )
    input_payload: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    summary_output: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    error_message: Mapped[str] = mapped_column(Text, default="", nullable=False)
    overall_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
