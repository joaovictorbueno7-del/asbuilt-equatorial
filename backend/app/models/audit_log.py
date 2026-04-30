from sqlalchemy import String, ForeignKey, JSON, Text
from sqlalchemy.orm import Mapped, mapped_column
from app.core.database import Base
from .base import UUIDMixin, TimestampMixin


class AuditLog(UUIDMixin, TimestampMixin, Base):
    """Append-only audit trail. Never updated or deleted by application code."""
    __tablename__ = "audit_logs"

    tenant_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="SET NULL"), nullable=True, index=True
    )
    user_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    action: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    resource_type: Mapped[str] = mapped_column(String(80), default="", nullable=False)
    resource_id: Mapped[str] = mapped_column(String(80), default="", nullable=False)
    ip_address: Mapped[str] = mapped_column(String(64), default="", nullable=False)
    user_agent: Mapped[str] = mapped_column(String(500), default="", nullable=False)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    success: Mapped[str] = mapped_column(String(8), default="true", nullable=False)
    notes: Mapped[str] = mapped_column(Text, default="", nullable=False)
