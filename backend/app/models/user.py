import enum
from datetime import datetime
from sqlalchemy import String, Boolean, ForeignKey, DateTime, Integer, Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.core.database import Base
from .base import UUIDMixin, TimestampMixin


class UserRole(str, enum.Enum):
    ADMIN = "admin"
    SUPERVISOR = "supervisor"
    OPERADOR = "operador"
    AUDITOR = "auditor"


class User(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "users"

    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    email: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str] = mapped_column(String(200), nullable=False, default="")
    role: Mapped[UserRole] = mapped_column(
        SAEnum(UserRole, native_enum=False, length=20),
        default=UserRole.OPERADOR,
        nullable=False,
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    mfa_enabled: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    mfa_secret_encrypted: Mapped[str | None] = mapped_column(String(500), nullable=True)
    mfa_recovery_codes_encrypted: Mapped[str | None] = mapped_column(String(2000), nullable=True)

    failed_login_attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    locked_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    tenant = relationship("Tenant", lazy="joined")
