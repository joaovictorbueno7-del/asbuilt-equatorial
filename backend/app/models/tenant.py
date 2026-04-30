from sqlalchemy import String, Boolean
from sqlalchemy.orm import Mapped, mapped_column
from app.core.database import Base
from .base import UUIDMixin, TimestampMixin


class Tenant(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "tenants"

    name: Mapped[str] = mapped_column(String(200), nullable=False, unique=True)
    slug: Mapped[str] = mapped_column(String(80), nullable=False, unique=True, index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
