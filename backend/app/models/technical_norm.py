"""Permanent technical norms knowledge base.
HARD CONSTRAINT: rows are NEVER deleted. Use `ativa=False` to retire.
Versioning: when a norm is superseded, set data_vigencia_fim and replaced_by_id."""
from datetime import datetime, date
from sqlalchemy import String, Boolean, ForeignKey, DateTime, Date, Text, JSON, Integer, Float
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.core.database import Base
from .base import UUIDMixin, TimestampMixin


class TechnicalNorm(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "technical_norms"

    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    concessionaria: Mapped[str] = mapped_column(String(80), nullable=False, index=True)
    codigo: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    nome: Mapped[str] = mapped_column(String(500), nullable=False)
    versao: Mapped[str] = mapped_column(String(40), default="1.0", nullable=False)

    pdf_filename: Mapped[str] = mapped_column(String(500), nullable=False)
    pdf_path: Mapped[str] = mapped_column(String(1000), nullable=False)
    pdf_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    pdf_size_bytes: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    text_extracted: Mapped[str] = mapped_column(Text, default="", nullable=False)
    page_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    data_vigencia_inicio: Mapped[date | None] = mapped_column(Date, nullable=True)
    data_vigencia_fim: Mapped[date | None] = mapped_column(Date, nullable=True)
    ativa: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False, index=True)

    replaced_by_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("technical_norms.id", ondelete="SET NULL"), nullable=True
    )

    criado_por_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    desativado_por_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    desativado_em: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Deep reprocessing status (vision-based pipeline)
    processing_status: Mapped[str] = mapped_column(String(20), default="idle", nullable=False)
    # idle | classifying | extracting | done | failed
    processing_progress: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    processing_message: Mapped[str] = mapped_column(String(500), default="", nullable=False)
    processing_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    processing_finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    pages_total: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    pages_with_drawings: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    pages_processed: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    structures = relationship("NormStructure", back_populates="norm",
                               cascade="save-update, merge", passive_deletes=False)
    materials = relationship("NormMaterial", back_populates="norm",
                              cascade="save-update, merge", passive_deletes=False)


class NormStructure(UUIDMixin, TimestampMixin, Base):
    """Single electrical structure parsed out of a norm PDF.
    Used for matching against KMZ analysis results and as PROJ field source."""
    __tablename__ = "norm_structures"

    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    norm_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("technical_norms.id"), nullable=False, index=True
    )

    codigo_estrutura: Mapped[str] = mapped_column(String(80), nullable=False, index=True)
    nome_completo: Mapped[str] = mapped_column(String(500), nullable=False)
    descricao_tecnica: Mapped[str] = mapped_column(Text, default="", nullable=False)
    caracteristicas_visuais: Mapped[str] = mapped_column(Text, default="", nullable=False)
    campos_proj: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    materiais: Mapped[list] = mapped_column(JSON, default=list, nullable=False)

    # Deep extraction extras
    tipo_rede: Mapped[str] = mapped_column(String(20), default="", nullable=False)  # AT/MT/BT
    tensao_nominal: Mapped[str] = mapped_column(String(40), default="", nullable=False)  # 13,8kV etc
    como_identificar_na_foto: Mapped[str] = mapped_column(Text, default="", nullable=False)
    restricoes_uso: Mapped[str] = mapped_column(Text, default="", nullable=False)
    desenho_numero: Mapped[str] = mapped_column(String(40), default="", nullable=False, index=True)
    pagina_referencia: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    imagem_desenho_path: Mapped[str] = mapped_column(String(1000), default="", nullable=False)
    fixacao: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)  # tabelas de fixacao
    source_text_excerpt: Mapped[str] = mapped_column(Text, default="", nullable=False)
    extraction_confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    requires_review: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    parent_structure_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("norm_structures.id", ondelete="SET NULL"), nullable=True
    )

    norm = relationship("TechnicalNorm", back_populates="structures")


class NormMaterial(UUIDMixin, TimestampMixin, Base):
    """Materials catalog parsed across norms. A material may appear in many structures."""
    __tablename__ = "norm_materials"

    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    norm_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("technical_norms.id"), nullable=False, index=True
    )

    codigo_material: Mapped[str] = mapped_column(String(80), nullable=False, index=True)
    codigo_item: Mapped[str] = mapped_column(String(40), default="", nullable=False)  # M-01, I-05
    descricao: Mapped[str] = mapped_column(Text, default="", nullable=False)
    tensao: Mapped[str] = mapped_column(String(40), default="", nullable=False)
    used_in_structures: Mapped[list] = mapped_column(JSON, default=list, nullable=False)

    norm = relationship("TechnicalNorm", back_populates="materials")
