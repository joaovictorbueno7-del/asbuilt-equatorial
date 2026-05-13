"""Agente 03 — Report Generator.

Recebe o output do description_filler (ou diretamente do kmz_analyzer)
e gera relatório fotográfico no formato oficial Equatorial:
  - DOCX (python-docx) → estrutura idêntica ao modelo Word da concessionária
  - PDF  (reportlab)   → pronto para envio digital

O arquivo é gravado em storage/reports/ e o path é retornado no output.
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from loguru import logger

from agents.base import BaseAgent, AgentMeta, AgentResult
from agents.registry import register

_STORAGE = Path(os.environ.get("STORAGE_LOCAL_PATH", "storage"))


@register
class ReportGeneratorAgent(BaseAgent):
    meta = AgentMeta(
        code="report_generator",
        name="Report Generator",
        description=(
            "Recebe saída do description_filler / kmz_analyzer. "
            "Gera relatório fotográfico Word/PDF no formato Equatorial."
        ),
        consumes_from="description_filler",
    )

    async def run(self, payload: dict, *, context=None) -> AgentResult:
        from app.services.report_builder import generate_report

        structures = (
            payload.get("structures")
            or payload.get("placemarks")
            or []
        )
        kmz_path = payload.get("kmz_path")
        run_id = payload.get("run_id") or datetime.utcnow().strftime("%Y%m%d%H%M%S")

        # Gera DOCX
        try:
            docx_bytes, docx_name = generate_report(
                run_id=run_id,
                output_payload=payload,
                kmz_path=kmz_path,
                fmt="docx",
            )
            docx_path = str(_STORAGE / "reports" / docx_name)
            docx_ok = True
            docx_size = len(docx_bytes)
        except Exception as e:
            logger.error(f"[report_generator] DOCX falhou: {e}")
            docx_path = None
            docx_ok = False
            docx_size = 0

        # Gera PDF
        try:
            pdf_bytes, pdf_name = generate_report(
                run_id=run_id,
                output_payload=payload,
                kmz_path=kmz_path,
                fmt="pdf",
            )
            pdf_path = str(_STORAGE / "reports" / pdf_name)
            pdf_ok = True
            pdf_size = len(pdf_bytes)
        except Exception as e:
            logger.error(f"[report_generator] PDF falhou: {e}")
            pdf_path = None
            pdf_ok = False
            pdf_size = 0

        confidence = 0.9 if (docx_ok and pdf_ok) else (0.6 if (docx_ok or pdf_ok) else 0.1)
        notes = (
            f"DOCX {'OK' if docx_ok else 'ERRO'} "
            f"({docx_size:,} bytes), "
            f"PDF {'OK' if pdf_ok else 'ERRO'} "
            f"({pdf_size:,} bytes), "
            f"{len(structures)} estruturas"
        )

        return AgentResult(
            output={
                "structures": structures,
                "report_pages": max(1, len(structures) // 2 + 1),
                "report_path_docx": docx_path,
                "report_path_pdf": pdf_path,
                "docx_size_bytes": docx_size,
                "pdf_size_bytes": pdf_size,
                "generated_at": datetime.utcnow().isoformat(),
                "stub": False,
            },
            confidence=confidence,
            needs_human=not (docx_ok and pdf_ok),
            notes=notes,
        )
