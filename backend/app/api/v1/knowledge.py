"""Knowledge base endpoints for technical norms.
HARD CONSTRAINT: there is NO DELETE method. Use PATCH /deactivate to retire a norm."""
from __future__ import annotations
from datetime import datetime, timezone, date
from pathlib import Path
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, BackgroundTasks
from pydantic import BaseModel
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db, session_scope
from app.models import User, TechnicalNorm, NormStructure, NormMaterial
from app.auth.dependencies import get_current_user, get_client_ip
from app.services.audit import write_audit
from app.services.norms_extractor import md5_of_bytes, extract_text, claude_structure
from app.services.norms_deep_extractor import deep_process_pdf
from loguru import logger

router = APIRouter()

KNOWLEDGE_DIR = Path("knowledge/normas")  # relative to backend cwd
PAGES_DIR_BASE = Path("knowledge/pages")
MAX_PDF_BYTES = 100 * 1024 * 1024


class StructureOut(BaseModel):
    id: str
    codigo_estrutura: str
    nome_completo: str
    descricao_tecnica: str
    caracteristicas_visuais: str
    campos_proj: dict
    materiais: list


class NormSummary(BaseModel):
    id: str
    concessionaria: str
    codigo: str
    nome: str
    versao: str
    pdf_filename: str
    pdf_size_bytes: int
    page_count: int
    structure_count: int
    ativa: bool
    data_vigencia_inicio: date | None
    data_vigencia_fim: date | None
    replaced_by_id: str | None
    created_at: datetime


class NormDetail(NormSummary):
    text_extracted: str
    pdf_hash: str
    structures: list[StructureOut]


@router.get("", response_model=list[NormSummary])
async def list_norms(concessionaria: str | None = None, ativa: bool | None = None,
                     db: AsyncSession = Depends(get_db),
                     user: User = Depends(get_current_user)):
    q = select(TechnicalNorm).where(TechnicalNorm.tenant_id == user.tenant_id)
    if concessionaria:
        q = q.where(TechnicalNorm.concessionaria == concessionaria)
    if ativa is not None:
        q = q.where(TechnicalNorm.ativa == ativa)
    q = q.order_by(desc(TechnicalNorm.created_at))
    rows = (await db.execute(q)).scalars().all()
    out: list[NormSummary] = []
    for n in rows:
        cnt = (await db.execute(
            select(NormStructure).where(NormStructure.norm_id == n.id)
        )).scalars().all()
        out.append(NormSummary(
            id=n.id, concessionaria=n.concessionaria, codigo=n.codigo, nome=n.nome,
            versao=n.versao, pdf_filename=n.pdf_filename, pdf_size_bytes=n.pdf_size_bytes,
            page_count=n.page_count, structure_count=len(cnt), ativa=n.ativa,
            data_vigencia_inicio=n.data_vigencia_inicio, data_vigencia_fim=n.data_vigencia_fim,
            replaced_by_id=n.replaced_by_id, created_at=n.created_at,
        ))
    return out


@router.get("/{norm_id}", response_model=NormDetail)
async def get_norm(norm_id: str, db: AsyncSession = Depends(get_db),
                   user: User = Depends(get_current_user)):
    n = (await db.execute(
        select(TechnicalNorm).where(TechnicalNorm.id == norm_id,
                                     TechnicalNorm.tenant_id == user.tenant_id)
    )).scalar_one_or_none()
    if not n:
        raise HTTPException(404, "Norm not found")
    structs = (await db.execute(
        select(NormStructure).where(NormStructure.norm_id == n.id)
        .order_by(NormStructure.codigo_estrutura)
    )).scalars().all()
    return NormDetail(
        id=n.id, concessionaria=n.concessionaria, codigo=n.codigo, nome=n.nome,
        versao=n.versao, pdf_filename=n.pdf_filename, pdf_size_bytes=n.pdf_size_bytes,
        page_count=n.page_count, structure_count=len(structs), ativa=n.ativa,
        data_vigencia_inicio=n.data_vigencia_inicio, data_vigencia_fim=n.data_vigencia_fim,
        replaced_by_id=n.replaced_by_id, created_at=n.created_at,
        text_extracted=n.text_extracted[:50000], pdf_hash=n.pdf_hash,
        structures=[StructureOut(
            id=s.id, codigo_estrutura=s.codigo_estrutura, nome_completo=s.nome_completo,
            descricao_tecnica=s.descricao_tecnica, caracteristicas_visuais=s.caracteristicas_visuais,
            campos_proj=s.campos_proj, materiais=s.materiais,
        ) for s in structs],
    )


@router.post("", status_code=201)
async def upload_norm(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    concessionaria: str = Form(...),
    codigo: str = Form(...),
    nome: str = Form(...),
    versao: str = Form(default="1.0"),
    data_vigencia_inicio: str | None = Form(default=None),
    replaces_norm_id: str | None = Form(default=None),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "File must be .pdf")
    content = await file.read()
    if not content:
        raise HTTPException(400, "Empty file")
    if len(content) > MAX_PDF_BYTES:
        raise HTTPException(413, "PDF too large")

    pdf_hash = md5_of_bytes(content)
    existing = (await db.execute(
        select(TechnicalNorm).where(TechnicalNorm.tenant_id == user.tenant_id,
                                     TechnicalNorm.pdf_hash == pdf_hash)
    )).scalar_one_or_none()
    if existing:
        raise HTTPException(409, f"PDF identico ja existe: norm={existing.id} ({existing.codigo})")

    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = file.filename.replace("/", "_").replace("\\", "_")
    final_path = KNOWLEDGE_DIR / f"{pdf_hash[:12]}_{safe_name}"
    if final_path.exists():
        # filesystem dup but no DB row — keep the PDF (immutable storage)
        logger.warning(f"[norms] file already on disk, reusing: {final_path}")
    else:
        final_path.write_bytes(content)
        logger.info(f"[norms] saved PDF {final_path} ({len(content)} bytes)")

    text, pages = extract_text(content)
    logger.info(f"[norms] extracted {len(text)} chars from {pages} pages")

    vig_ini = None
    if data_vigencia_inicio:
        try:
            vig_ini = date.fromisoformat(data_vigencia_inicio)
        except ValueError:
            raise HTTPException(400, "data_vigencia_inicio formato invalido (YYYY-MM-DD)")

    norm = TechnicalNorm(
        tenant_id=user.tenant_id,
        concessionaria=concessionaria,
        codigo=codigo,
        nome=nome,
        versao=versao,
        pdf_filename=file.filename,
        pdf_path=str(final_path),
        pdf_hash=pdf_hash,
        pdf_size_bytes=len(content),
        text_extracted=text,
        page_count=pages,
        data_vigencia_inicio=vig_ini,
        ativa=True,
        criado_por_id=user.id,
    )

    if replaces_norm_id:
        old = (await db.execute(
            select(TechnicalNorm).where(TechnicalNorm.id == replaces_norm_id,
                                         TechnicalNorm.tenant_id == user.tenant_id)
        )).scalar_one_or_none()
        if old:
            old.ativa = False
            old.data_vigencia_fim = vig_ini or date.today()
            db.add(old)

    db.add(norm)
    await db.flush()

    if replaces_norm_id and (await db.execute(
        select(TechnicalNorm).where(TechnicalNorm.id == replaces_norm_id)
    )).scalar_one_or_none():
        old = (await db.execute(
            select(TechnicalNorm).where(TechnicalNorm.id == replaces_norm_id)
        )).scalar_one()
        old.replaced_by_id = norm.id

    await write_audit(
        db, action="norm.created", user_id=user.id, tenant_id=user.tenant_id,
        resource_type="technical_norm", resource_id=norm.id,
        metadata={"codigo": codigo, "concessionaria": concessionaria,
                  "pdf_hash": pdf_hash, "size": len(content)},
    )
    await db.commit()

    background.add_task(_extract_structures_async, norm.id, user.tenant_id, text)
    return {"id": norm.id, "status": "created", "structures_extraction": "in_progress"}


async def _extract_structures_async(norm_id: str, tenant_id: str, text: str):
    logger.info(f"[norms] starting structure extraction for {norm_id}")
    structures = await claude_structure(text)
    logger.info(f"[norms] {norm_id}: got {len(structures)} structures from Claude")
    if not structures:
        return
    async with session_scope() as db:
        for s in structures:
            db.add(NormStructure(
                tenant_id=tenant_id,
                norm_id=norm_id,
                codigo_estrutura=str(s.get("codigo_estrutura", ""))[:80],
                nome_completo=str(s.get("nome_completo", ""))[:500],
                descricao_tecnica=str(s.get("descricao_tecnica", "")),
                caracteristicas_visuais=str(s.get("caracteristicas_visuais", "")),
                campos_proj=s.get("campos_proj", {}) if isinstance(s.get("campos_proj"), dict) else {},
                materiais=s.get("materiais", []) if isinstance(s.get("materiais"), list) else [],
            ))


class DeactivateIn(BaseModel):
    motivo: str = ""


@router.patch("/{norm_id}/deactivate")
async def deactivate_norm(norm_id: str, payload: DeactivateIn,
                          db: AsyncSession = Depends(get_db),
                          user: User = Depends(get_current_user)):
    """SOFT-disable. PDF and DB row are preserved. Cannot be DELETED, only deactivated."""
    n = (await db.execute(
        select(TechnicalNorm).where(TechnicalNorm.id == norm_id,
                                     TechnicalNorm.tenant_id == user.tenant_id)
    )).scalar_one_or_none()
    if not n:
        raise HTTPException(404, "Norm not found")
    if not n.ativa:
        return {"id": n.id, "ativa": False, "already": True}
    n.ativa = False
    n.desativado_por_id = user.id
    n.desativado_em = datetime.now(timezone.utc)
    if not n.data_vigencia_fim:
        n.data_vigencia_fim = date.today()
    await write_audit(
        db, action="norm.deactivated", user_id=user.id, tenant_id=user.tenant_id,
        resource_type="technical_norm", resource_id=n.id,
        metadata={"motivo": payload.motivo[:500]},
    )
    await db.commit()
    return {"id": n.id, "ativa": False}


@router.patch("/{norm_id}/reactivate")
async def reactivate_norm(norm_id: str, db: AsyncSession = Depends(get_db),
                           user: User = Depends(get_current_user)):
    n = (await db.execute(
        select(TechnicalNorm).where(TechnicalNorm.id == norm_id,
                                     TechnicalNorm.tenant_id == user.tenant_id)
    )).scalar_one_or_none()
    if not n:
        raise HTTPException(404, "Norm not found")
    n.ativa = True
    n.data_vigencia_fim = None
    n.desativado_em = None
    await write_audit(
        db, action="norm.reactivated", user_id=user.id, tenant_id=user.tenant_id,
        resource_type="technical_norm", resource_id=n.id,
    )
    await db.commit()
    return {"id": n.id, "ativa": True}


@router.get("/{norm_id}/pdf")
async def download_pdf(norm_id: str, db: AsyncSession = Depends(get_db),
                        user: User = Depends(get_current_user)):
    """Serve the original PDF (read-only, immutable)."""
    from fastapi.responses import FileResponse
    n = (await db.execute(
        select(TechnicalNorm).where(TechnicalNorm.id == norm_id,
                                     TechnicalNorm.tenant_id == user.tenant_id)
    )).scalar_one_or_none()
    if not n:
        raise HTTPException(404, "Norm not found")
    p = Path(n.pdf_path)
    if not p.is_file():
        raise HTTPException(410, "PDF original missing on disk")
    return FileResponse(str(p), media_type="application/pdf", filename=n.pdf_filename)


# ─── DEEP REPROCESS (Vision-based) ────────────────────────────────────────────

@router.post("/{norm_id}/reprocess", status_code=202)
async def reprocess_norm(norm_id: str, background: BackgroundTasks,
                          db: AsyncSession = Depends(get_db),
                          user: User = Depends(get_current_user)):
    """Re-extract structures + materials from the original PDF using deep vision pipeline.
    Drops existing structures/materials for this norm and replaces with the new extraction."""
    n = (await db.execute(
        select(TechnicalNorm).where(TechnicalNorm.id == norm_id,
                                     TechnicalNorm.tenant_id == user.tenant_id)
    )).scalar_one_or_none()
    if not n:
        raise HTTPException(404, "Norm not found")
    if n.processing_status in ("classifying", "extracting"):
        raise HTTPException(409, f"Already in progress: {n.processing_status}")
    if not Path(n.pdf_path).is_file():
        raise HTTPException(410, "PDF file missing on disk")

    n.processing_status = "classifying"
    n.processing_progress = 0
    n.processing_message = "iniciando..."
    n.processing_started_at = datetime.now(timezone.utc)
    n.processing_finished_at = None
    await write_audit(
        db, action="norm.reprocess_started", user_id=user.id, tenant_id=user.tenant_id,
        resource_type="technical_norm", resource_id=n.id,
    )
    await db.commit()

    background.add_task(_run_deep_reprocess, n.id, n.tenant_id, n.pdf_path)
    return {"id": n.id, "status": "started", "message": "Reprocessamento iniciado em background"}


async def _run_deep_reprocess(norm_id: str, tenant_id: str, pdf_path: str):
    pages_dir = PAGES_DIR_BASE / norm_id
    pages_dir.mkdir(parents=True, exist_ok=True)

    async def progress_cb(step: str, current: int, total: int, message: str):
        async with session_scope() as db:
            n = (await db.execute(
                select(TechnicalNorm).where(TechnicalNorm.id == norm_id)
            )).scalar_one_or_none()
            if not n:
                return
            n.processing_status = step
            n.processing_progress = int(current / total * 100) if total > 0 else 0
            n.processing_message = message[:500]
            if step == "extracting":
                n.pages_with_drawings = total
                n.pages_processed = current

    try:
        result = await deep_process_pdf(pdf_path, norm_id, pages_dir, progress_cb)
    except Exception as e:
        logger.exception(f"[reprocess] {norm_id} failed: {e}")
        async with session_scope() as db:
            n = (await db.execute(
                select(TechnicalNorm).where(TechnicalNorm.id == norm_id)
            )).scalar_one_or_none()
            if n:
                n.processing_status = "failed"
                n.processing_message = f"{type(e).__name__}: {str(e)[:300]}"
                n.processing_finished_at = datetime.now(timezone.utc)
        return

    # Persist: drop old structures+materials of this norm, insert new
    async with session_scope() as db:
        # delete existing structures+materials for this norm (only DB rows; PDF stays)
        old_structs = (await db.execute(
            select(NormStructure).where(NormStructure.norm_id == norm_id)
        )).scalars().all()
        for s in old_structs:
            await db.delete(s)
        old_mats = (await db.execute(
            select(NormMaterial).where(NormMaterial.norm_id == norm_id)
        )).scalars().all()
        for m in old_mats:
            await db.delete(m)
        await db.flush()

        # Insert structures
        struct_id_by_code: dict[str, str] = {}
        for s in result.structures:
            row = NormStructure(
                tenant_id=tenant_id, norm_id=norm_id,
                codigo_estrutura=str(s.get("codigo_estrutura", ""))[:80],
                nome_completo=str(s.get("nome_completo", ""))[:500],
                descricao_tecnica=str(s.get("descricao_tecnica", "")),
                caracteristicas_visuais=str(s.get("caracteristicas_visuais", "")),
                como_identificar_na_foto=str(s.get("como_identificar_na_foto", "")),
                restricoes_uso=str(s.get("restricoes_uso", "")),
                tipo_rede=str(s.get("tipo_rede", ""))[:20],
                tensao_nominal=str(s.get("tensao_nominal", ""))[:40],
                desenho_numero=str(s.get("desenho_numero", ""))[:40],
                pagina_referencia=int(s.get("pagina_referencia", 0) or 0),
                imagem_desenho_path=str(s.get("imagem_desenho_path", "")),
                fixacao=s.get("fixacao", {}) if isinstance(s.get("fixacao"), dict) else {},
                extraction_confidence=float(s.get("extraction_confidence", 0.0) or 0.0),
                requires_review=bool(s.get("requires_review", False)),
                campos_proj={},
                materiais=[],
            )
            db.add(row)
            await db.flush()
            struct_id_by_code[row.codigo_estrutura] = row.id

        # Insert materials, mapping structure_codes -> structure ids
        for codigo, m in result.materials_by_code.items():
            ids = [struct_id_by_code[c] for c in m["structure_codes"]
                   if c in struct_id_by_code]
            db.add(NormMaterial(
                tenant_id=tenant_id, norm_id=norm_id,
                codigo_material=str(codigo)[:80],
                codigo_item=str(m.get("codigo_item", ""))[:40],
                descricao=str(m.get("descricao", "")),
                tensao=str(m.get("tensao", ""))[:40],
                used_in_structures=ids,
            ))

        # Update norm status
        n = (await db.execute(
            select(TechnicalNorm).where(TechnicalNorm.id == norm_id)
        )).scalar_one()
        n.processing_status = "done"
        n.processing_progress = 100
        n.processing_message = (
            f"Concluido: {len(result.structures)} estruturas, "
            f"{len(result.materials_by_code)} materiais, "
            f"{result.pages_processed}/{result.pages_drawing} paginas com Vision "
            f"de {result.pages_total} totais"
        )
        n.processing_finished_at = datetime.now(timezone.utc)
        n.pages_total = result.pages_total
        n.pages_with_drawings = result.pages_drawing
        n.pages_processed = result.pages_processed

    logger.info(f"[reprocess] {norm_id} done: {len(result.structures)} structures, "
                 f"{len(result.materials_by_code)} materials")


@router.get("/{norm_id}/reprocess_status")
async def reprocess_status(norm_id: str, db: AsyncSession = Depends(get_db),
                            user: User = Depends(get_current_user)):
    n = (await db.execute(
        select(TechnicalNorm).where(TechnicalNorm.id == norm_id,
                                     TechnicalNorm.tenant_id == user.tenant_id)
    )).scalar_one_or_none()
    if not n:
        raise HTTPException(404, "Norm not found")
    return {
        "id": n.id,
        "status": n.processing_status,
        "progress": n.processing_progress,
        "message": n.processing_message,
        "started_at": n.processing_started_at.isoformat() if n.processing_started_at else None,
        "finished_at": n.processing_finished_at.isoformat() if n.processing_finished_at else None,
        "pages_total": n.pages_total,
        "pages_with_drawings": n.pages_with_drawings,
        "pages_processed": n.pages_processed,
    }


@router.get("/{norm_id}/materials")
async def list_norm_materials(norm_id: str, db: AsyncSession = Depends(get_db),
                               user: User = Depends(get_current_user)):
    n = (await db.execute(
        select(TechnicalNorm).where(TechnicalNorm.id == norm_id,
                                     TechnicalNorm.tenant_id == user.tenant_id)
    )).scalar_one_or_none()
    if not n:
        raise HTTPException(404, "Norm not found")
    rows = (await db.execute(
        select(NormMaterial).where(NormMaterial.norm_id == norm_id)
        .order_by(NormMaterial.codigo_item, NormMaterial.codigo_material)
    )).scalars().all()
    return [{
        "id": m.id,
        "codigo_item": m.codigo_item,
        "codigo_material": m.codigo_material,
        "descricao": m.descricao,
        "tensao": m.tensao,
        "used_in_structures": m.used_in_structures,
        "structure_count": len(m.used_in_structures),
    } for m in rows]


@router.get("/{norm_id}/page_image")
async def page_image(norm_id: str, page: int,
                      db: AsyncSession = Depends(get_db),
                      user: User = Depends(get_current_user)):
    """Serve a specific rendered page (drawing) of a norm."""
    from fastapi.responses import FileResponse
    n = (await db.execute(
        select(TechnicalNorm).where(TechnicalNorm.id == norm_id,
                                     TechnicalNorm.tenant_id == user.tenant_id)
    )).scalar_one_or_none()
    if not n:
        raise HTTPException(404, "Norm not found")
    p = PAGES_DIR_BASE / norm_id / f"page_{page:04d}.jpg"
    if not p.is_file():
        raise HTTPException(404, "Page image not rendered")
    return FileResponse(str(p), media_type="image/jpeg")
