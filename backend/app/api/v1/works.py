"""High-level 'work' endpoint: upload a KMZ, get back a run that processes it."""
from __future__ import annotations
import uuid
from pathlib import Path
from datetime import datetime
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from pydantic import BaseModel
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.core.config import settings
from app.core.queue import queue
from app.models import User, AgentRun, AgentRunStatus
from app.auth.dependencies import get_current_user
from app.api.v1.agents import _execute_run

router = APIRouter()

MAX_KMZ_BYTES = 100 * 1024 * 1024  # 100 MB


class WorkOut(BaseModel):
    run_id: str
    status: str
    agent_code: str
    original_filename: str


class WorkDetail(BaseModel):
    run_id: str
    agent_code: str
    status: str
    confidence_score: float
    output: dict
    error: str
    started_at: datetime | None
    finished_at: datetime | None
    created_at: datetime


from fastapi import Form


@router.post("", response_model=WorkOut, status_code=202)
async def create_work(
    file: UploadFile = File(...),
    work_name: str = Form(default=""),
    concessionaria: str = Form(default=""),
    tipo: str = Form(default="as_built"),
    municipio: str = Form(default=""),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    if not file.filename or not file.filename.lower().endswith(".kmz"):
        raise HTTPException(400, "File must have .kmz extension")
    content = await file.read()
    if not content:
        raise HTTPException(400, "Empty file")
    if len(content) > MAX_KMZ_BYTES:
        raise HTTPException(413, f"KMZ exceeds max size {MAX_KMZ_BYTES} bytes")

    storage = Path(settings.STORAGE_LOCAL_PATH) / "kmz"
    storage.mkdir(parents=True, exist_ok=True)
    kmz_path = storage / f"{uuid.uuid4()}.kmz"
    kmz_path.write_bytes(content)

    run = AgentRun(
        tenant_id=user.tenant_id,
        user_id=user.id,
        agent_code="kmz_analyzer",
        status=AgentRunStatus.PENDING,
        input_payload={
            "kmz_path": str(kmz_path),
            "original_filename": file.filename,
            "size_bytes": len(content),
            "work_name": work_name or file.filename,
            "concessionaria": concessionaria,
            "tipo": tipo,
            "municipio": municipio,
        },
    )
    db.add(run)
    await db.commit()
    await db.refresh(run)
    await queue.enqueue(_execute_run, run.id)

    return WorkOut(run_id=run.id, status=run.status.value,
                   agent_code=run.agent_code, original_filename=file.filename)


@router.get("", response_model=list[dict])
async def list_works(limit: int = 50, db: AsyncSession = Depends(get_db),
                     user: User = Depends(get_current_user)):
    rows = (await db.execute(
        select(AgentRun)
        .where(AgentRun.tenant_id == user.tenant_id, AgentRun.agent_code == "kmz_analyzer")
        .order_by(desc(AgentRun.created_at))
        .limit(min(limit, 200))
    )).scalars().all()
    return [{
        "run_id": r.id,
        "status": r.status.value,
        "confidence_score": r.confidence_score,
        "work_name": r.input_payload.get("work_name") or r.input_payload.get("original_filename", ""),
        "concessionaria": r.input_payload.get("concessionaria", ""),
        "tipo": r.input_payload.get("tipo", ""),
        "image_count": r.output_payload.get("image_count", 0),
        "quality_score": r.output_payload.get("quality_score", 0),
        "non_conformity_count": len(r.output_payload.get("non_conformities", [])),
        "created_at": r.created_at.isoformat(),
        "finished_at": r.finished_at.isoformat() if r.finished_at else None,
    } for r in rows]


@router.get("/{run_id}/image")
async def get_work_image(run_id: str, key: str,
                         db: AsyncSession = Depends(get_db),
                         user: User = Depends(get_current_user)):
    """Serve an image extracted from the KMZ. `key` is the path inside the zip."""
    from fastapi.responses import Response
    import zipfile
    run = (await db.execute(
        select(AgentRun).where(AgentRun.id == run_id, AgentRun.tenant_id == user.tenant_id)
    )).scalar_one_or_none()
    if not run:
        raise HTTPException(404, "Work not found")
    kmz_path = run.input_payload.get("kmz_path")
    if not kmz_path or not Path(kmz_path).is_file():
        raise HTTPException(404, "KMZ file missing")
    safe = key.replace("..", "").lstrip("/").lstrip("\\")
    try:
        with zipfile.ZipFile(kmz_path, "r") as zf:
            names = zf.namelist()
            target = next((n for n in names if n == safe or Path(n).name == Path(safe).name), None)
            if not target:
                raise HTTPException(404, "Image not in KMZ")
            data = zf.read(target)
    except (zipfile.BadZipFile, KeyError):
        raise HTTPException(404, "Image not readable")
    ext = Path(target).suffix.lower()
    mt = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}.get(ext, "application/octet-stream")
    return Response(content=data, media_type=mt, headers={"Cache-Control": "private, max-age=3600"})


@router.get("/{run_id}/report/download")
async def download_report(
    run_id: str,
    fmt: str = "docx",
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Gera e devolve o relatório DOCX ou PDF de um work.

    Query param `fmt`: "docx" (padrão) ou "pdf".
    """
    from fastapi.responses import Response
    from app.services.report_builder import generate_report

    run = (await db.execute(
        select(AgentRun).where(AgentRun.id == run_id, AgentRun.tenant_id == user.tenant_id)
    )).scalar_one_or_none()
    if not run:
        raise HTTPException(404, "Work not found")

    if run.status.value not in ("completed", "done", "success"):
        if not run.output_payload:
            raise HTTPException(409, "Work ainda não foi processado")

    fmt = fmt.lower().strip(".")
    if fmt not in ("docx", "pdf"):
        raise HTTPException(400, "fmt deve ser 'docx' ou 'pdf'")

    kmz_path = run.input_payload.get("kmz_path")
    work_name = run.input_payload.get("work_name") or run.input_payload.get("original_filename", "")

    # Injeta metadados da obra no payload do report builder
    payload = dict(run.output_payload)
    payload.setdefault("nota", work_name)
    payload.setdefault("municipio", run.input_payload.get("municipio", ""))
    payload.setdefault("parceira",
                       run.input_payload.get("parceira")
                       or run.input_payload.get("concessionaria", ""))
    payload.setdefault("tipo", run.input_payload.get("tipo", "Postes, Estruturas e Redes"))
    payload["run_id"] = run_id

    try:
        data, filename = generate_report(
            run_id=run_id,
            output_payload=payload,
            kmz_path=kmz_path,
            fmt=fmt,
        )
    except Exception as e:
        raise HTTPException(500, f"Erro ao gerar relatório: {e}")

    media_types = {
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "pdf": "application/pdf",
    }
    return Response(
        content=data,
        media_type=media_types[fmt],
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Length": str(len(data)),
        },
    )


@router.get("/{run_id}", response_model=WorkDetail)
async def get_work(run_id: str, db: AsyncSession = Depends(get_db),
                   user: User = Depends(get_current_user)):
    run = (await db.execute(
        select(AgentRun).where(AgentRun.id == run_id, AgentRun.tenant_id == user.tenant_id)
    )).scalar_one_or_none()
    if not run:
        raise HTTPException(404, "Work not found")
    return WorkDetail(
        run_id=run.id,
        agent_code=run.agent_code,
        status=run.status.value,
        confidence_score=run.confidence_score,
        output=run.output_payload,
        error=run.error_message,
        started_at=run.started_at,
        finished_at=run.finished_at,
        created_at=run.created_at,
    )
