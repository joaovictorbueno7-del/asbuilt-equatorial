from __future__ import annotations
import uuid
from pathlib import Path
from datetime import datetime
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.core.config import settings
from app.models import User, PipelineRun, PipelineStatus, AgentRun
from app.auth.dependencies import get_current_user
from app.services.pipeline import enqueue_pipeline, PIPELINE_DAG

router = APIRouter()

MAX_KMZ_BYTES = 100 * 1024 * 1024


class PipelineOut(BaseModel):
    id: str
    status: str
    work_name: str


@router.post("", response_model=PipelineOut, status_code=202)
async def create_pipeline(
    file: UploadFile = File(...),
    work_name: str = Form(default=""),
    concessionaria: str = Form(default=""),
    tipo: str = Form(default="as_built"),
    municipio: str = Form(default=""),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    if not file.filename or not file.filename.lower().endswith(".kmz"):
        raise HTTPException(400, "File must be .kmz")
    content = await file.read()
    if not content:
        raise HTTPException(400, "Empty file")
    if len(content) > MAX_KMZ_BYTES:
        raise HTTPException(413, "KMZ too large")

    storage = Path(settings.STORAGE_LOCAL_PATH) / "kmz"
    storage.mkdir(parents=True, exist_ok=True)
    kmz_path = storage / f"{uuid.uuid4()}.kmz"
    kmz_path.write_bytes(content)

    pipe = PipelineRun(
        tenant_id=user.tenant_id,
        user_id=user.id,
        work_name=work_name or file.filename,
        concessionaria=concessionaria,
        tipo=tipo,
        status=PipelineStatus.PENDING,
        input_payload={
            "kmz_path": str(kmz_path),
            "original_filename": file.filename,
            "size_bytes": len(content),
            "municipio": municipio,
        },
    )
    db.add(pipe)
    await db.commit()
    await db.refresh(pipe)
    await enqueue_pipeline(pipe.id)
    return PipelineOut(id=pipe.id, status=pipe.status.value, work_name=pipe.work_name)


@router.get("")
async def list_pipelines(limit: int = 50, db: AsyncSession = Depends(get_db),
                         user: User = Depends(get_current_user)):
    rows = (await db.execute(
        select(PipelineRun)
        .where(PipelineRun.tenant_id == user.tenant_id)
        .order_by(desc(PipelineRun.created_at))
        .limit(min(limit, 200))
    )).scalars().all()
    return [{
        "id": p.id,
        "work_name": p.work_name,
        "concessionaria": p.concessionaria,
        "tipo": p.tipo,
        "status": p.status.value,
        "overall_score": p.overall_score,
        "created_at": p.created_at.isoformat(),
        "started_at": p.started_at.isoformat() if p.started_at else None,
        "finished_at": p.finished_at.isoformat() if p.finished_at else None,
    } for p in rows]


@router.get("/{pipeline_id}")
async def get_pipeline(pipeline_id: str, db: AsyncSession = Depends(get_db),
                       user: User = Depends(get_current_user)):
    pipe = (await db.execute(
        select(PipelineRun).where(PipelineRun.id == pipeline_id,
                                  PipelineRun.tenant_id == user.tenant_id)
    )).scalar_one_or_none()
    if not pipe:
        raise HTTPException(404, "Pipeline not found")

    runs = (await db.execute(
        select(AgentRun).where(AgentRun.pipeline_id == pipeline_id)
        .order_by(AgentRun.created_at)
    )).scalars().all()

    runs_by_code = {r.agent_code: r for r in runs}
    agents_status = []
    for code, deps in PIPELINE_DAG.items():
        r = runs_by_code.get(code)
        agents_status.append({
            "agent_code": code,
            "depends_on": deps,
            "run_id": r.id if r else None,
            "status": r.status.value if r else "pending",
            "confidence_score": r.confidence_score if r else 0.0,
            "started_at": r.started_at.isoformat() if r and r.started_at else None,
            "finished_at": r.finished_at.isoformat() if r and r.finished_at else None,
            "error": r.error_message if r else "",
            "output_summary": _summary(r) if r else None,
        })

    return {
        "id": pipe.id,
        "work_name": pipe.work_name,
        "concessionaria": pipe.concessionaria,
        "tipo": pipe.tipo,
        "status": pipe.status.value,
        "overall_score": pipe.overall_score,
        "summary_output": pipe.summary_output,
        "error_message": pipe.error_message,
        "created_at": pipe.created_at.isoformat(),
        "started_at": pipe.started_at.isoformat() if pipe.started_at else None,
        "finished_at": pipe.finished_at.isoformat() if pipe.finished_at else None,
        "agents": agents_status,
    }


def _summary(r: AgentRun) -> dict:
    out = r.output_payload or {}
    return {
        "image_count": out.get("image_count"),
        "quality_score": out.get("quality_score"),
        "filled_count": out.get("filled_count"),
        "blocking_issues": len(out.get("blocking_issues", [])) if isinstance(out.get("blocking_issues"), list) else None,
        "warnings": len(out.get("warnings", [])) if isinstance(out.get("warnings"), list) else None,
        "ready_to_send": out.get("ready_to_send"),
        "decision": out.get("decision"),
        "overall_score": out.get("overall_score"),
        "executive_summary": out.get("executive_summary"),
    }
