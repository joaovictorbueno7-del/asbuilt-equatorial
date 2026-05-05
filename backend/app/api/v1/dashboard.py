"""Dashboard aggregate stats and notifications derived from pipelines/runs."""
from __future__ import annotations
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from fastapi import APIRouter, Depends
from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.models import (
    User, PipelineRun, PipelineStatus, AgentRun, AgentRunStatus, LearningCase,
)
from app.auth.dependencies import get_current_user
from agents import AGENTS

router = APIRouter()


def _agent_mode(cases_total: int, accuracy: float) -> str:
    """training (< 10 cases) | shadow (>=10 but accuracy<0.95) | production (accuracy>=0.95)."""
    if cases_total < 10:
        return "training"
    if accuracy < 0.95:
        return "shadow"
    return "production"


@router.get("/dashboard/stats")
async def dashboard_stats(db: AsyncSession = Depends(get_db),
                          user: User = Depends(get_current_user)):
    tenant = user.tenant_id

    pipes = (await db.execute(
        select(PipelineRun).where(PipelineRun.tenant_id == tenant)
        .order_by(desc(PipelineRun.created_at))
    )).scalars().all()
    total = len(pipes)
    completed = sum(1 for p in pipes if p.status == PipelineStatus.COMPLETED)
    needs_human = sum(1 for p in pipes if p.status == PipelineStatus.NEEDS_HUMAN)
    failed = sum(1 for p in pipes if p.status == PipelineStatus.FAILED)
    running = sum(1 for p in pipes if p.status in (PipelineStatus.PENDING, PipelineStatus.RUNNING))
    finished = total - running
    success_rate = round(completed / finished * 100, 1) if finished else 0.0

    # Weekly avg score for last 8 weeks (Mon-Sun)
    now = datetime.now(timezone.utc)
    week_buckets: dict[str, list[float]] = defaultdict(list)
    for w in range(8):
        week_start = (now - timedelta(days=now.weekday() + 7 * w)).date()
        week_buckets[week_start.isoformat()] = []
    for p in pipes:
        if not p.finished_at or p.overall_score <= 0:
            continue
        week_start = (p.finished_at - timedelta(days=p.finished_at.weekday())).date()
        key = week_start.isoformat()
        if key in week_buckets:
            week_buckets[key].append(p.overall_score)
    weekly = sorted([
        {"week": k, "avg_score": round(sum(v) / len(v), 1) if v else None, "count": len(v)}
        for k, v in week_buckets.items()
    ], key=lambda x: x["week"])

    # Per-agent stats
    runs = (await db.execute(
        select(AgentRun).where(AgentRun.tenant_id == tenant)
    )).scalars().all()
    cases = (await db.execute(
        select(LearningCase).where(LearningCase.tenant_id == tenant)
    )).scalars().all()
    cases_by_agent: dict[str, list[LearningCase]] = defaultdict(list)
    for c in cases:
        cases_by_agent[c.agent_code].append(c)

    per_agent = []
    alerts = []
    for code, cls in AGENTS.items():
        agent_runs = [r for r in runs if r.agent_code == code]
        finished_runs = [r for r in agent_runs if r.status in (AgentRunStatus.COMPLETED,
                                                                AgentRunStatus.NEEDS_HUMAN,
                                                                AgentRunStatus.FAILED)]
        avg_conf = (sum(r.confidence_score for r in finished_runs) / len(finished_runs)) if finished_runs else 0.0
        agent_cases = cases_by_agent.get(code, [])
        # accuracy: cases with score > 0.5 / total reviewed cases
        reviewed = [c for c in agent_cases if c.times_used > 0]
        accuracy = (sum(1 for c in reviewed if c.is_correct) / len(reviewed)) if reviewed else 0.0
        mode = _agent_mode(len(agent_cases), accuracy)
        pending_feedback = sum(1 for c in agent_cases if c.times_used == 0)

        per_agent.append({
            "code": code,
            "name": cls.meta.name,
            "avg_confidence": round(avg_conf, 3),
            "runs_count": len(agent_runs),
            "completed_runs": sum(1 for r in agent_runs if r.status == AgentRunStatus.COMPLETED),
            "needs_human_runs": sum(1 for r in agent_runs if r.status == AgentRunStatus.NEEDS_HUMAN),
            "failed_runs": sum(1 for r in agent_runs if r.status == AgentRunStatus.FAILED),
            "cases_total": len(agent_cases),
            "cases_pending_feedback": pending_feedback,
            "accuracy": round(accuracy, 3),
            "mode": mode,
            "is_alert": avg_conf > 0 and avg_conf < 0.70,
        })
        if avg_conf > 0 and avg_conf < 0.70:
            alerts.append({
                "type": "low_confidence",
                "agent_code": code,
                "agent_name": cls.meta.name,
                "value": round(avg_conf, 2),
                "message": f"{cls.meta.name}: confiança média {round(avg_conf * 100)}% (alvo ≥ 70%)",
            })

    return {
        "totals": {
            "pipelines": total,
            "completed": completed,
            "needs_human": needs_human,
            "failed": failed,
            "running": running,
        },
        "success_rate": success_rate,
        "weekly_scores": weekly,
        "per_agent": per_agent,
        "alerts": alerts,
    }


@router.get("/notifications")
async def notifications(limit: int = 20, db: AsyncSession = Depends(get_db),
                        user: User = Depends(get_current_user)):
    """Derive notifications from recent pipelines and runs. No separate table needed."""
    tenant = user.tenant_id
    cutoff = datetime.now(timezone.utc) - timedelta(days=7)

    pipes = (await db.execute(
        select(PipelineRun)
        .where(PipelineRun.tenant_id == tenant, PipelineRun.created_at >= cutoff)
        .order_by(desc(PipelineRun.created_at))
        .limit(50)
    )).scalars().all()

    items = []
    for p in pipes:
        if p.status == PipelineStatus.COMPLETED:
            items.append({
                "id": f"pipe-completed-{p.id}",
                "type": "pipeline_completed",
                "level": "success",
                "title": f"Obra concluída: {p.work_name}",
                "message": f"Score {round(p.overall_score)}/100",
                "link": f"/pipelines/{p.id}",
                "ts": (p.finished_at or p.updated_at).isoformat(),
            })
        elif p.status == PipelineStatus.NEEDS_HUMAN:
            items.append({
                "id": f"pipe-needs-{p.id}",
                "type": "pipeline_needs_human",
                "level": "warning",
                "title": f"Revisão humana: {p.work_name}",
                "message": "Score baixo ou agente requer revisão",
                "link": f"/pipelines/{p.id}",
                "ts": (p.finished_at or p.updated_at).isoformat(),
            })
        elif p.status == PipelineStatus.FAILED:
            items.append({
                "id": f"pipe-failed-{p.id}",
                "type": "pipeline_failed",
                "level": "error",
                "title": f"Falha: {p.work_name}",
                "message": p.error_message[:120] or "Erro no processamento",
                "link": f"/pipelines/{p.id}",
                "ts": (p.finished_at or p.updated_at).isoformat(),
            })

    # Special: agent 07 milestone — accuracy >= 0.95
    rpa_cases = (await db.execute(
        select(LearningCase)
        .where(LearningCase.tenant_id == tenant,
               LearningCase.agent_code == "rpa_screen_learner")
    )).scalars().all()
    reviewed = [c for c in rpa_cases if c.times_used > 0]
    if len(reviewed) >= 10:
        accuracy = sum(1 for c in reviewed if c.is_correct) / len(reviewed)
        if accuracy >= 0.95:
            items.append({
                "id": "rpa-milestone-95",
                "type": "rpa_milestone",
                "level": "milestone",
                "title": "🎯 Agente 07 atingiu 95% de acerto",
                "message": f"RPA Screen Learner está pronto para autonomia ({round(accuracy * 100)}%)",
                "link": "/agents/rpa_screen_learner",
                "ts": datetime.now(timezone.utc).isoformat(),
            })

    items.sort(key=lambda x: x["ts"], reverse=True)
    return {"items": items[:limit], "unread_count": len(items)}


@router.get("/agents/{code}/stats")
async def agent_stats(code: str, db: AsyncSession = Depends(get_db),
                      user: User = Depends(get_current_user)):
    if code not in AGENTS:
        from fastapi import HTTPException
        raise HTTPException(404, "Unknown agent")
    cls = AGENTS[code]
    tenant = user.tenant_id

    runs = (await db.execute(
        select(AgentRun).where(AgentRun.tenant_id == tenant, AgentRun.agent_code == code)
        .order_by(desc(AgentRun.created_at)).limit(50)
    )).scalars().all()
    cases = (await db.execute(
        select(LearningCase).where(LearningCase.tenant_id == tenant, LearningCase.agent_code == code)
        .order_by(desc(LearningCase.created_at))
    )).scalars().all()
    finished = [r for r in runs if r.status in (AgentRunStatus.COMPLETED, AgentRunStatus.NEEDS_HUMAN, AgentRunStatus.FAILED)]
    avg_conf = (sum(r.confidence_score for r in finished) / len(finished)) if finished else 0.0
    reviewed = [c for c in cases if c.times_used > 0]
    accuracy = (sum(1 for c in reviewed if c.is_correct) / len(reviewed)) if reviewed else 0.0
    mode = _agent_mode(len(cases), accuracy)

    return {
        "code": code,
        "name": cls.meta.name,
        "description": cls.meta.description,
        "version": cls.meta.version,
        "consumes_from": cls.meta.consumes_from,
        "mode": mode,
        "avg_confidence": round(avg_conf, 3),
        "accuracy": round(accuracy, 3),
        "totals": {
            "runs": len(runs),
            "completed": sum(1 for r in runs if r.status == AgentRunStatus.COMPLETED),
            "needs_human": sum(1 for r in runs if r.status == AgentRunStatus.NEEDS_HUMAN),
            "failed": sum(1 for r in runs if r.status == AgentRunStatus.FAILED),
            "cases": len(cases),
            "cases_pending_feedback": sum(1 for c in cases if c.times_used == 0),
        },
        "recent_runs": [{
            "id": r.id,
            "status": r.status.value,
            "confidence_score": r.confidence_score,
            "created_at": r.created_at.isoformat(),
            "pipeline_id": r.pipeline_id,
        } for r in runs[:10]],
    }
