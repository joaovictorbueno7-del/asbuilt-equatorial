from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.core.queue import queue
from app.models import User, AgentRun, AgentRunStatus
from app.auth.dependencies import get_current_user
from agents import list_agents, get_agent, AGENTS

router = APIRouter()


class AgentInfo(BaseModel):
    code: str
    name: str
    description: str
    version: str
    consumes_from: str | None
    requires_human_review_below: float


class RunIn(BaseModel):
    agent_code: str
    payload: dict = Field(default_factory=dict)
    parent_run_id: str | None = None


class RunOut(BaseModel):
    id: str
    agent_code: str
    status: str
    confidence_score: float
    output_payload: dict
    error_message: str
    started_at: datetime | None
    finished_at: datetime | None


@router.get("", response_model=list[AgentInfo])
async def get_agents(_: User = Depends(get_current_user)):
    return list_agents()


@router.post("/run", response_model=RunOut, status_code=202)
async def run_agent(payload: RunIn, db: AsyncSession = Depends(get_db),
                    user: User = Depends(get_current_user)):
    if payload.agent_code not in AGENTS:
        raise HTTPException(404, f"Unknown agent: {payload.agent_code}")

    run = AgentRun(
        tenant_id=user.tenant_id,
        user_id=user.id,
        agent_code=payload.agent_code,
        status=AgentRunStatus.PENDING,
        input_payload=payload.payload,
        parent_run_id=payload.parent_run_id,
    )
    db.add(run)
    await db.commit()
    await db.refresh(run)

    await queue.enqueue(_execute_run, run.id)

    return _to_out(run)


async def _execute_run(run_id: str):
    """Background execution. Reopens its own session."""
    from app.core.database import session_scope
    async with session_scope() as db:
        run = (await db.execute(select(AgentRun).where(AgentRun.id == run_id))).scalar_one_or_none()
        if not run:
            return
        run.status = AgentRunStatus.RUNNING
        run.started_at = datetime.now(timezone.utc)
        await db.flush()

        try:
            agent = get_agent(run.agent_code)
            await agent.validate_input(run.input_payload)
            result = await agent.run(run.input_payload, context={"tenant_id": run.tenant_id, "user_id": run.user_id})
            run.output_payload = result.output
            run.confidence_score = result.confidence
            if result.needs_human:
                run.status = AgentRunStatus.NEEDS_HUMAN
            else:
                run.status = AgentRunStatus.COMPLETED
            run.error_message = result.notes
        except Exception as e:
            run.status = AgentRunStatus.FAILED
            run.error_message = f"{type(e).__name__}: {e}"
        finally:
            run.finished_at = datetime.now(timezone.utc)


@router.get("/runs/{run_id}", response_model=RunOut)
async def get_run(run_id: str, db: AsyncSession = Depends(get_db),
                  user: User = Depends(get_current_user)):
    run = (
        await db.execute(
            select(AgentRun).where(AgentRun.id == run_id, AgentRun.tenant_id == user.tenant_id)
        )
    ).scalar_one_or_none()
    if not run:
        raise HTTPException(404, "Run not found")
    return _to_out(run)


@router.get("/runs", response_model=list[RunOut])
async def list_runs(limit: int = 50, agent_code: str | None = None,
                    db: AsyncSession = Depends(get_db),
                    user: User = Depends(get_current_user)):
    q = select(AgentRun).where(AgentRun.tenant_id == user.tenant_id)
    if agent_code:
        q = q.where(AgentRun.agent_code == agent_code)
    q = q.order_by(desc(AgentRun.created_at)).limit(min(limit, 200))
    rows = (await db.execute(q)).scalars().all()
    return [_to_out(r) for r in rows]


def _to_out(run: AgentRun) -> RunOut:
    return RunOut(
        id=run.id,
        agent_code=run.agent_code,
        status=run.status.value,
        confidence_score=run.confidence_score,
        output_payload=run.output_payload,
        error_message=run.error_message,
        started_at=run.started_at,
        finished_at=run.finished_at,
    )
