from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.core.queue import queue
from app.models import User, AgentRun, AgentRunStatus, LearningCase
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
            for case in result.learning_cases:
                db.add(LearningCase(
                    tenant_id=run.tenant_id,
                    agent_code=run.agent_code,
                    source_run_id=run.id,
                    input_payload=case.get("input", {}),
                    observed_output=case.get("output", {}),
                    feedback_score=0.5,
                ))
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


class FeedbackIn(BaseModel):
    is_correct: bool
    notes: str = ""


class CorrectionIn(BaseModel):
    """Structured correction payload. Saved as expected_output for few-shot training."""
    structure_type: str | None = None
    condition: str | None = None
    non_conformities: list[str] | None = None
    details: str | None = None
    notes: str = ""


@router.get("/{agent_code}/cases", response_model=list[dict])
async def list_cases(agent_code: str, limit: int = 50,
                     source_run_id: str | None = None,
                     db: AsyncSession = Depends(get_db),
                     user: User = Depends(get_current_user)):
    q = select(LearningCase).where(
        LearningCase.tenant_id == user.tenant_id,
        LearningCase.agent_code == agent_code,
    )
    if source_run_id:
        q = q.where(LearningCase.source_run_id == source_run_id)
    q = q.order_by(desc(LearningCase.created_at)).limit(min(limit, 500))
    rows = (await db.execute(q)).scalars().all()
    return [{
        "id": c.id,
        "agent_code": c.agent_code,
        "feedback_score": c.feedback_score,
        "is_correct": c.is_correct,
        "human_notes": c.human_notes,
        "input": c.input_payload,
        "output": c.observed_output,
        "expected_output": c.expected_output,
        "source_run_id": c.source_run_id,
        "created_at": c.created_at,
    } for c in rows]


@router.post("/{agent_code}/cases/{case_id}/correct")
async def submit_correction(agent_code: str, case_id: str, payload: CorrectionIn,
                            db: AsyncSession = Depends(get_db),
                            user: User = Depends(get_current_user)):
    """Operator provides the *correct* answer. Marks case as not correct,
    saves expected_output, applies -0.10 penalty to feedback_score.
    Used as positive training example (the corrected version) on next runs."""
    case = (await db.execute(
        select(LearningCase).where(
            LearningCase.id == case_id,
            LearningCase.tenant_id == user.tenant_id,
            LearningCase.agent_code == agent_code,
        )
    )).scalar_one_or_none()
    if not case:
        raise HTTPException(404, "Case not found")
    obs = case.observed_output or {}
    expected = {
        "structure_type": payload.structure_type or obs.get("structure_type", "outro"),
        "condition": payload.condition or obs.get("condition", "regular"),
        "non_conformities": payload.non_conformities if payload.non_conformities is not None
                            else obs.get("non_conformities", []),
        "details": payload.details or obs.get("details", ""),
    }
    case.expected_output = expected
    case.is_correct = True   # corrected version becomes the canonical positive example
    case.human_notes = payload.notes
    case.feedback_score = max(0.0, min(1.0, case.feedback_score - 0.10))
    case.times_used += 1
    await db.commit()
    return {
        "id": case.id,
        "feedback_score": case.feedback_score,
        "expected_output": case.expected_output,
        "times_used": case.times_used,
    }


@router.post("/{agent_code}/cases/{case_id}/feedback")
async def submit_feedback(agent_code: str, case_id: str, payload: FeedbackIn,
                          db: AsyncSession = Depends(get_db),
                          user: User = Depends(get_current_user)):
    """Operator feedback. Score starts at 0.50 (initial), +0.05 if correct, -0.10 if wrong."""
    case = (await db.execute(
        select(LearningCase).where(
            LearningCase.id == case_id,
            LearningCase.tenant_id == user.tenant_id,
            LearningCase.agent_code == agent_code,
        )
    )).scalar_one_or_none()
    if not case:
        raise HTTPException(404, "Case not found")
    delta = 0.05 if payload.is_correct else -0.10
    case.feedback_score = max(0.0, min(1.0, case.feedback_score + delta))
    case.is_correct = payload.is_correct
    case.human_notes = payload.notes
    case.times_used += 1
    await db.commit()
    return {
        "id": case.id,
        "feedback_score": case.feedback_score,
        "is_correct": case.is_correct,
        "times_used": case.times_used,
    }


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
