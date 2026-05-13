"""Pipeline orchestrator: runs the 9 agents respecting dependencies, in parallel where possible."""
from __future__ import annotations
import asyncio
from datetime import datetime, timezone
from loguru import logger
from sqlalchemy import select, desc
from app.core.database import session_scope
from app.core.queue import queue
from app.models import (
    PipelineRun, PipelineStatus, AgentRun, AgentRunStatus, LearningCase,
)
from agents import get_agent

FEW_SHOT_LIMIT = 5


async def _fetch_few_shot(db, tenant_id: str, agent_code: str) -> list[dict]:
    """Pull up to N approved cases (is_correct=true, times_used>0) for few-shot."""
    rows = (await db.execute(
        select(LearningCase)
        .where(LearningCase.tenant_id == tenant_id,
               LearningCase.agent_code == agent_code,
               LearningCase.is_correct == True,  # noqa: E712
               LearningCase.times_used > 0)
        .order_by(desc(LearningCase.feedback_score), desc(LearningCase.created_at))
        .limit(FEW_SHOT_LIMIT)
    )).scalars().all()
    return [{
        "input": c.input_payload,
        "expected_output": c.expected_output or {},
        "observed_output": c.observed_output or {},
        "human_notes": c.human_notes,
    } for c in rows]

# DAG: agent_code -> list of upstream agent_codes that must complete first.
# Empty list = can start immediately.
PIPELINE_DAG: dict[str, list[str]] = {
    "kmz_analyzer": [],
    "utm_converter": [],          # parallel to 01
    "adherence_tester": [],       # parallel
    "description_filler": ["kmz_analyzer"],
    "report_generator": ["description_filler"],
    "anti_reprova": ["report_generator"],
    "pipeline_supervisor": ["kmz_analyzer", "description_filler",
                             "report_generator", "anti_reprova"],
    "rpa_screen_learner": ["report_generator"],
    "master_supervisor": [
        "kmz_analyzer", "description_filler", "report_generator",
        "anti_reprova", "pipeline_supervisor", "utm_converter",
        "adherence_tester", "rpa_screen_learner",
    ],
}


async def start_pipeline(pipeline_id: str):
    """Background entrypoint. Loads pipeline, runs DAG to completion."""
    logger.info(f"[pipeline] start {pipeline_id}")
    async with session_scope() as db:
        pipe = (await db.execute(
            select(PipelineRun).where(PipelineRun.id == pipeline_id)
        )).scalar_one_or_none()
        if not pipe:
            logger.error(f"[pipeline] {pipeline_id} not found")
            return
        logger.info(f"[pipeline] {pipeline_id} work='{pipe.work_name}' tenant={pipe.tenant_id}")
        pipe.status = PipelineStatus.RUNNING
        pipe.started_at = datetime.now(timezone.utc)
        # Pre-create all AgentRun rows in PENDING so the timeline UI sees them immediately
        for agent_code in PIPELINE_DAG.keys():
            db.add(AgentRun(
                tenant_id=pipe.tenant_id,
                user_id=pipe.user_id,
                agent_code=agent_code,
                status=AgentRunStatus.PENDING,
                pipeline_id=pipe.id,
                input_payload={},
            ))

    try:
        await _execute_dag(pipeline_id)
        logger.info(f"[pipeline] {pipeline_id} DAG complete")
    except Exception as e:
        logger.exception(f"[pipeline] {pipeline_id} orchestrator failed: {e}")
        async with session_scope() as db:
            pipe = (await db.execute(
                select(PipelineRun).where(PipelineRun.id == pipeline_id)
            )).scalar_one_or_none()
            if pipe:
                pipe.status = PipelineStatus.FAILED
                pipe.error_message = f"orchestrator: {type(e).__name__}: {e}"
                pipe.finished_at = datetime.now(timezone.utc)


async def _execute_dag(pipeline_id: str):
    completed_outputs: dict[str, dict] = {}
    completed_runs: dict[str, dict] = {}
    pending = set(PIPELINE_DAG.keys())

    while pending:
        # Find agents whose dependencies are all completed
        ready = [c for c in pending if all(d in completed_outputs for d in PIPELINE_DAG[c])]
        if not ready:
            logger.warning(f"[pipeline] {pipeline_id} no ready agents, stuck. pending={pending}")
            break
        logger.info(f"[pipeline] {pipeline_id} batch ready: {ready}")
        # Run all ready agents in parallel
        results = await asyncio.gather(*[
            _run_one(pipeline_id, code, completed_outputs, completed_runs)
            for code in ready
        ], return_exceptions=True)
        for code, res in zip(ready, results):
            if isinstance(res, Exception):
                completed_outputs[code] = {}
                completed_runs[code] = {"agent_code": code, "status": "failed",
                                        "confidence_score": 0.0,
                                        "output_payload": {}, "error": str(res)}
            else:
                completed_outputs[code] = res["output"]
                completed_runs[code] = res
            pending.discard(code)

    # Aggregate pipeline summary
    async with session_scope() as db:
        pipe = (await db.execute(
            select(PipelineRun).where(PipelineRun.id == pipeline_id)
        )).scalar_one_or_none()
        if not pipe:
            return
        master = completed_outputs.get("master_supervisor", {})
        pipe.summary_output = {
            "by_agent": {c: o for c, o in completed_outputs.items()},
            "executive_summary": master.get("executive_summary", ""),
        }
        pipe.overall_score = float(master.get("overall_score", 0.0))
        any_failed = any(r.get("status") == "failed" for r in completed_runs.values())
        any_human = any(r.get("status") == "needs_human" for r in completed_runs.values())
        if any_failed:
            pipe.status = PipelineStatus.FAILED
        elif any_human or not master.get("ready_to_send", False):
            pipe.status = PipelineStatus.NEEDS_HUMAN
        else:
            pipe.status = PipelineStatus.COMPLETED
        pipe.finished_at = datetime.now(timezone.utc)


async def _run_one(pipeline_id: str, agent_code: str,
                   completed_outputs: dict, completed_runs: dict) -> dict:
    """Execute a single agent within the pipeline. Updates its AgentRun row."""
    upstream_codes = PIPELINE_DAG[agent_code]

    # Build input payload by combining upstream outputs
    input_payload: dict = {}
    if agent_code == "kmz_analyzer":
        # Special: gets kmz_path + obra metadata from pipeline input
        async with session_scope() as db:
            pipe = (await db.execute(
                select(PipelineRun).where(PipelineRun.id == pipeline_id)
            )).scalar_one()
            input_payload = {
                "kmz_path": pipe.input_payload.get("kmz_path"),
                "nota": pipe.work_name,
                "municipio": pipe.input_payload.get("municipio", ""),
                "parceira": pipe.concessionaria,
                "tipo": pipe.tipo,
            }
    elif agent_code in ("utm_converter", "adherence_tester"):
        # Parallel agents: take kmz_analyzer output if available, otherwise kmz_path
        if "kmz_analyzer" in completed_outputs:
            input_payload = completed_outputs["kmz_analyzer"]
        else:
            input_payload = {}
    elif agent_code in ("pipeline_supervisor", "master_supervisor"):
        # Supervisors take a list of upstream run results
        input_payload = {
            "upstream_runs": [completed_runs[c] for c in upstream_codes if c in completed_runs],
        }
    elif len(upstream_codes) == 1:
        input_payload = completed_outputs.get(upstream_codes[0], {})
    else:
        input_payload = {c: completed_outputs.get(c, {}) for c in upstream_codes}

    # Mark RUNNING
    async with session_scope() as db:
        run = (await db.execute(
            select(AgentRun).where(
                AgentRun.pipeline_id == pipeline_id,
                AgentRun.agent_code == agent_code,
            )
        )).scalar_one()
        run.status = AgentRunStatus.RUNNING
        run.started_at = datetime.now(timezone.utc)
        run.input_payload = _trim(input_payload)

    # Fetch few-shot examples for this agent
    async with session_scope() as db:
        pipe = (await db.execute(
            select(PipelineRun).where(PipelineRun.id == pipeline_id)
        )).scalar_one()
        few_shot = await _fetch_few_shot(db, pipe.tenant_id, agent_code)
    logger.info(f"[pipeline] {pipeline_id} agent={agent_code} starting (few_shot={len(few_shot)})")

    # Execute
    try:
        agent = get_agent(agent_code)
        await agent.validate_input(input_payload)
        result = await agent.run(input_payload, context={
            "pipeline_id": pipeline_id,
            "tenant_id": pipe.tenant_id,
            "few_shot_examples": few_shot,
        })
        logger.info(f"[pipeline] {pipeline_id} agent={agent_code} done "
                    f"conf={result.confidence:.2f} needs_human={result.needs_human}")
        async with session_scope() as db:
            run = (await db.execute(
                select(AgentRun).where(
                    AgentRun.pipeline_id == pipeline_id,
                    AgentRun.agent_code == agent_code,
                )
            )).scalar_one()
            run.output_payload = result.output
            run.confidence_score = result.confidence
            run.error_message = result.notes
            run.finished_at = datetime.now(timezone.utc)
            if result.needs_human:
                run.status = AgentRunStatus.NEEDS_HUMAN
            else:
                run.status = AgentRunStatus.COMPLETED
            for case in result.learning_cases:
                db.add(LearningCase(
                    tenant_id=run.tenant_id,
                    agent_code=run.agent_code,
                    source_run_id=run.id,
                    input_payload=case.get("input", {}),
                    observed_output=case.get("output", {}),
                    feedback_score=0.5,
                ))
            return {
                "agent_code": agent_code,
                "status": run.status.value,
                "confidence_score": run.confidence_score,
                "output_payload": result.output,
                "output": result.output,
            }
    except Exception as e:
        logger.exception(f"[pipeline] {pipeline_id} agent={agent_code} FAILED: {e}")
        async with session_scope() as db:
            run = (await db.execute(
                select(AgentRun).where(
                    AgentRun.pipeline_id == pipeline_id,
                    AgentRun.agent_code == agent_code,
                )
            )).scalar_one()
            run.status = AgentRunStatus.FAILED
            run.error_message = f"{type(e).__name__}: {e}"
            run.finished_at = datetime.now(timezone.utc)
        return {"agent_code": agent_code, "status": "failed", "confidence_score": 0.0,
                "output_payload": {}, "output": {}, "error": str(e)}


def _trim(d: dict, max_len: int = 8000) -> dict:
    """Avoid storing massive nested objects in input_payload (e.g., full structures dump)."""
    import json
    s = json.dumps(d, default=str)
    if len(s) <= max_len:
        return d
    return {"_truncated": True, "_keys": list(d.keys())[:20]}


async def enqueue_pipeline(pipeline_id: str):
    await queue.enqueue(start_pipeline, pipeline_id)
