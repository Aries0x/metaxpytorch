"""
FastAPI server exposing the OpenEnv HTTP API for ClinicalTriageEnv.

Endpoints:
  POST /reset   — start a new episode
  POST /step    — submit an action
  GET  /state   — full state snapshot
  GET  /health  — liveness probe
  GET  /tasks   — list available tasks
"""

from __future__ import annotations

import logging
import traceback
from typing import Dict, List, Optional, Union

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from environment.env import ClinicalTriageEnv, TASK_DESCRIPTIONS, TASK_NAMES, MAX_STEPS
from environment.models import (
    Observation,
    Task1Action,
    Task2Action,
    Task3Action,
)

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger("clinical-triage-env")

# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ClinicalTriageEnv",
    description="OpenEnv environment — Indian diagnostic lab triage",
    version="1.0.0",
)

# CORS — allow all origins for HuggingFace Space compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static UI ────────────────────────────────────────────────────────────────
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
async def serve_ui():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "UI not built yet."}

# Global environment instance
env = ClinicalTriageEnv()

# ── Request / response schemas ───────────────────────────────────────────────


class ResetRequest(BaseModel):
    task_id: str = Field(default="task1", description="task1 | task2 | task3")


class StepRequest(BaseModel):
    """Union action payload — the server figures out which task is active."""
    # Shared / Optional fields for RL actions
    order_tests: Optional[List[str]] = None
    
    # Task 1 fields
    flagged_tests: Optional[Dict[str, str]] = None
    
    # Task 2 fields
    identified_patterns: Optional[List[str]] = None
    severity: Optional[str] = None
    start_treatment: Optional[bool] = None
    
    # Task 3 fields
    urgency_ranking: Optional[List[str]] = None
    justification: Optional[str] = None
    assign_doctor: Optional[str] = None
    order_stat_test: Optional[Dict[str, List[str]]] = None
    update_acuity: Optional[Dict[str, int]] = None


class StepResponse(BaseModel):
    observation: dict
    reward: float
    done: bool
    info: dict


class HealthResponse(BaseModel):
    status: str
    env: str
    version: str


class TaskInfo(BaseModel):
    id: str
    name: str
    difficulty: str
    description: str
    max_steps: int


# ── Endpoints ────────────────────────────────────────────────────────────────


@app.post("/reset", response_model=dict)
async def reset_endpoint(body: ResetRequest = ResetRequest()):
    """Start a fresh episode for the specified task."""
    try:
        obs = await env.reset(task_id=body.task_id)
        log.info("Reset → task=%s", body.task_id)
        return obs.model_dump()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        log.error("Reset error: %s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/step", response_model=StepResponse)
async def step_endpoint(body: StepRequest):
    """Submit an action and receive observation + reward."""
    try:
        action = _parse_action(body)
        result = await env.step(action)
        log.info(
            "Step → task=%s step=%d reward=%.4f done=%s",
            env._current_task, env._current_step, result.reward, result.done,
        )
        return StepResponse(
            observation=result.observation.model_dump(),
            reward=result.reward,
            done=result.done,
            info=result.info,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        log.error("Step error: %s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/state", response_model=dict)
async def state_endpoint():
    """Return the full current environment state."""
    try:
        full = await env.state()
        return full.model_dump()
    except Exception as exc:
        log.error("State error: %s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health", response_model=HealthResponse)
async def health_endpoint():
    """Simple liveness / readiness check."""
    return HealthResponse(status="ok", env="ClinicalTriageEnv", version="1.0.0")


@app.get("/tasks", response_model=List[TaskInfo])
async def tasks_endpoint():
    """List all available tasks."""
    difficulties = {"task1": "easy", "task2": "medium", "task3": "hard"}
    return [
        TaskInfo(
            id=tid,
            name=TASK_NAMES[tid],
            difficulty=difficulties[tid],
            description=TASK_DESCRIPTIONS[tid],
            max_steps=MAX_STEPS[tid],
        )
        for tid in ("task1", "task2", "task3")
    ]


# ── Action parser ────────────────────────────────────────────────────────────


def _parse_action(body: StepRequest):
    """Convert the generic StepRequest into the correct typed Action."""
    task = env._current_task

    if task == "task1":
        return Task1Action(
            order_tests=body.order_tests or [],
            flagged_tests=body.flagged_tests or {}
        )

    elif task == "task2":
        return Task2Action(
            order_tests=body.order_tests or [],
            identified_patterns=body.identified_patterns or [],
            severity=body.severity,
            start_treatment=body.start_treatment or False
        )

    elif task == "task3":
        return Task3Action(
            assign_doctor=body.assign_doctor,
            order_stat_test=body.order_stat_test,
            update_acuity=body.update_acuity,
            urgency_ranking=body.urgency_ranking or [],
            justification=body.justification or "",
        )

    raise HTTPException(status_code=400, detail=f"Unknown task: {task}")


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
