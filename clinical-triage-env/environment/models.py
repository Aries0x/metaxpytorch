"""
Pydantic v2 models for ClinicalTriageEnv.

All models are fully typed — no ``Any`` types.
Covers observations, actions (union across 3 tasks), and reward/state.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Core lab-report primitives
# ---------------------------------------------------------------------------

class TestResult(BaseModel):
    """A single test result as it appears on a messy lab report."""

    test_name: str = Field(..., description="Canonical or variant test name")
    value: Optional[float] = Field(
        default=None, description="Parsed numeric value; None when missing"
    )
    raw_value: str = Field(
        ..., description="Original string from the report (e.g. '--', 'N/A', '  120  ')"
    )
    unit: str = Field(..., description="Unit string (may contain typos)")
    normal_min: float = Field(..., description="Lower bound of normal range")
    normal_max: float = Field(..., description="Upper bound of normal range")
    status: Literal[
        "NORMAL", "LOW", "HIGH", "CRITICAL_LOW", "CRITICAL_HIGH", "MISSING"
    ] = Field(..., description="Ground-truth classification of this value")


class VitalSigns(BaseModel):
    """Evolving vital signs for a patient."""
    heart_rate: float
    bp_systolic: float
    bp_diastolic: float
    spo2: float
    temperature: float
    pain_scale: int


class PatientState(BaseModel):
    """Real-time patient state including changing vitals and revealed data."""
    patient_id: str
    patient_name: str
    age: int
    gender: Literal["Male", "Female", "Other"]
    presenting_complaint: str
    status: Literal["WAITING", "BEING_ASSESSED", "BEING_TREATED", "STABILIZED", "CRITICAL_EVENT"]
    time_waiting_mins: int
    acuity: int = Field(..., description="Ground truth acuity 1-5")
    
    # Real-time data
    vitals: VitalSigns
    
    # Test information (only revealed tests)
    revealed_tests: Dict[str, TestResult]
    available_tests: List[str]
    
    # Flags / actions taken
    doctor_assigned: bool
    underlying_condition: str = Field(..., exclude=True) # Exclude from API observation, used by sim
    true_tests: Dict[str, TestResult] = Field(default_factory=dict, exclude=True)


# For backward compatibility with tests/grader expectations (used if static)
class PatientReport(BaseModel):
    patient_id: str
    patient_name: str
    age: int
    gender: Literal["Male", "Female", "Other"]
    symptoms: str
    tests: Dict[str, TestResult]
    report_date: str
    center_name: str


class ResourceState(BaseModel):
    doctors_available: int
    doctors_total: int
    test_budget: int # How many more tests can be ordered
    

class Observation(BaseModel):
    """Observation returned after reset / each step."""
    task_id: str
    task_name: str
    description: str
    step_number: int
    max_steps: int
    instructions: str
    sim_time_mins: int
    
    patients: List[PatientState]
    resources: ResourceState
    events: List[str] = Field(default_factory=list)
    
    # Legacy field
    patient_reports: List[PatientReport] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Actions — what the AI agent submits (RL Style)
# ---------------------------------------------------------------------------

class Task1Action(BaseModel):
    """Lab-technician: order tests and flag results."""
    order_tests: List[str] = Field(default_factory=list)
    flagged_tests: Dict[str, str] = Field(default_factory=dict)


class Task2Action(BaseModel):
    """Junior-doctor: order tests, diagnose, treat."""
    order_tests: List[str] = Field(default_factory=list)
    identified_patterns: List[str] = Field(default_factory=list)
    severity: Optional[str] = None
    start_treatment: bool = False


class Task3Action(BaseModel):
    """Triage-nurse patient ranking & assignments."""
    assign_doctor: Optional[str] = None    # patient_id
    order_stat_test: Optional[Dict[str, List[str]]] = None # patient_id -> list of tests
    update_acuity: Optional[Dict[str, int]] = None
    urgency_ranking: List[str] = Field(default_factory=list) # Reorder queue
    justification: str = ""


# A single Action type that covers all tasks
Action = Union[Task1Action, Task2Action, Task3Action]


# ---------------------------------------------------------------------------
# Reward / scoring
# ---------------------------------------------------------------------------

class Reward(BaseModel):
    """Grading result returned to the agent."""
    score: float = Field(..., ge=0.0, le=1.0)
    breakdown: Dict[str, float] = Field(default_factory=dict)
    feedback: str = Field(default="")
    partial_credits: Dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Step result & full state
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    """Returned by ``env.step()``."""
    observation: Observation
    reward: float = Field(..., ge=0.0, le=1.0)
    done: bool
    info: Dict[str, Union[str, float, bool, List[str], Dict[str, float]]] = Field(
        default_factory=dict
    )


class EpisodeRecord(BaseModel):
    """One (action, reward) pair kept in history."""
    action: Dict[str, Union[str, float, List[str], Dict[str, str], Dict[str, List[str]], Dict[str, int], None]]
    reward: float


class FullState(BaseModel):
    """Snapshot of the environment's complete state."""
    current_task: str
    current_step: int
    max_steps: int
    done: bool
    patients: List[PatientState]
    episode_history: List[EpisodeRecord]
    scores: List[float]
    sim_time_mins: int
