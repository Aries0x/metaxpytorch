"""
ClinicalTriageEnv — main environment class implementing the OpenEnv spec.

Provides async reset / step / state methods and manages the Real-Time RL episode.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Union

from environment.simulation import SimulationEngine, PatientSimulator
from environment.data_generator import PATTERNS
from environment.graders import grade_task1, grade_task2, grade_task3
from environment.models import (
    Action,
    EpisodeRecord,
    FullState,
    Observation,
    Reward,
    StepResult,
    Task1Action,
    Task2Action,
    Task3Action,
    ResourceState,
    PatientState
)

# ── Constants ────────────────────────────────────────────────────────────────

MAX_STEPS: Dict[str, int] = {
    "task1": 6,
    "task2": 8,
    "task3": 12,
}

TASK_NAMES: Dict[str, str] = {
    "task1": "lab-technician",
    "task2": "junior-doctor",
    "task3": "triage-nurse",
}

TASK_DESCRIPTIONS: Dict[str, str] = {
    "task1": "Investigate and flag abnormal values in a single patient report by ordering specific tests step by step.",
    "task2": "Identify disease patterns, order necessary tests, and treat the patient before they deteriorate.",
    "task3": "Manage an active ED queue, order stat tests, assign doctors, and prevent critical crashes as new patients arrive.",
}

INSTRUCTIONS: Dict[str, str] = {
    "task1": (
        "You are a lab technician. Progressively order up to 3 tests per step to reveal results. "
        "Flag tests as NORMAL, LOW, HIGH, CRITICAL_LOW, CRITICAL_HIGH, or MISSING. "
        "Respond in JSON: {\"order_tests\": [\"Hemoglobin\"], \"flagged_tests\": {\"Hemoglobin\": \"LOW\"}}"
    ),
    "task2": (
        "You are a junior doctor. Order tests to confirm your diagnosis, then start treatment. "
        "If you don't treat a critical condition, the patient will deteriorate. "
        "Respond in JSON: {\"order_tests\": [\"SGPT (ALT)\"], \"identified_patterns\": [\"Liver Dysfunction\"], \"severity\": \"MODERATE\", \"start_treatment\": true}"
    ),
    "task3": (
        "You are the triage nurse. Watch for worsening vitals and new arrivals. "
        "You have a test budget and limited doctor slots. Assign doctors to critical patients to stabilize them. "
        "Respond in JSON: {\"assign_doctor\": \"P002\", \"order_stat_test\": {\"P001\": [\"Creatinine\"]}, \"urgency_ranking\": [\"P002\", \"P001\", \"P003\"]}"
    ),
}


class ClinicalTriageEnv:
    """OpenEnv-compliant RL environment for clinical triage tasks."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self._seed = seed
        self._engine = SimulationEngine(seed=seed)
        if seed is not None:
            self._rng = random.Random(seed + 1)
        else:
            self._rng = random.Random()

        # Episode state
        self._current_task: str = "task1"
        self._current_step: int = 0
        self._done: bool = True
        self._sim_time_mins: int = 0
        
        self._patients: List[PatientState] = []
        self._simulators: Dict[str, PatientSimulator] = {}
        
        self._resources = ResourceState(doctors_available=2, doctors_total=2, test_budget=10)
        self._events: List[str] = []
        
        self._episode_history: List[EpisodeRecord] = []
        self._scores: List[float] = []
        
        self._next_patient_id: int = 4

    # ── Public API ────────────────────────────────────────────────────────

    async def reset(self, task_id: str = "task1") -> Observation:
        """Start a fresh episode for the given task."""
        if task_id not in MAX_STEPS:
            raise ValueError(f"Unknown task_id: {task_id!r}")

        # Completely clear state
        self._current_task = task_id
        self._current_step = 0
        self._sim_time_mins = 0
        self._done = False
        self._episode_history = []
        self._scores = []
        self._events = []
        self._simulators = {}

        # Fresh random generator
        if self._seed is not None:
            new_seed = self._seed + hash(task_id) + self._rng.randint(0, 10000)
            self._engine = SimulationEngine(seed=new_seed)
            self._rng = random.Random(new_seed + 1)
        else:
            self._engine = SimulationEngine()
            self._rng = random.Random()

        # Generate data
        if task_id == "task3":
            self._patients = self._engine.generate_task3_patients()
            self._next_patient_id = 4
            self._resources = ResourceState(doctors_available=2, doctors_total=2, test_budget=15)
        elif task_id == "task2":
            pattern = self._rng.choice(list(PATTERNS.keys()))
            severity = self._rng.choice(["MILD", "MODERATE", "SEVERE"])
            self._patients = [self._engine.generate_initial_patient("task2", pattern, severity)]
            self._resources = ResourceState(doctors_available=1, doctors_total=1, test_budget=8)
        else:
            pattern = self._rng.choice(list(PATTERNS.keys()))
            severity = self._rng.choice(["MILD", "MODERATE", "SEVERE"])
            self._patients = [self._engine.generate_initial_patient("task1", pattern, severity)]
            self._resources = ResourceState(doctors_available=0, doctors_total=0, test_budget=10)
            
        for p in self._patients:
            self._simulators[p.patient_id] = PatientSimulator(p, self._rng)

        return self._build_observation()

    async def step(self, action: Action) -> StepResult:
        """Process one agent action, advance physics, and grade."""
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new one.")

        self._current_step += 1
        self._events.clear()
        
        # 1. Apply Action effects
        self._process_action(action)
        
        # 2. Advance time and physics (15 mins per step)
        self._sim_time_mins += 15
        for sim in self._simulators.values():
            new_events = sim.tick(15)
            self._events.extend(new_events)
            
        # Task 3 specific physics: patient arrivals
        if self._current_task == "task3" and self._rng.random() < 0.4:
            new_patient = self._engine.generate_new_arrival(self._next_patient_id)
            self._next_patient_id += 1
            self._patients.append(new_patient)
            self._simulators[new_patient.patient_id] = PatientSimulator(new_patient, self._rng)
            self._events.append(f"ALERT: New patient {new_patient.patient_id} arrived at triage.")

        # 3. Grade
        reward = self._grade(action)

        # Record history
        action_dict = action.model_dump() if hasattr(action, "model_dump") else {}
        self._episode_history.append(EpisodeRecord(action=action_dict, reward=reward.score))
        self._scores.append(reward.score)

        # 4. Check termination
        if self._current_step >= MAX_STEPS[self._current_task]:
            self._done = True
            
        # Early termination conditions
        if self._current_task == "task3":
            # If all stabilized and nobody waiting
            if all(p.status == "STABILIZED" for p in self._patients):
                self._done = True
        elif self._current_task == "task2":
            if self._patients[0].status == "STABILIZED" and hasattr(action, "identified_patterns") and action.identified_patterns:
                self._done = True
        elif self._current_task == "task1":
            # Just goes to max steps unless they use all budget
            if self._resources.test_budget <= 0:
                self._done = True

        return StepResult(
            observation=self._build_observation(),
            reward=reward.score,
            done=self._done,
            info={
                "feedback": reward.feedback,
                "breakdown": reward.breakdown,
                "partial_credits": reward.partial_credits,
                "step": self._current_step,
            },
        )

    def _process_action(self, action: Action):
        """Map action commands to state changes."""
        # Handle Task 1 & 2 test ordering
        if hasattr(action, 'order_tests') and action.order_tests:
            patient = self._patients[0]
            ordered = 0
            for t in action.order_tests:
                if self._resources.test_budget > 0 and t in patient.true_tests and t not in patient.revealed_tests:
                    patient.revealed_tests[t] = patient.true_tests[t]
                    self._resources.test_budget -= 1
                    ordered += 1
            if ordered > 0:
                self._events.append(f"Lab results returned for {ordered} tests.")

        # Handle Task 2 treatment
        if getattr(action, 'start_treatment', False):
            patient = self._patients[0]
            if not patient.doctor_assigned:
                patient.doctor_assigned = True
                patient.status = "BEING_TREATED"
                self._events.append(f"Treatment started for {patient.patient_id}. Vitals stabilizing.")

        # Handle Task 3 stat test ordering
        if getattr(action, 'order_stat_test', None):
            for pid, tests in action.order_stat_test.items():
                patient = next((p for p in self._patients if p.patient_id == pid), None)
                if patient:
                    ordered = 0
                    for t in tests:
                        if self._resources.test_budget > 0 and t in patient.true_tests and t not in patient.revealed_tests:
                            patient.revealed_tests[t] = patient.true_tests[t]
                            self._resources.test_budget -= 1
                            ordered += 1
                    if ordered > 0:
                        self._events.append(f"Stat labs for {pid} returned {ordered} results.")

        # Handle Task 3 doctor assignment
        if getattr(action, 'assign_doctor', None):
            pid = action.assign_doctor
            patient = next((p for p in self._patients if p.patient_id == pid), None)
            if patient and not patient.doctor_assigned and self._resources.doctors_available > 0:
                patient.doctor_assigned = True
                patient.status = "BEING_TREATED"
                self._resources.doctors_available -= 1
                self._events.append(f"Doctor assigned to Patient {pid}. Vitals will begin to stabilize.")

    async def state(self) -> FullState:
        """Return a full snapshot of the current environment state."""
        return FullState(
            current_task=self._current_task,
            current_step=self._current_step,
            max_steps=MAX_STEPS[self._current_task],
            done=self._done,
            patients=self._patients,
            episode_history=self._episode_history,
            scores=self._scores,
            sim_time_mins=self._sim_time_mins
        )

    def _build_observation(self) -> Observation:
        return Observation(
            task_id=self._current_task,
            task_name=TASK_NAMES[self._current_task],
            description=TASK_DESCRIPTIONS[self._current_task],
            step_number=self._current_step,
            max_steps=MAX_STEPS[self._current_task],
            instructions=INSTRUCTIONS[self._current_task],
            sim_time_mins=self._sim_time_mins,
            patients=self._patients,
            resources=self._resources,
            events=self._events,
            patient_reports=[] # Deprecated
        )

    def _grade(self, action: Action) -> Reward:
        """Dispatch to the appropriate task grader using simulation state."""
        if self._current_task == "task1":
            if getattr(action, 'flagged_tests', None) is None:
                return Reward(score=0.0, feedback="Invalid action type for task1.")
            return grade_task1(self._patients[0], action, self._resources)

        elif self._current_task == "task2":
            return grade_task2(self._patients[0], action, self._current_step)

        elif self._current_task == "task3":
            return grade_task3(self._patients, action, self._current_step)

        return Reward(score=0.0, feedback=f"Unknown task: {self._current_task}")
