#!/usr/bin/env python3
"""
inference.py — Baseline agent for ClinicalTriageEnv RL Variant.

Runs all three tasks against the FastAPI environment using Qwen2.5-72B
via the HuggingFace router.  Emits the mandatory [START] / [STEP] / [END]
log lines exactly as the OpenEnv evaluator expects.
"""

from __future__ import annotations

import json
import os
import re
import sys
import traceback
from typing import Dict, List, Optional

import httpx
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ── Configuration ────────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")

SUCCESS_THRESHOLD = 0.5

TASK_CONFIG = {
    "task1": {"max_steps": 6, "name": "lab-technician"},
    "task2": {"max_steps": 8, "name": "junior-doctor"},
    "task3": {"max_steps": 12, "name": "triage-nurse"},
}

# ── System prompts per task ──────────────────────────────────────────────────

SYSTEM_PROMPTS = {
    "task1": (
        "You are a lab technician. You have a budget for ordering tests.\n"
        "Order tests to reveal their initial results, then flag them.\n"
        "Tests must be flagged as NORMAL, LOW, HIGH, CRITICAL_LOW, CRITICAL_HIGH, or MISSING based on standard ranges.\n"
        "You can order up to 3 tests per step.\n"
        'Respond ONLY in JSON format:\n'
        '{"order_tests": ["Hemoglobin", "RBC Count"], "flagged_tests": {"Hemoglobin": "LOW"}}\n'
    ),
    "task2": (
        "You are a junior doctor. Order tests to confirm a diagnosis, then start treatment.\n"
        "Known patterns: Anemia, Diabetes, Liver Dysfunction, Kidney Impairment, Hypothyroidism.\n"
        "If you do not treat the patient, they will deteriorate.\n"
        "If you have enough information to diagnose, provide identified_patterns and severity, and set start_treatment to True.\n"
        'Respond ONLY in JSON:\n'
        '{"order_tests": ["SGPT (ALT)"], "identified_patterns": ["Liver Dysfunction"], "severity": "MODERATE", "start_treatment": true}'
    ),
    "task3": (
        "You are the triage nurse managing the ED. Consider vitals and deterioration.\n"
        "New patients will occasionally arrive. You have limited doctor slots and a test budget.\n"
        "Order stat tests to assess condition. Assign available doctors to the most critical patients to stabilize them.\n"
        "Patients that are unstable will crash if not treated.\n"
        'Respond ONLY in JSON:\n'
        '{"assign_doctor": "P001", "order_stat_test": {"P001": ["Creatinine"]}, "urgency_ranking": ["P001", "P002", "P003"]}'
    ),
}

# ── Helpers ──────────────────────────────────────────────────────────────────

def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))

def _format_vitals(v: dict) -> str:
    return f"HR: {v['heart_rate']:.0f} | BP: {v['bp_systolic']:.0f}/{v['bp_diastolic']:.0f} | SpO2: {v['spo2']:.1f}% | Temp: {v['temperature']:.1f} | Pain: {v['pain_scale']}/10"

def _format_patient_for_prompt(patient: dict) -> str:
    lines = [
        f"Patient ID: {patient['patient_id']}",
        f"Name: {patient['patient_name']} ({patient['age']} {patient['gender']})",
        f"Complaint: {patient['presenting_complaint']}",
        f"Status: {patient['status']} (Waiting: {patient['time_waiting_mins']}m, Doctor Assigned: {patient['doctor_assigned']})",
        f"Vitals: {_format_vitals(patient['vitals'])}",
        "Revealed Test Results:"
    ]
    if not patient.get('revealed_tests'):
        lines.append("  (None ordered yet)")
    else:
        for tname, tinfo in patient['revealed_tests'].items():
            lines.append(f"  - {tname}: {tinfo['raw_value']} {tinfo['unit']} (Range: {tinfo['normal_min']}-{tinfo['normal_max']})")
            
    lines.append("Available Tests that can be ordered: " + ", ".join(patient['available_tests']))
    return "\n".join(lines)


def _build_user_prompt(observation: dict) -> str:
    task_id = observation.get("task_id", "task1")
    patients = observation.get("patients", [])
    resources = observation.get("resources", {})
    events = observation.get("events", [])
    step = observation.get("step_number", 0)
    max_steps = observation.get("max_steps", 3)
    sim_time = observation.get("sim_time_mins", 0)

    parts = [
        f"Step {step}/{max_steps} - Elapsed Sim Time: {sim_time} mins",
        f"Resources: Doctors: {resources.get('doctors_available')}/{resources.get('doctors_total')} | Test Budget: {resources.get('test_budget')}",
        ""
    ]
    
    if events:
        parts.append("Recent Events:")
        for ev in events:
            parts.append(f" - {ev}")
        parts.append("")

    parts.append("--- Patient Statuses ---")
    for r in patients:
        parts.append(_format_patient_for_prompt(r))
        parts.append("")

    return "\n".join(parts)


def _extract_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {}

def _default_action(task_id: str, observation: dict) -> dict:
    if task_id == "task1":
        tests = {}
        for p in observation.get("patients", []):
            for t in p.get("revealed_tests", {}):
                tests[t] = "NORMAL"
        return {"order_tests": [], "flagged_tests": tests}
    elif task_id == "task2":
        return {"order_tests": [], "identified_patterns": [], "start_treatment": False}
    elif task_id == "task3":
        return {"urgency_ranking": [p["patient_id"] for p in observation.get("patients", [])], "assign_doctor": None}
    return {}

# ── Environment client ───────────────────────────────────────────────────────

class EnvClient:
    def __init__(self, base_url: str) -> None:
        self._url = base_url.rstrip("/")
        self._http = httpx.Client(timeout=60.0)

    def reset(self, task_id: str) -> dict:
        r = self._http.post(f"{self._url}/reset", json={"task_id": task_id})
        r.raise_for_status()
        return r.json()

    def step(self, action: dict) -> dict:
        r = self._http.post(f"{self._url}/step", json=action)
        r.raise_for_status()
        return r.json()

# ── Main runner ──────────────────────────────────────────────────────────────

def run_task(task_id: str, llm: OpenAI, env_client: EnvClient) -> float:
    task_name = TASK_CONFIG[task_id]["name"]
    max_steps = TASK_CONFIG[task_id]["max_steps"]

    print(f"[START] task={task_name} env=clinical-triage model={MODEL_NAME}")

    rewards: List[float] = []
    final_score = 0.0
    success = False
    steps_taken = 0

    try:
        observation = env_client.reset(task_id)
        done = False

        for step_n in range(1, max_steps + 1):
            if done:
                break

            steps_taken = step_n
            error_msg: Optional[str] = None
            user_prompt = _build_user_prompt(observation)
            system_prompt = SYSTEM_PROMPTS[task_id]

            action_dict: dict = {}
            try:
                response = llm.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.2,
                    max_tokens=2048,
                )
                raw_text = response.choices[0].message.content or ""
                action_dict = _extract_json(raw_text)

                if not action_dict:
                    response = llm.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt + "\n\nRemember: respond ONLY with valid JSON."},
                        ],
                        temperature=0.1,
                        max_tokens=2048,
                    )
                    raw_text = response.choices[0].message.content or ""
                    action_dict = _extract_json(raw_text)
            except Exception as e:
                error_msg = str(e)

            if not action_dict:
                action_dict = _default_action(task_id, observation)
                if error_msg is None:
                    error_msg = "JSON parse failed, using default action"

            try:
                step_result = env_client.step(action_dict)
                reward = _clamp(step_result.get("reward", 0.0))
                done = step_result.get("done", False)
                observation = step_result.get("observation", observation)
            except Exception as e:
                reward = 0.0
                done = True
                error_msg = str(e)

            rewards.append(reward)

            action_str = json.dumps(action_dict, separators=(",", ":"))
            if len(action_str) > 200:
                action_str = action_str[:197] + "..."

            error_field = error_msg if error_msg else "null"
            print(f"[STEP] step={step_n} action={action_str} reward={reward:.2f} done={done} error={error_field}")

        final_score = _clamp(max(rewards)) if rewards else 0.0
        success = final_score >= SUCCESS_THRESHOLD

    except Exception as exc:
        error_str = str(exc)
        print(f"[STEP] step={steps_taken} action=error reward=0.00 done=true error={error_str}")

    reward_strs = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
    print(f"[END] success={str(success).lower()} steps={steps_taken} score={final_score:.2f} rewards={reward_strs}")
    return final_score


def main() -> None:
    if not HF_TOKEN:
        print("Warning: HF_TOKEN not found in environment. Inference to HuggingFace API will fail.")

    llm = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "dummy_token")
    env_client = EnvClient(ENV_URL)

    scores: List[float] = []
    for task_id in ("task1", "task2", "task3"):
        score = run_task(task_id, llm, env_client)
        scores.append(score)

    avg = sum(scores) / len(scores) if scores else 0.0
    print(f"\n=== Overall average score: {avg:.2f} ===")

if __name__ == "__main__":
    main()
