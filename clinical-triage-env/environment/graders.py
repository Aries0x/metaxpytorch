"""
Grading logic for ClinicalTriageEnv RL variant.
Scores are outcome-based, focusing on real-time decisions, resource usage, 
and whether patient vitals stabilize or crash.
"""

from __future__ import annotations

from typing import Dict, List, Optional
from scipy.stats import kendalltau

from environment.models import (
    PatientState,
    Reward,
    ResourceState,
    Task1Action,
    Task2Action,
    Task3Action,
)
from environment.data_generator import PATTERNS

def _clamp(v: float) -> float:
    return max(0.001, min(0.999, v))

# ═════════════════════════════════════════════════════════════════════════════
# TASK 1 — Lab Technician: order tests progressively and flag accurately
# ═════════════════════════════════════════════════════════════════════════════

def grade_task1(patient: PatientState, ai_action: Task1Action, resources: ResourceState) -> Reward:
    """
    Score how well the AI flagged tests, with an efficiency bonus for tests ordered.
    Total tests should be fully covered.
    """
    if not hasattr(ai_action, "flagged_tests") or not ai_action.flagged_tests:
        return Reward(score=0.001, breakdown={}, feedback="No flagged tests submitted yet.")

    gt: Dict[str, str] = {name: tr.status for name, tr in patient.true_tests.items()}
    total = len(gt)
    if total == 0:
        return Reward(score=0.001, breakdown={}, feedback="No tests in report.")

    ai_flags: Dict[str, str] = ai_action.flagged_tests

    correct = 0
    false_positives = 0
    missed_critical = 0

    for name, true_status in gt.items():
        ai_status = ai_flags.get(name, None)
        is_critical = true_status.startswith("CRITICAL")

        if ai_status == true_status:
            correct += 1
        elif ai_status is not None:
            if true_status == "NORMAL" and ai_status != "NORMAL":
                false_positives += 1
            if is_critical and not ai_status.startswith("CRITICAL"):
                missed_critical += 1
        else:
            # Not flagged yet.
            if is_critical:
                missed_critical += 1

    base_score = correct / total
    fp_penalty = false_positives * 0.05
    mc_penalty = missed_critical * 0.15
    
    # Efficiency bonus: higher if used fewer tests from budget
    used_tests = 10 - resources.test_budget
    efficiency = max(0, (10 - used_tests) / 10.0) * 0.1  # Up to 0.1 bonus
    
    score = _clamp(base_score + efficiency - fp_penalty - mc_penalty)

    return Reward(
        score=round(score, 4),
        breakdown={
            "base_accuracy": round(base_score, 4),
            "efficiency_bonus": round(efficiency, 4),
            "false_positive_penalty": -round(fp_penalty, 4),
            "missed_critical_penalty": -round(mc_penalty, 4),
        },
        feedback=f"Flagged {correct}/{total} tests. Used {used_tests} test orders.",
    )

# ═════════════════════════════════════════════════════════════════════════════
# TASK 2 — Junior Doctor: diagnosis & treatment outcome
# ═════════════════════════════════════════════════════════════════════════════

def grade_task2(patient: PatientState, ai_action: Task2Action, step: int) -> Reward:
    """
    Score based on diagnosis accuracy, speed, and patient outcome (did vitals crash?).
    """
    gt_pattern = patient.underlying_condition
    
    ai_patterns = ai_action.identified_patterns if hasattr(ai_action, "identified_patterns") else []
    
    diag_score = 0.0
    if gt_pattern in ai_patterns:
        diag_score = 1.0
    elif len(ai_patterns) > 0:
        # Partial match if tests overlap
        g_tests = set(PATTERNS.get(gt_pattern, {}).get("tests", []))
        a_tests = set(PATTERNS.get(ai_patterns[0], {}).get("tests", []))
        if g_tests & a_tests:
            diag_score = 0.5
            
    # Outcome score based on status
    outcome_score = {
        "STABILIZED": 1.0,
        "BEING_TREATED": 0.8,
        "BEING_ASSESSED": 0.4,
        "CRITICAL_EVENT": 0.0,
        "WAITING": 0.2
    }.get(patient.status, 0.0)
    
    # Speed bonus (if treated fast)
    speed_bonus = max(0, (8 - step) / 8.0) * 0.1 if patient.status in ["STABILIZED", "BEING_TREATED"] else 0.0

    score = _clamp((diag_score * 0.5) + (outcome_score * 0.5) + speed_bonus)
    
    return Reward(
        score=round(score, 4),
        breakdown={
            "diagnosis_accuracy": round(diag_score * 0.5, 4),
            "patient_outcome": round(outcome_score * 0.5, 4),
            "speed_bonus": round(speed_bonus, 4),
        },
        feedback=f"Patient is {patient.status}. Diagnosis match: {diag_score}.",
    )

# ═════════════════════════════════════════════════════════════════════════════
# TASK 3 — Triage Nurse: multi-patient ED management
# ═════════════════════════════════════════════════════════════════════════════

def grade_task3(patients: List[PatientState], ai_action: Task3Action, step: int) -> Reward:
    """
    Reward is Average Patient Outcome across the entire ED queue.
    Penalty for critical events (crashes).
    Bonus if the ranking actually correlates with underlying acuity.
    """
    if not patients:
        return Reward(score=0.001)
        
    outcome_scores = {
        "STABILIZED": 1.0,
        "BEING_TREATED": 0.8,
        "BEING_ASSESSED": 0.5,
        "WAITING": 0.4,
        "CRITICAL_EVENT": -0.5
    }
    
    total_outcome = sum(outcome_scores.get(p.status, 0.0) for p in patients)
    avg_outcome = total_outcome / len(patients)
    
    # Optional Kendall-Tau ranking bonus if urgency_ranking is provided
    ranking_bonus = 0.0
    if hasattr(ai_action, "urgency_ranking") and len(ai_action.urgency_ranking) > 1:
        # Ground truth rank by acuity then waiting time
        ranked_true = sorted(patients, key=lambda x: (x.acuity, -x.time_waiting_mins))
        true_ids = [p.patient_id for p in ranked_true]
        
        ai_order = ai_action.urgency_ranking
        if len(set(ai_order) & set(true_ids)) >= 2:
            correct_rank = {pid: i for i, pid in enumerate(true_ids)}
            ai_rank_list = []
            correct_rank_list = []
            for i, pid in enumerate(ai_order):
                if pid in correct_rank:
                    ai_rank_list.append(i)
                    correct_rank_list.append(correct_rank[pid])
            
            if len(ai_rank_list) >= 2:
                tau, _ = kendalltau(correct_rank_list, ai_rank_list)
                if tau == tau: # Not NaN
                    ranking_bonus = ((tau + 1.0) / 2.0) * 0.2 # Max 0.2 bonus
                    
    score = _clamp(avg_outcome + ranking_bonus)
    
    crashes = sum(1 for p in patients if p.status == "CRITICAL_EVENT")
    feedback = f"ED has {len(patients)} patients. Crashes: {crashes}. Avg outcome: {avg_outcome:.2f}."

    return Reward(
        score=round(score, 4),
        breakdown={
            "avg_outcome": round(avg_outcome, 4),
            "ranking_bonus": round(ranking_bonus, 4),
            "crashes": crashes
        },
        feedback=feedback,
    )
