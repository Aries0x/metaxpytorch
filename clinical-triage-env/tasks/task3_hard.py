"""
Task 3 — Triage Nurse (Hard)

The AI agent receives 5 patient reports and must rank them by medical
urgency so the single available doctor sees the most critical patient first.
"""

TASK_ID = "task3"
TASK_NAME = "triage-nurse"
DIFFICULTY = "hard"
DESCRIPTION = "Rank 5 patients by medical urgency for doctor prioritization"
MAX_STEPS = 12
EXPECTED_BASELINE_SCORE = 0.40

SYSTEM_PROMPT = (
    "You are a triage nurse with 5 patient reports.\n"
    "Rank patients from most urgent (1) to least urgent (5).\n"
    "Consider: critical values, number of abnormalities, severity, "
    "patient age.\n\n"
    "Respond in JSON:\n"
    '{\"urgency_ranking\": [\"P001\", \"P003\", \"P002\", \"P005\", \"P004\"],\n'
    ' \"justification\": \"Patient P001 has critical kidney failure...\"}'
)
