"""
Task 2 — Junior Doctor (Medium)

The AI agent analyses a complete diagnostic report and identifies disease
patterns along with severity level.
"""

TASK_ID = "task2"
TASK_NAME = "junior-doctor"
DIFFICULTY = "medium"
DESCRIPTION = "Identify disease patterns from a complete diagnostic report"
MAX_STEPS = 8
EXPECTED_BASELINE_SCORE = 0.55

SYSTEM_PROMPT = (
    "You are a junior doctor reviewing a complete diagnostic report.\n"
    "Identify any disease patterns from the abnormal values you see.\n\n"
    "Known patterns: Anemia, Diabetes, Liver Dysfunction, "
    "Kidney Impairment, Hypothyroidism\n"
    "Rate severity as: MILD, MODERATE, or SEVERE\n\n"
    "Respond in JSON:\n"
    '{\"identified_patterns\": [\"Pattern1\"], \"severity\": \"MODERATE\", '
    '\"reasoning\": \"...\"}'
)
