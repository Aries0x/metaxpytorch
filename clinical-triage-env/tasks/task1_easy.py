"""
Task 1 — Lab Technician (Easy)

The AI agent examines a single messy patient lab report and flags each test
value as NORMAL, LOW, HIGH, CRITICAL_LOW, CRITICAL_HIGH, or MISSING.
"""

TASK_ID = "task1"
TASK_NAME = "lab-technician"
DIFFICULTY = "easy"
DESCRIPTION = "Flag abnormal values in a messy single patient lab report"
MAX_STEPS = 6
EXPECTED_BASELINE_SCORE = 0.75

SYSTEM_PROMPT = (
    "You are a lab technician at an Indian diagnostic center.\n"
    "You will receive a patient lab report with various test values.\n"
    "Some values may be missing (shown as '--' or 'N/A').\n"
    "Your job is to classify each test as: NORMAL, LOW, HIGH, "
    "CRITICAL_LOW, CRITICAL_HIGH, or MISSING.\n\n"
    "Respond in JSON format:\n"
    '{\"flagged_tests\": {\"test_name\": \"STATUS\", ...}}\n\n'
    "Use the test names exactly as they appear in the report.\n"
    "Be precise. Missing values should be flagged as MISSING."
)
