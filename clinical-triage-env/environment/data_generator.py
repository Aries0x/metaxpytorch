"""
Synthetic messy Indian lab-report generator.

Produces realistic PatientReport objects with configurable mess injection
(missing values, unit typos, test-name variants, borderline values, whitespace
noise, and rare impossible values).  A random seed can be provided for
reproducibility.
"""

from __future__ import annotations

import random
import string
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from environment.models import PatientReport, TestResult

# ── Normal ranges ────────────────────────────────────────────────────────────
# Each entry: (default_min, default_max, unit, gender_specific?)
# Gender-specific ranges are keyed as (male_min, male_max, female_min, female_max)

NORMAL_RANGES: Dict[str, Dict] = {
    "Hemoglobin": {
        "unit": "g/dL",
        "gender_specific": True,
        "Male": (13.0, 17.0),
        "Female": (12.0, 15.0),
    },
    "RBC Count": {"unit": "million/µL", "min": 4.5, "max": 5.5},
    "WBC Count": {"unit": "/µL", "min": 4000, "max": 11000},
    "Platelets": {"unit": "/µL", "min": 150000, "max": 400000},
    "Blood Sugar (Fasting)": {"unit": "mg/dL", "min": 70, "max": 100},
    "HbA1c": {"unit": "%", "min": 4.0, "max": 5.6},
    "Creatinine": {"unit": "mg/dL", "min": 0.7, "max": 1.2},
    "Urea": {"unit": "mg/dL", "min": 7, "max": 20},
    "SGPT (ALT)": {"unit": "U/L", "min": 7, "max": 40},
    "SGOT (AST)": {"unit": "U/L", "min": 10, "max": 40},
    "Bilirubin Total": {"unit": "mg/dL", "min": 0.2, "max": 1.2},
    "TSH": {"unit": "mIU/L", "min": 0.4, "max": 4.0},
    "T3": {"unit": "ng/dL", "min": 80, "max": 200},
    "T4": {"unit": "µg/dL", "min": 5.0, "max": 12.0},
    "Albumin": {"unit": "g/dL", "min": 3.5, "max": 5.0},
    "GFR": {"unit": "mL/min", "min": 90, "max": 120},
    "MCV": {"unit": "fL", "min": 80, "max": 100},
    "MCH": {"unit": "pg", "min": 27, "max": 32},
}

# ── Disease patterns ─────────────────────────────────────────────────────────

PATTERNS: Dict[str, Dict] = {
    "Anemia": {
        "tests": ["Hemoglobin", "RBC Count", "MCV", "MCH"],
        "direction": ["LOW", "LOW", "HIGH", "LOW"],  # B12-deficiency profile
        "severity_thresholds": {
            "MILD": {"Hemoglobin": 10.0},
            "MODERATE": {"Hemoglobin": 8.0},
            "SEVERE": {"Hemoglobin": 6.0},
        },
    },
    "Diabetes": {
        "tests": ["Blood Sugar (Fasting)", "HbA1c"],
        "direction": ["HIGH", "HIGH"],
        "severity_thresholds": {
            "MILD": {"HbA1c": 6.5},
            "MODERATE": {"HbA1c": 8.0},
            "SEVERE": {"HbA1c": 10.0},
        },
    },
    "Liver Dysfunction": {
        "tests": ["SGPT (ALT)", "SGOT (AST)", "Bilirubin Total", "Albumin"],
        "direction": ["HIGH", "HIGH", "HIGH", "LOW"],
        "severity_thresholds": {
            "MILD": {"SGPT (ALT)": 80},
            "MODERATE": {"SGPT (ALT)": 200},
            "SEVERE": {"SGPT (ALT)": 500},
        },
    },
    "Kidney Impairment": {
        "tests": ["Creatinine", "Urea", "GFR"],
        "direction": ["HIGH", "HIGH", "LOW"],
        "severity_thresholds": {
            "MILD": {"Creatinine": 1.5},
            "MODERATE": {"Creatinine": 2.5},
            "SEVERE": {"Creatinine": 4.0},
        },
    },
    "Hypothyroidism": {
        "tests": ["TSH", "T3", "T4"],
        "direction": ["HIGH", "LOW", "LOW"],
        "severity_thresholds": {
            "MILD": {"TSH": 5.0},
            "MODERATE": {"TSH": 10.0},
            "SEVERE": {"TSH": 20.0},
        },
    },
}

# ── Mess-injection data ──────────────────────────────────────────────────────

MISSING_SENTINELS = ["--", "N/A", "not done", "", "pending"]

UNIT_TYPOS: Dict[str, List[str]] = {
    "mg/dL": ["mg/dl", "mg/DL", "MG/DL", "mgdl", "mg per dL"],
    "g/dL": ["g/dl", "gm/dL", "G/DL"],
    "/µL": ["/ul", "/uL", "per uL", "/mcL"],
    "million/µL": ["million/ul", "mill/uL", "x10^6/uL"],
    "mIU/L": ["mIU/l", "uIU/mL", "miu/L"],
    "ng/dL": ["ng/dl", "NG/DL"],
    "µg/dL": ["ug/dL", "mcg/dL", "ug/dl"],
    "U/L": ["u/L", "U/l", "IU/L"],
    "mL/min": ["ml/min", "ML/MIN"],
    "%": [" %", "percent"],
    "fL": ["fl", "FL"],
    "pg": ["PG", "Pg"],
}

TEST_NAME_VARIANTS: Dict[str, List[str]] = {
    "Hemoglobin": ["Haemoglobin", "Hb", "HGB", "hgb", "Hgb"],
    "SGPT (ALT)": ["SGPT", "ALT", "S.G.P.T", "Alanine Aminotransferase"],
    "SGOT (AST)": ["SGOT", "AST", "S.G.O.T", "Aspartate Aminotransferase"],
    "Creatinine": ["Creatinin", "S.Creatinine", "Serum Creatinine"],
    "Blood Sugar (Fasting)": ["BSF", "FBS", "Fasting Sugar", "F.Blood Sugar"],
    "TSH": ["T.S.H", "Thyroid Stimulating Hormone", "TSH (Ultrasensitive)"],
    "Bilirubin Total": ["Total Bilirubin", "S.Bilirubin", "Bilirubin (Total)"],
    "RBC Count": ["RBC", "Red Blood Cells", "R.B.C"],
    "WBC Count": ["WBC", "White Blood Cells", "W.B.C", "TLC"],
    "Platelets": ["Platelet Count", "PLT", "Thrombocytes"],
    "Urea": ["Blood Urea", "BUN", "S.Urea"],
    "HbA1c": ["Glycated Hemoglobin", "HbA1C", "A1C"],
    "T3": ["Triiodothyronine", "Total T3"],
    "T4": ["Thyroxine", "Total T4"],
    "Albumin": ["Serum Albumin", "S.Albumin"],
    "GFR": ["eGFR", "Glomerular Filtration Rate"],
    "MCV": ["Mean Corpuscular Volume"],
    "MCH": ["Mean Corpuscular Hemoglobin"],
}

# ── Indian contextual data ───────────────────────────────────────────────────

INDIAN_NAMES: Dict[str, List[str]] = {
    "Male": [
        "Rajesh Kumar", "Suresh Patel", "Arun Sharma", "Venkat Rao",
        "Mohammed Ali", "Arjun Singh", "Ravi Krishnan", "Deepak Nair",
        "Sanjay Gupta", "Vikram Joshi", "Manoj Tiwari", "Prakash Reddy",
    ],
    "Female": [
        "Priya Sharma", "Lakshmi Devi", "Fatima Begum", "Anjali Gupta",
        "Meena Kumari", "Sunita Patel", "Kavitha Raj", "Anita Reddy",
        "Deepa Nair", "Savitri Devi", "Rashida Bi", "Pooja Verma",
    ],
}

CENTERS = [
    "Sri Ramakrishna Diagnostics, Coimbatore",
    "Thyrocare Collection Center, Nagpur",
    "Dr. Lal PathLabs, Bhopal",
    "SRL Diagnostics, Chennai",
    "Metropolis Healthcare, Pune",
    "Vijaya Diagnostic Centre, Hyderabad",
    "Neuberg Diagnostics, Bangalore",
]

SYMPTOMS = [
    "c/o fatigue and weakness since 3 weeks",
    "complaints of excessive thirst and frequent urination",
    "presenting with jaundice and abdominal pain",
    "referred for routine health checkup",
    "c/o swelling in legs, reduced urine output",
    "presenting with weight gain, cold intolerance",
    "severe chest pain, referred from emergency",
    "c/o breathlessness on exertion",
    "complaints of easy bruising and gum bleeding",
    "routine antenatal checkup",
    "c/o persistent headache and blurred vision",
    "presenting with unexplained weight loss",
]

# ── Critical thresholds used for status classification ────────────────────────

CRITICAL_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "Hemoglobin": {"CRITICAL_LOW": 7.0},
    "Blood Sugar (Fasting)": {"CRITICAL_HIGH": 300.0},
    "Platelets": {"CRITICAL_LOW": 50000.0},
    "Creatinine": {"CRITICAL_HIGH": 5.0},
    "SGPT (ALT)": {"CRITICAL_HIGH": 500.0},
}


# ── Helper: classify a numeric value ─────────────────────────────────────────

def _classify_value(
    test_name: str,
    value: Optional[float],
    normal_min: float,
    normal_max: float,
) -> str:
    """Return the ground-truth status label for a single test value."""
    if value is None:
        return "MISSING"

    # Check critical thresholds first
    crit = CRITICAL_THRESHOLDS.get(test_name, {})
    if "CRITICAL_LOW" in crit and value < crit["CRITICAL_LOW"]:
        return "CRITICAL_LOW"
    if "CRITICAL_HIGH" in crit and value > crit["CRITICAL_HIGH"]:
        return "CRITICAL_HIGH"

    if value < normal_min:
        return "LOW"
    if value > normal_max:
        return "HIGH"
    return "NORMAL"


# ── Main generator class ─────────────────────────────────────────────────────

class DataGenerator:
    """Generates synthetic, messy Indian lab reports."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)

    # -- public API ---------------------------------------------------------

    def generate_report(
        self,
        pattern: Optional[str] = None,
        severity: Optional[str] = None,
        patient_id: Optional[str] = None,
        force_critical: bool = False,
    ) -> PatientReport:
        """
        Generate a single patient report.

        Args:
            pattern: If given, embed this disease pattern into the report.
            severity: MILD / MODERATE / SEVERE — only used with pattern.
            patient_id: Override the auto-generated ID.
            force_critical: If True, ensure at least one CRITICAL value.

        Returns:
            A fully populated PatientReport with messy data.
        """
        gender = self._rng.choice(["Male", "Female"])
        age = self._rng.randint(18, 80)
        name = self._rng.choice(INDIAN_NAMES[gender])
        pid = patient_id or f"P{self._rng.randint(100, 999)}"
        center = self._rng.choice(CENTERS)
        symptom = self._rng.choice(SYMPTOMS)
        report_date = self._random_date()

        tests: Dict[str, TestResult] = {}
        for test_name, info in NORMAL_RANGES.items():
            nmin, nmax = self._get_range(test_name, gender)
            unit = info["unit"]
            value = self._generate_normal_value(nmin, nmax)
            raw_value = str(round(value, 2))
            tests[test_name] = TestResult(
                test_name=test_name,
                value=value,
                raw_value=raw_value,
                unit=unit,
                normal_min=nmin,
                normal_max=nmax,
                status=_classify_value(test_name, value, nmin, nmax),
            )

        # Overlay disease pattern
        if pattern and pattern in PATTERNS:
            self._apply_pattern(tests, pattern, severity or "MODERATE", gender)

        # Force a critical if requested and none exists yet
        if force_critical and not any(
            t.status.startswith("CRITICAL") for t in tests.values()
        ):
            self._inject_critical(tests)

        # Inject messiness
        tests = self._inject_mess(tests)

        return PatientReport(
            patient_id=pid,
            patient_name=name,
            age=age,
            gender=gender,
            symptoms=symptom,
            tests=tests,
            report_date=report_date,
            center_name=center,
        )

    def generate_task1_report(self) -> PatientReport:
        """Single report with random abnormalities for flagging task."""
        pattern = self._rng.choice(list(PATTERNS.keys()))
        severity = self._rng.choice(["MILD", "MODERATE", "SEVERE"])
        return self.generate_report(pattern=pattern, severity=severity)

    def generate_task2_report(self) -> PatientReport:
        """Single report with a specific disease pattern for detection task."""
        pattern = self._rng.choice(list(PATTERNS.keys()))
        severity = self._rng.choice(["MILD", "MODERATE", "SEVERE"])
        return self.generate_report(pattern=pattern, severity=severity)

    def generate_task3_reports(self) -> List[PatientReport]:
        """Five patients with varying urgency for triage ranking task."""
        # Ensure variety: pick different patterns, severity levels, ages
        patterns = self._rng.sample(
            list(PATTERNS.keys()), min(5, len(PATTERNS))
        )
        severities = ["SEVERE", "MODERATE", "MILD", "MILD", "MODERATE"]
        self._rng.shuffle(severities)

        reports: List[PatientReport] = []
        for i, (pat, sev) in enumerate(zip(patterns, severities)):
            force_crit = i == 0  # guarantee at least one critical patient
            report = self.generate_report(
                pattern=pat,
                severity=sev,
                patient_id=f"P{i + 1:03d}",
                force_critical=force_crit,
            )
            reports.append(report)

        # Shuffle so ground-truth order ≠ list order
        self._rng.shuffle(reports)
        return reports

    # -- internal helpers ---------------------------------------------------

    def _get_range(self, test_name: str, gender: str) -> Tuple[float, float]:
        info = NORMAL_RANGES[test_name]
        if info.get("gender_specific"):
            return info[gender]  # type: ignore[return-value]
        return info["min"], info["max"]  # type: ignore[return-value]

    def _generate_normal_value(self, nmin: float, nmax: float) -> float:
        """Value inside the normal range with slight noise."""
        mid = (nmin + nmax) / 2
        spread = (nmax - nmin) / 2
        return round(mid + self._rng.gauss(0, spread * 0.3), 2)

    def _generate_abnormal_value(
        self, nmin: float, nmax: float, direction: str, severity_factor: float = 1.0
    ) -> float:
        """Value outside the normal range in the requested direction."""
        span = nmax - nmin
        if direction in ("LOW", "CRITICAL_LOW"):
            offset = span * 0.3 * severity_factor
            val = nmin - offset - abs(self._rng.gauss(0, span * 0.1))
            return round(max(0, val), 2)
        else:  # HIGH / CRITICAL_HIGH
            offset = span * 0.3 * severity_factor
            val = nmax + offset + abs(self._rng.gauss(0, span * 0.1))
            return round(val, 2)

    def _apply_pattern(
        self,
        tests: Dict[str, TestResult],
        pattern: str,
        severity: str,
        gender: str,
    ) -> None:
        """Overwrite relevant tests to match a disease pattern."""
        pinfo = PATTERNS[pattern]
        sev_factor = {"MILD": 0.8, "MODERATE": 1.5, "SEVERE": 3.0}[severity]

        # Apply severity-specific threshold values for the key indicator test
        thresholds = pinfo["severity_thresholds"][severity]

        for tname, direction in zip(pinfo["tests"], pinfo["direction"]):
            if tname not in tests:
                continue
            tr = tests[tname]
            nmin, nmax = self._get_range(tname, gender)

            # If the threshold dict has a specific value for this test, use it
            if tname in thresholds:
                value = float(thresholds[tname])
                # Add small noise so values aren't exact round numbers
                value += self._rng.gauss(0, abs(value) * 0.05)
                value = round(value, 2)
                if direction in ("LOW", "CRITICAL_LOW"):
                    value = min(value, nmin - 0.1)
                else:
                    value = max(value, nmax + 0.1)
            else:
                value = self._generate_abnormal_value(nmin, nmax, direction, sev_factor)

            raw_value = str(value)
            status = _classify_value(tname, value, nmin, nmax)
            tests[tname] = TestResult(
                test_name=tname,
                value=value,
                raw_value=raw_value,
                unit=tr.unit,
                normal_min=nmin,
                normal_max=nmax,
                status=status,
            )

    def _inject_critical(self, tests: Dict[str, TestResult]) -> None:
        """Force at least one test into a CRITICAL range."""
        crit_candidates = list(CRITICAL_THRESHOLDS.keys())
        test_name = self._rng.choice(crit_candidates)
        if test_name not in tests:
            return
        tr = tests[test_name]
        crit = CRITICAL_THRESHOLDS[test_name]
        if "CRITICAL_LOW" in crit:
            val = round(crit["CRITICAL_LOW"] * 0.6 + self._rng.gauss(0, 0.5), 2)
            val = max(0, val)
            status = "CRITICAL_LOW"
        else:
            val = round(crit["CRITICAL_HIGH"] * 1.4 + abs(self._rng.gauss(0, 10)), 2)
            status = "CRITICAL_HIGH"
        tests[test_name] = TestResult(
            test_name=tr.test_name,
            value=val,
            raw_value=str(val),
            unit=tr.unit,
            normal_min=tr.normal_min,
            normal_max=tr.normal_max,
            status=status,
        )

    def _inject_mess(self, tests: Dict[str, TestResult]) -> Dict[str, TestResult]:
        """Apply all mess-injection types with configured probabilities."""
        messy: Dict[str, TestResult] = {}
        for canonical_name, tr in tests.items():
            value = tr.value
            raw_value = tr.raw_value
            unit = tr.unit
            display_name = canonical_name

            # 1. Missing value (15 %)
            if self._rng.random() < 0.15:
                sentinel = self._rng.choice(MISSING_SENTINELS)
                value = None
                raw_value = sentinel
                status = "MISSING"
            else:
                status = tr.status

            # 2. Unit typo (20 %)
            if self._rng.random() < 0.20 and unit in UNIT_TYPOS:
                unit = self._rng.choice(UNIT_TYPOS[unit])

            # 3. Test-name variant (25 %)
            if self._rng.random() < 0.25 and canonical_name in TEST_NAME_VARIANTS:
                display_name = self._rng.choice(TEST_NAME_VARIANTS[canonical_name])

            # 4. Borderline value (10 %) — only when value exists & currently NORMAL
            if value is not None and status == "NORMAL" and self._rng.random() < 0.10:
                edge = self._rng.choice(["low", "high"])
                if edge == "low":
                    value = round(tr.normal_min * (1 + self._rng.uniform(-0.02, 0.02)), 2)
                else:
                    value = round(tr.normal_max * (1 + self._rng.uniform(-0.02, 0.02)), 2)
                raw_value = str(value)
                status = _classify_value(canonical_name, value, tr.normal_min, tr.normal_max)

            # 5. Whitespace noise (30 %)
            if self._rng.random() < 0.30 and raw_value:
                pad_l = " " * self._rng.randint(1, 3)
                pad_r = " " * self._rng.randint(1, 3)
                raw_value = pad_l + raw_value + pad_r

            # 6. Impossible value (2 %)
            if value is not None and self._rng.random() < 0.02:
                impossible = self._rng.choice([-abs(value), 0.0])
                value = impossible
                raw_value = str(value)
                status = _classify_value(canonical_name, value, tr.normal_min, tr.normal_max)

            messy[canonical_name] = TestResult(
                test_name=display_name,
                value=value,
                raw_value=raw_value,
                unit=unit,
                normal_min=tr.normal_min,
                normal_max=tr.normal_max,
                status=status,
            )

        return messy

    def _random_date(self) -> str:
        base = datetime(2025, 1, 1)
        delta = timedelta(days=self._rng.randint(0, 365))
        return (base + delta).strftime("%d-%m-%Y")
