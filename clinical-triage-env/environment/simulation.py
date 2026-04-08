"""
Simulation logic for ClinicalTriageEnv RL environment.
Contains the physics engine for vital signs evolution and patient arrivals.
"""
import random
from typing import Dict, List, Optional, Tuple

from environment.models import PatientState, VitalSigns
from environment.data_generator import DataGenerator, PATTERNS, NORMAL_RANGES, CRITICAL_THRESHOLDS

# Normal baseline vitals
# HR: 60-100, BP_SYS: 100-130, BP_DIA: 60-80, SpO2: 95-100, Temp: 36.5-37.5, Pain: 0
BASELINE_VITALS = {
    "heart_rate": (70.0, 80.0),
    "bp_systolic": (110.0, 120.0),
    "bp_diastolic": (70.0, 80.0),
    "spo2": (97.0, 99.0),
    "temperature": (36.5, 37.2),
    "pain_scale": (0, 2)
}

# Attractors for specific conditions (where vitals trend over time)
CONDITION_ATTRACTORS = {
    "Anemia": {
        "heart_rate": 110.0,  # Tachycardia to compensate for low O2
        "bp_systolic": 95.0,  # Hypotension
        "spo2": 92.0,         # Slightly low
        "pain_scale": 1
    },
    "Diabetes": {
        "heart_rate": 100.0,
        "bp_systolic": 140.0,
        "temperature": 37.5,
        "pain_scale": 3
    },
    "Liver Dysfunction": {
        "bp_systolic": 100.0,
        "temperature": 38.0,
        "pain_scale": 5
    },
    "Kidney Impairment": {
        "bp_systolic": 160.0, # Hypertension
        "heart_rate": 90.0,
        "pain_scale": 4       # Flank pain
    },
    "Hypothyroidism": {
        "heart_rate": 50.0,   # Bradycardia
        "bp_systolic": 100.0,
        "temperature": 35.8,  # Hypothermia
        "pain_scale": 2
    }
}

class PatientSimulator:
    """Manages the evolving state of a single patient in the ED."""
    
    def __init__(self, state: PatientState, rng: random.Random):
        self.state = state
        self._rng = rng
        self.deterioration_rate = self._compute_deterioration_rate()
        
    def _compute_deterioration_rate(self) -> float:
        """Rate of change depends on acuity (1=fastest, 5=slowest)."""
        # Critical patients deteriorate much faster
        rates = {1: 0.15, 2: 0.08, 3: 0.04, 4: 0.02, 5: 0.01}
        return rates.get(self.state.acuity, 0.05)

    def tick(self, time_delta_mins: int = 15) -> List[str]:
        """Advance time by time_delta_mins. Update vitals. Return any events."""
        events = []
        
        # Don't evolve if stabilized or crashed
        if self.state.status in ["STABILIZED", "CRITICAL_EVENT"]:
            return events
            
        self.state.time_waiting_mins += time_delta_mins
        
        # Treatment logic
        if self.state.status == "BEING_TREATED":
            events.extend(self._apply_treatment(time_delta_mins))
            return events
            
        # Deterioration logic (Waiting or Being Assessed)
        events.extend(self._apply_deterioration(time_delta_mins))
        
        # Check for critical events (crashing)
        if self._check_critical_thresholds():
            self.state.status = "CRITICAL_EVENT"
            events.append(f"CRITICAL EVENT: Patient {self.state.patient_id} has crashed! Vitals breached critical thresholds.")
            
        return events
        
    def _apply_treatment(self, time_delta_mins: int) -> List[str]:
        """Vitals drift toward normal baseline."""
        changes = []
        v = self.state.vitals
        
        # Gradually normalize
        v.heart_rate += (self._rng.uniform(*BASELINE_VITALS["heart_rate"]) - v.heart_rate) * 0.1
        v.bp_systolic += (self._rng.uniform(*BASELINE_VITALS["bp_systolic"]) - v.bp_systolic) * 0.1
        v.spo2 += min(100.0, (98.0 - v.spo2) * 0.2)
        v.pain_scale = max(0, v.pain_scale - 1)
        
        # Add slight noise
        self._add_noise(v)
        
        # Check if stabilized enough based on a generic heuristic
        if 60 <= v.heart_rate <= 100 and v.bp_systolic >= 90 and v.spo2 >= 95 and v.pain_scale <= 2:
            self.state.status = "STABILIZED"
            changes.append(f"Patient {self.state.patient_id} has fully stabilized.")
            
        return changes

    def _apply_deterioration(self, time_delta_mins: int) -> List[str]:
        """Vitals drift toward disease attractors or just worsen randomly."""
        v = self.state.vitals
        condition = self.state.underlying_condition
        attractors = CONDITION_ATTRACTORS.get(condition, {})
        
        rate = self.deterioration_rate * (time_delta_mins / 15.0)
        
        # Drift toward attractor if available, otherwise just worsen
        if "heart_rate" in attractors:
            v.heart_rate += (attractors["heart_rate"] - v.heart_rate) * rate
        elif v.heart_rate > 90:
            v.heart_rate += self._rng.uniform(0, 5) * rate  # Tachycardia worsens
            
        if "bp_systolic" in attractors:
            v.bp_systolic += (attractors["bp_systolic"] - v.bp_systolic) * rate
        elif v.bp_systolic < 100:
            v.bp_systolic -= self._rng.uniform(0, 5) * rate # Hypotension worsens
            
        if "spo2" in attractors:
            v.spo2 += (attractors["spo2"] - v.spo2) * rate
        else:
            v.spo2 -= self._rng.uniform(0, 1) * rate # Oxygen slowly drops
            
        # Pain increases slowly while waiting
        if self._rng.random() < 0.2 * rate and v.pain_scale < 10:
            v.pain_scale += 1
            
        self._add_noise(v)
        return []

    def _add_noise(self, v: VitalSigns):
        """Add small random walk noise to vitals for realism."""
        v.heart_rate += self._rng.gauss(0, 2)
        v.bp_systolic += self._rng.gauss(0, 3)
        v.bp_diastolic += self._rng.gauss(0, 2)
        v.spo2 = min(100.0, v.spo2 + self._rng.gauss(0, 0.5))
        v.temperature += self._rng.gauss(0, 0.1)

    def _check_critical_thresholds(self) -> bool:
        """Return True if patient is crashing."""
        v = self.state.vitals
        if v.spo2 < 85.0: return True
        if v.heart_rate > 150.0 or v.heart_rate < 40.0: return True
        if v.bp_systolic < 70.0 or v.bp_systolic > 200.0: return True
        return False


class SimulationEngine:
    """The overall hospital simulation controlling all patients and resources."""
    
    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)
        self._data_gen = DataGenerator(seed=seed)
        
    def generate_initial_patient(self, task_id: str, pattern: Optional[str] = None, severity: str = "MODERATE") -> PatientState:
        """Generate a single patient with hidden ground truth tests."""
        report = self._data_gen.generate_report(pattern=pattern, severity=severity)
        return self._report_to_state(report, task_id, pattern or "Unknown")
        
    def generate_task3_patients(self) -> List[PatientState]:
        """Generate 3 initial patients for the triage queue."""
        patterns = self._rng.sample(list(PATTERNS.keys()), min(3, len(PATTERNS)))
        severities = ["SEVERE", "MODERATE", "MILD"]
        self._rng.shuffle(severities)
        
        patients = []
        for i, (pat, sev) in enumerate(zip(patterns, severities)):
            report = self._data_gen.generate_report(pattern=pat, severity=sev, patient_id=f"P{i + 1:03d}")
            patients.append(self._report_to_state(report, "task3", pat))
            
        return patients
        
    def generate_new_arrival(self, next_id: int) -> PatientState:
        """Generate a stochastic new patient arriving mid-episode."""
        pattern = self._rng.choice(list(PATTERNS.keys()))
        # Mostly mild/moderate for mid-episode arrivals
        severity = self._rng.choices(["MILD", "MODERATE", "SEVERE"], weights=[0.5, 0.3, 0.2])[0]
        report = self._data_gen.generate_report(pattern=pattern, severity=severity, patient_id=f"P{next_id:03d}")
        return self._report_to_state(report, "task3", pattern)

    def _report_to_state(self, report, task_id: str, pattern: str) -> PatientState:
        """Convert a static PatientReport into a real-time PatientState."""
        
        # Calculate initial acuity (1=resuscitation, 5=non-urgent)
        criticals = sum(1 for t in report.tests.values() if t.status.startswith("CRITICAL"))
        abnormals = sum(1 for t in report.tests.values() if t.status in ["LOW", "HIGH"])
        
        if criticals > 0: acuity = 1 if criticals > 1 else 2
        elif abnormals > 3: acuity = 3
        elif abnormals > 0: acuity = 4
        else: acuity = 5
        
        # Generate initial vitals based on condition and acuity
        vitals = VitalSigns(
            heart_rate=self._rng.uniform(*BASELINE_VITALS["heart_rate"]),
            bp_systolic=self._rng.uniform(*BASELINE_VITALS["bp_systolic"]),
            bp_diastolic=self._rng.uniform(*BASELINE_VITALS["bp_diastolic"]),
            spo2=self._rng.uniform(*BASELINE_VITALS["spo2"]),
            temperature=self._rng.uniform(*BASELINE_VITALS["temperature"]),
            pain_scale=self._rng.randint(*BASELINE_VITALS["pain_scale"])
        )
        
        # Pull toward attractors immediately
        attr = CONDITION_ATTRACTORS.get(pattern, {})
        if "heart_rate" in attr: vitals.heart_rate = attr["heart_rate"] + self._rng.gauss(0, 5)
        if "bp_systolic" in attr: vitals.bp_systolic = attr["bp_systolic"] + self._rng.gauss(0, 10)
        if "spo2" in attr: vitals.spo2 = attr["spo2"] + self._rng.gauss(0, 1)
        if "temperature" in attr: vitals.temperature = attr["temperature"] + self._rng.gauss(0, 0.3)
        if "pain_scale" in attr: vitals.pain_scale = attr["pain_scale"]
        
        # Enforce limits
        vitals.spo2 = min(100.0, max(0.0, vitals.spo2))
        
        # For task 2, we reveal 3 basic tests initially
        revealed = {}
        if task_id == "task2":
            basics = [t for t in ["Hemoglobin", "Blood Sugar (Fasting)", "Creatinine", "SGPT (ALT)"] if t in report.tests]
            revealed = {k: report.tests[k] for k in basics[:3]}
            
        return PatientState(
            patient_id=report.patient_id,
            patient_name=report.patient_name,
            age=report.age,
            gender=report.gender,
            presenting_complaint=report.symptoms,
            status="WAITING" if task_id == "task3" else "BEING_ASSESSED",
            time_waiting_mins=0,
            acuity=acuity,
            vitals=vitals,
            revealed_tests=revealed,
            available_tests=list(report.tests.keys()),
            doctor_assigned=False,
            underlying_condition=pattern,
            true_tests=report.tests
        )
