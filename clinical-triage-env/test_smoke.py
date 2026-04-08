import asyncio
import httpx

async def main():
    async with httpx.AsyncClient(base_url="http://localhost:7860") as client:
        # Check health
        r = await client.get("/health")
        assert r.status_code == 200
        print("Health OK")

        # Test Task 1
        print("--- Testing Task 1 ---")
        r = await client.post("/reset", json={"task_id": "task1"})
        assert r.status_code == 200
        obs = r.json()
        print("Task 1 Reset OK. Budget:", obs["resources"]["test_budget"])
        
        # Order a test
        action = {"order_tests": ["Hemoglobin"], "flagged_tests": {}}
        r = await client.post("/step", json=action)
        res = r.json()
        print(f"Task 1 Step 1: ordered Hemoglobin. Events: {res['observation']['events']}")
        
        # Test Task 2
        print("--- Testing Task 2 ---")
        r = await client.post("/reset", json={"task_id": "task2"})
        obs = r.json()
        
        action = {"order_tests": ["TSH"], "identified_patterns": [], "start_treatment": True}
        r = await client.post("/step", json=action)
        res = r.json()
        print(f"Task 2 Step 1 (Treatment Started). Events: {res['observation']['events']}")
        
        # Patient should start stabilizing
        action = {"order_tests": [], "identified_patterns": ["Hypothyroidism"], "severity": "MODERATE"}
        r = await client.post("/step", json=action)
        res = r.json()
        print(f"Task 2 Step 2 (Diagnosed). Status: {res['observation']['patients'][0]['status']}")

        # Test Task 3
        print("--- Testing Task 3 ---")
        r = await client.post("/reset", json={"task_id": "task3"})
        obs = r.json()
        patients = obs["patients"]
        print(f"Task 3 Reset OK. Started with {len(patients)} patients.")
        
        pid_to_treat = patients[0]["patient_id"]
        action = {"assign_doctor": pid_to_treat, "urgency_ranking": [p["patient_id"] for p in patients]}
        r = await client.post("/step", json=action)
        res = r.json()
        print(f"Task 3 Step 1 (Assigned doctor to {pid_to_treat}). Events: {res['observation']['events']}")
        
        # Advance multiple steps to see arrivals and deterioration
        for i in range(2, 6):
            r = await client.post("/step", json={})
            res = r.json()
            print(f"Task 3 Step {i}. Patients: {len(res['observation']['patients'])}, Events: {res['observation']['events']}")

if __name__ == "__main__":
    asyncio.run(main())
