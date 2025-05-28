import requests
import time
import pytest
import os
from typing import Dict, Any

# API base URL
BASE_URL = "http://127.0.0.1:8000"

# Helper to ensure the 'turning-data.xlsx' file exists for the tests
@pytest.fixture(scope="session", autouse=True)
def check_data_file():
    if not os.path.exists("turning-data.xlsx"):
        pytest.fail("'turning-data.xlsx' not found in the root directory. Tests cannot run.")

def test_happy_path_optimization():
    """
    Tests the full optimization process from POSTing parameters to GETting a completed plan.
    """
    print("\n--- Test: Happy Path Optimization ---")
    params = {
        "planning_horizon_days": 7, # Reduced for quicker testing
        "initial_inventory": {"Finished_P1": 10, "Finished_P2": 5}, # Example initial inventory
        "n_potential_batches_per_item": 2,
        "max_lot_size_multiplier": 1.5, # Allow some flexibility but not too large
        "solver_time_limit_seconds": 60 # Reduced from 120 for CI/testing speed
    }
    print(f"POST /optimize/plan with params: {params}")
    response_post = requests.post(f"{BASE_URL}/optimize/plan", json=params)
    assert response_post.status_code == 202, f"Expected 202 Accepted, got {response_post.status_code}. Response: {response_post.text}"
    
    run_response_data = response_post.json()
    run_id = run_response_data.get("run_id")
    assert run_id is not None, "run_id not found in POST response"
    print(f"Optimization run initiated with run_id: {run_id}")

    # Polling for results
    max_wait_time = params["solver_time_limit_seconds"] + 30  # Solver time + buffer
    poll_interval = 10  # seconds
    start_poll_time = time.time()
    
    result_data = None
    final_status = ""

    while time.time() - start_poll_time < max_wait_time:
        print(f"Polling for results for run_id: {run_id}...")
        response_get = requests.get(f"{BASE_URL}/optimize/plan/results/{run_id}")
        
        if response_get.status_code == 200:
            result_data = response_get.json()
            final_status = result_data.get("status")
            print(f"Current status: {final_status}")
            if final_status == "completed":
                print("Optimization completed successfully.")
                break
            elif final_status == "failed":
                pytest.fail(f"Optimization failed. Error: {result_data.get('message')} - {result_data.get('error_message')}")
            elif final_status not in ["processing", "queued"]: # Unknown status
                 pytest.fail(f"Unexpected optimization status: {final_status}. Full response: {result_data}")
        elif response_get.status_code == 404:
            print("Results not found yet (404), continuing to poll...")
        else: # Other unexpected errors
            pytest.fail(f"Error fetching results. Status: {response_get.status_code}. Response: {response_get.text}")
            
        time.sleep(poll_interval)
    else: # Timeout
        pytest.fail(f"Timeout: Optimization did not complete within {max_wait_time} seconds. Last status: {final_status}")

    # Assertions for completed optimization
    assert result_data is not None, "Result data should not be None after loop"
    assert final_status == "completed", "Final status should be 'completed'"
    
    print("\n--- Verifying OptimizationResult Structure ---")
    assert "total_cost" in result_data, "total_cost missing"
    assert result_data["total_cost"] is None or isinstance(result_data["total_cost"], (float, int)), "total_cost should be float or None"
    
    assert "plan_per_day" in result_data, "plan_per_day missing"
    plan_per_day = result_data["plan_per_day"]
    assert plan_per_day is None or isinstance(plan_per_day, dict), "plan_per_day should be a dict or None"

    if plan_per_day: # If there's a plan
        print(f"Number of days with scheduled tasks: {len(plan_per_day)}")
        for day_idx, tasks_on_day in plan_per_day.items():
            assert isinstance(day_idx, str), "Day index in plan_per_day should be string (JSON key)" # JSON keys are strings
            assert isinstance(tasks_on_day, list), f"Tasks on day {day_idx} should be a list"
            for task in tasks_on_day:
                assert "item_id" in task and isinstance(task["item_id"], str)
                assert "lot_size" in task and isinstance(task["lot_size"], int) and task["lot_size"] >= 0
                assert "machine_id" in task and isinstance(task["machine_id"], str)
                assert "start_time_minutes" in task and isinstance(task["start_time_minutes"], int)
                assert "end_time_minutes" in task and isinstance(task["end_time_minutes"], int)
                assert "duration_minutes" in task and isinstance(task["duration_minutes"], int)
                assert task["duration_minutes"] >= 0
                assert task["end_time_minutes"] >= task["start_time_minutes"]
                assert "start_datetime_utc" in task and isinstance(task["start_datetime_utc"], str)
                assert "end_datetime_utc" in task and isinstance(task["end_datetime_utc"], str)
                # Basic ISO format check (does not validate datetime itself fully)
                assert task["start_datetime_utc"].endswith("Z") or "+" in task["start_datetime_utc"]
                assert task["end_datetime_utc"].endswith("Z") or "+" in task["end_datetime_utc"]
    else:
        print("Plan_per_day is None or empty.")


    assert "inventory_projection" in result_data, "inventory_projection missing"
    inventory_projection = result_data["inventory_projection"]
    assert inventory_projection is None or isinstance(inventory_projection, list), "inventory_projection should be a list or None"

    if inventory_projection:
        print(f"Number of inventory entries: {len(inventory_projection)}")
        for entry in inventory_projection:
            assert "item_id" in entry and isinstance(entry["item_id"], str)
            assert "day" in entry and isinstance(entry["day"], int)
            assert "inventory_level" in entry and isinstance(entry["inventory_level"], int)
            # Allowing small epsilon for solver float precision if it were to occur, though inventory is int
            assert entry["inventory_level"] >= -1, f"Inventory level for {entry['item_id']} on day {entry['day']} is {entry['inventory_level']}, should be non-negative."
    else:
        print("Inventory_projection is None or empty.")

    print("\n--- Optimization Plan Summary ---")
    print(f"Run ID: {run_id}")
    print(f"Total Cost: {result_data.get('total_cost')}")
    if plan_per_day:
        num_tasks = sum(len(tasks) for tasks in plan_per_day.values())
        print(f"Number of Production Tasks Scheduled: {num_tasks}")
    else:
        print("Number of Production Tasks Scheduled: 0")
    
    solver_stats = result_data.get("solver_stats", {})
    print(f"Solver Wall Time: {solver_stats.get('wall_time', 'N/A')} seconds")
    print("Happy path optimization test completed.")


def test_invalid_input_to_optimize():
    """
    Tests the API's response to invalid input parameters.
    """
    print("\n--- Test: Invalid Input ---")
    invalid_params = {"planning_horizon_days": -1} # Example of invalid input
    
    print(f"POST /optimize/plan with invalid params: {invalid_params}")
    response = requests.post(f"{BASE_URL}/optimize/plan", json=invalid_params)
    
    assert response.status_code == 422, f"Expected 422 Unprocessable Entity, got {response.status_code}. Response: {response.text}"
    print("Invalid input test completed successfully (422 received).")


def test_get_non_existent_run_id():
    """
    Tests the API's response when requesting results for a non-existent run_id.
    """
    print("\n--- Test: Non-Existent Run ID ---")
    non_existent_run_id = "this-id-does-not-exist-12345"
    
    print(f"GET /optimize/plan/results/{non_existent_run_id}")
    response = requests.get(f"{BASE_URL}/optimize/plan/results/{non_existent_run_id}")
    
    assert response.status_code == 404, f"Expected 404 Not Found, got {response.status_code}. Response: {response.text}"
    print("Non-existent run_id test completed successfully (404 received).")

# To run these tests, ensure the FastAPI server is running:
# uvicorn app.main:app --reload
# And then run pytest:
# pytest -s tests/test_integration.py
# The -s flag shows print statements, which is useful for this test script.
