import os
import uuid
import datetime
import json
from typing import Tuple, Dict, Any, List # Added List

# Using Flask's Request object for type hinting, actual object provided by Firebase/Flask environment
from flask import Request, jsonify 

from app.models import OptimizationParams, OptimizationResult, ProductionTask, DailyInventory # Ensure all needed models are imported
from app.optimization_core.data_parser import load_production_data
from app.optimization_core.solver import build_and_solve_advanced_model
from pydantic import ValidationError
from google.cloud import storage # For GCS interactions


def _get_gcs_client():
    """Helper to get GCS client, memoized for efficiency if called multiple times in one invocation."""
    if not hasattr(_get_gcs_client, "storage_client"):
        _get_gcs_client.storage_client = storage.Client()
    return _get_gcs_client.storage_client

def _upload_to_gcs(bucket_name: str, blob_path: str, data_dict: Dict[str, Any], run_id: str):
    """Uploads a dictionary as JSON to GCS."""
    try:
        client = _get_gcs_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        blob.upload_from_string(
            json.dumps(data_dict, indent=2), # Use indent for readability in GCS
            content_type="application/json"
        )
        print(f"INFO: [{run_id}] Successfully uploaded to gs://{bucket_name}/{blob_path}")
    except Exception as e:
        print(f"ERROR: [{run_id}] Failed to upload to gs://{bucket_name}/{blob_path}. Error: {str(e)}")
        # Depending on policy, this might need to raise an error or handle more gracefully.

def _normalize_solver_status(solver_status_str: str) -> str:
    """ Normalizes solver status to API status """
    solver_status_str = solver_status_str.upper() # Normalize to uppercase
    if solver_status_str in ["OPTIMAL", "FEASIBLE"]:
        return "completed"
    elif solver_status_str in ["INFEASIBLE", "MODEL_INVALID", "ABORTED"]: # Consider other CP-SAT non-success statuses
        return "failed"
    elif solver_status_str in ["PROCESSING", "QUEUED"]:
        return solver_status_str.lower()
    return "unknown" # Default for other statuses like "UNKNOWN_STATUS" from solver

def api_optimize_plan(request: Request) -> Tuple[Dict[str, Any], int]:
    """
    Firebase Cloud Function entry point for initiating an optimization plan.
    Takes a Flask-like request object. Results are stored in GCS.
    """
    try:
        params_data = request.get_json()
        if not params_data:
            return {"error": "No JSON body provided or content type not application/json"}, 400
    except Exception as e:
        return {"error": f"Invalid JSON format: {str(e)}"}, 400

    try:
        validated_params = OptimizationParams(**params_data)
    except ValidationError as e:
        return {"error": "Validation error", "details": e.errors()}, 400
    except Exception as e:
        return {"error": f"Parameter parsing error: {str(e)}"}, 400

    data_bucket_name = os.environ.get("DATA_BUCKET_NAME")
    data_blob_name = os.environ.get("DATA_FILE_BLOB_NAME")
    results_bucket_name = os.environ.get("RESULTS_BUCKET_NAME")

    if not data_bucket_name or not data_blob_name:
        print("ERROR: GCS data environment variables (DATA_BUCKET_NAME, DATA_FILE_BLOB_NAME) not set.")
        return {"error": "Server configuration error: GCS data source not set"}, 500
    if not results_bucket_name:
        print("ERROR: GCS results environment variable (RESULTS_BUCKET_NAME) not set.")
        return {"error": "Server configuration error: GCS results destination not set"}, 500

    run_id = str(uuid.uuid4())
    optimization_start_datetime_utc_iso = datetime.datetime.utcnow().isoformat()
    result_blob_path = f"run_results/{run_id}.json"

    # Initial Status Upload to GCS
    initial_status = {
        "run_id": run_id,
        "status": "processing", # Directly to processing
        "message": "Optimization initiated and processing started.",
        "optimization_start_datetime_utc": optimization_start_datetime_utc_iso,
        "params_submitted": validated_params.model_dump() # Store submitted params for reference
    }
    _upload_to_gcs(results_bucket_name, result_blob_path, initial_status, run_id)
    
    print(f"INFO: [{run_id}] Loading data from GCS: gs://{data_bucket_name}/{data_blob_name}")
    parsed_data_tuple = load_production_data(data_bucket_name, data_blob_name)
    
    if parsed_data_tuple[0] is None: # Check if items_data is None
        error_msg = "Failed to load production data from GCS. Check logs for data_parser errors."
        print(f"ERROR: [{run_id}] {error_msg}")
        # Update status in GCS to failed
        failure_status = {**initial_status, "status": "failed", "message": error_msg, 
                          "error_message": "Data loading failed."}
        _upload_to_gcs(results_bucket_name, result_blob_path, failure_status, run_id)
        return {"error": error_msg, "details": f"GCS path: gs://{data_bucket_name}/{data_blob_name}"}, 500

    print(f"INFO: [{run_id}] Starting synchronous optimization model build and solve...")
    print(f"INFO: [{run_id}] Parameters: {validated_params.model_dump_json(indent=2)}")

    try:
        solver_result_dict = build_and_solve_advanced_model(
            parsed_data=parsed_data_tuple,
            planning_horizon_days=validated_params.planning_horizon_days,
            initial_inventory_dict=validated_params.initial_inventory if validated_params.initial_inventory is not None else {},
            n_potential_batches_per_item=validated_params.n_potential_batches_per_item,
            max_lot_size_multiplier=validated_params.max_lot_size_multiplier,
            time_limit_seconds=validated_params.solver_time_limit_seconds
        )
        print(f"INFO: [{run_id}] Solver finished. Raw status: {solver_result_dict.get('status', 'UNKNOWN')}")

        # Ensure run_id and start time are in the final results, and update status interpretation
        final_solver_status_str = solver_result_dict.get("status", "UNKNOWN_STATUS")
        api_status = _normalize_solver_status(final_solver_status_str)

        final_result_to_upload = {
            **solver_result_dict, # Includes original solver status, objective, details, stats, messages
            "run_id": run_id, # Ensure it's there
            "status": api_status, # Overwrite with normalized API status
            "optimization_start_datetime_utc": optimization_start_datetime_utc_iso,
            "params_submitted": validated_params.model_dump() # Re-add for completeness in final result
        }
        # Example: "completed_optimal" or "failed_infeasible"
        final_result_to_upload["message"] = f"Optimization {api_status}. Solver status: {final_solver_status_str}."


        _upload_to_gcs(results_bucket_name, result_blob_path, final_result_to_upload, run_id)

        http_status_code = 202 # Accepted (processing started, results stored)
        return {
            "run_id": run_id, 
            "status": "processing_initiated", # The job was initiated, final status in GCS
            "message": f"Optimization task {run_id} initiated. Results will be stored in GCS at gs://{results_bucket_name}/{result_blob_path}.",
            "results_gcs_path": f"gs://{results_bucket_name}/{result_blob_path}"
        }, http_status_code

    except Exception as e:
        print(f"ERROR: [{run_id}] Unexpected error during solver execution: {str(e)}")
        import traceback
        traceback.print_exc()
        # Update status in GCS to failed
        error_status = {
            **initial_status, 
            "status": "failed", 
            "message": "An unexpected error occurred during solver execution.",
            "error_message": str(e)
        }
        _upload_to_gcs(results_bucket_name, result_blob_path, error_status, run_id)
        return {
            "run_id": run_id, 
            "error": "An unexpected error occurred during solver execution.",
            "details": str(e),
        }, 500


def api_get_plan_results(request: Request) -> Tuple[Dict[str, Any], int]:
    """
    Firebase Cloud Function entry point for retrieving and transforming optimization plan results from GCS.
    """
    path_parts = request.path.split('/')
    run_id = path_parts[-1] if len(path_parts) > 0 and path_parts[-1] else request.args.get("run_id")
    
    if not run_id:
        return {"error": "run_id missing from path or query parameters"}, 400

    results_bucket_name = os.environ.get("RESULTS_BUCKET_NAME")
    if not results_bucket_name:
        print("ERROR: GCS results environment variable (RESULTS_BUCKET_NAME) not set.")
        return {"error": "Server configuration error: GCS results source not set"}, 500

    result_blob_path = f"run_results/{run_id}.json"
    print(f"INFO: [{run_id}] Attempting to fetch results from GCS: gs://{results_bucket_name}/{result_blob_path}")

    try:
        client = _get_gcs_client()
        bucket = client.bucket(results_bucket_name)
        blob = bucket.blob(result_blob_path)

        if not blob.exists():
            print(f"WARN: [{run_id}] Results not found at gs://{results_bucket_name}/{result_blob_path}")
            return {"error": f"Results for run_id '{run_id}' not found."}, 404

        result_data_string = blob.download_as_string()
        result_data_dict = json.loads(result_data_string)
        print(f"INFO: [{run_id}] Successfully downloaded and parsed results from GCS.")

    except Exception as e:
        print(f"ERROR: [{run_id}] Failed to download or parse results from GCS. Error: {str(e)}")
        return {"error": "Failed to retrieve or parse results from GCS.", "details": str(e)}, 500

    # Transform to OptimizationResult Pydantic model
    try:
        opt_start_utc_iso_str = result_data_dict.get("optimization_start_datetime_utc")
        reference_datetime_utc = datetime.datetime.fromisoformat(opt_start_utc_iso_str) if opt_start_utc_iso_str else datetime.datetime.utcnow()
        if not opt_start_utc_iso_str:
            print(f"WARN: [{run_id}] 'optimization_start_datetime_utc' missing in GCS result. Using current time for datetime calculations.")


        transformed_plan_per_day: Dict[int, List[ProductionTask]] = {}
        active_batches = result_data_dict.get("active_batches_details")
        if isinstance(active_batches, list):
            for task_data in active_batches:
                start_mins = task_data["start_time_minutes"]
                end_mins = task_data["end_time_minutes"]
                start_dt_utc = reference_datetime_utc + datetime.timedelta(minutes=start_mins)
                end_dt_utc = reference_datetime_utc + datetime.timedelta(minutes=end_mins)
                day_index = start_mins // (24 * 60)
                
                transformed_plan_per_day.setdefault(day_index, []).append(ProductionTask(
                    item_id=task_data["item_id"],
                    lot_size=task_data["lot_size"],
                    machine_id=task_data["machine_id"],
                    start_time_minutes=start_mins,
                    end_time_minutes=end_mins,
                    duration_minutes=task_data.get("duration_minutes", end_mins - start_mins),
                    start_datetime_utc=start_dt_utc.isoformat() + "Z", # Explicit Z for UTC
                    end_datetime_utc=end_dt_utc.isoformat() + "Z",   # Explicit Z for UTC
                ))
        
        transformed_inventory_projection: List[DailyInventory] = []
        inventory_details_list = result_data_dict.get("inventory_details")
        if isinstance(inventory_details_list, list):
            for inv_data in inventory_details_list:
                transformed_inventory_projection.append(DailyInventory(
                    item_id=inv_data["item_id"],
                    day=inv_data["day"],
                    inventory_level=inv_data["inventory_level"]
                ))
        
        # Use the 'status' field that was already normalized and saved by api_optimize_plan
        api_status = result_data_dict.get("status", "unknown") 

        optimization_result_model = OptimizationResult(
            run_id=result_data_dict.get("run_id", run_id),
            status=api_status,
            total_cost=result_data_dict.get("objective_value"),
            plan_per_day=transformed_plan_per_day if transformed_plan_per_day else None,
            inventory_projection=transformed_inventory_projection if transformed_inventory_projection else None,
            solver_statistics=result_data_dict.get("solver_stats"),
            message=result_data_dict.get("message"),
            optimization_start_datetime_utc=opt_start_utc_iso_str,
            error_message=result_data_dict.get("error_message")
        )
        return optimization_result_model.model_dump(exclude_none=True), 200 # Use model_dump for Pydantic v2

    except ValidationError as e:
        print(f"ERROR: [{run_id}] Pydantic validation error during result transformation: {e}")
        return {"error": "Result data validation failed.", "details": e.errors()}, 500
    except Exception as e:
        print(f"ERROR: [{run_id}] Unexpected error during result transformation: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": "Unexpected error processing result data.", "details": str(e)}, 500


# --- Example of how to call these functions with a mock Flask request (for local testing) ---
def _run_mock_api_calls():
    """Mock test for API functions. Requires GCS env vars to be set."""
    
    # Common class for mock requests
    class MockFlaskRequest:
        def __init__(self, json_data=None, path_suffix="", args_dict=None):
            self._json_data = json_data
            self.path = f"/mock/{path_suffix}" 
            self.args = args_dict if args_dict else {}

        def get_json(self, silent=False, force=False, cache=True):
            if self._json_data is None and not silent and not force :
                 # Flask's get_json() raises an error if no JSON and not silent/force
                 raise Exception("Attempted to get JSON data when none was provided and not silent/force.")
            return self._json_data

    # --- Setup Environment Variables for Mock Test ---
    # These would be set in the Firebase Function's runtime environment configuration
    # For local testing, you MUST set these to point to actual GCS buckets you can write to.
    # IMPORTANT: Replace with your actual GCS bucket names for testing.
    test_data_bucket = "your-dev-gcs-bucket-for-input-data" # e.g., my-optimizer-data-bucket
    test_data_blob = "turning-data.xlsx"                 # e.g., test-data/turning-data.xlsx
    test_results_bucket = "your-dev-gcs-bucket-for-results" # e.g., my-optimizer-results-bucket
    
    os.environ["DATA_BUCKET_NAME"] = test_data_bucket
    os.environ["DATA_FILE_BLOB_NAME"] = test_data_blob
    os.environ["RESULTS_BUCKET_NAME"] = test_results_bucket
    
    print(f"Mock Test Note: Ensure GCS bucket for data '{test_data_bucket}' and blob '{test_data_blob}' exist and are accessible.")
    print(f"Mock Test Note: Results will be written to GCS bucket '{test_results_bucket}'. Ensure it exists and is writable.")
    print("If these are placeholder values, the GCS operations in the test will fail.")
    print("To run a meaningful local test, replace these with your actual GCS bucket names and ensure GOOGLE_APPLICATION_CREDENTIALS is set.")


    # --- Mocking api_optimize_plan call ---
    sample_params_for_post = {
        "planning_horizon_days": 1, # Very short for quick test
        "initial_inventory": {"Finished_P1": 0},
        "n_potential_batches_per_item": 1,
        "max_lot_size_multiplier": 1, 
        "solver_time_limit_seconds": 5 # Very short time limit
    }
    mock_request_post = MockFlaskRequest(json_data=sample_params_for_post, path_suffix="optimize")
    
    print("\n--- Mocking api_optimize_plan call ---")
    response_data_post, status_code_post = api_optimize_plan(mock_request_post)
    
    print(f"\nMock POST Response Status Code: {status_code_post}")
    print(f"Mock POST Response Data: {json.dumps(response_data_post, indent=2)}")

    # --- Mocking api_get_plan_results call ---
    run_id_from_post = None
    if status_code_post == 202: # Check for 'Accepted' status
        run_id_from_post = response_data_post.get("run_id")
    
    if run_id_from_post:
        print(f"\n--- Mocking api_get_plan_results call for run_id: {run_id_from_post} ---")
        # Simulate waiting for processing (important if testing against real async behavior, less so for sync)
        print("Simulating a short wait for results to be written to GCS...")
        import time
        time.sleep(2) # Give a moment for potential GCS write, though it's sync here.

        mock_request_get = MockFlaskRequest(path_suffix=f"results/{run_id_from_post}")
        response_data_get, status_code_get = api_get_plan_results(mock_request_get)
        
        print(f"\nMock GET Response Status Code: {status_code_get}")
        print(f"Mock GET Response Data: {json.dumps(response_data_get, indent=2)}")
    else:
        print("\nSkipping mock api_get_plan_results call as run_id was not obtained from POST.")


if __name__ == "__main__":
    print("Running local mock tests for refactored API functions...")
    print("IMPORTANT: This test interacts with GCS. Ensure your environment is configured:")
    print("1. `GOOGLE_APPLICATION_CREDENTIALS` environment variable is set.")
    print("2. The GCS buckets specified in `_run_mock_api_calls` (or your modified versions) exist and are accessible.")
    print("3. The input data file (e.g., `turning-data.xlsx`) exists in the specified DATA_BUCKET_NAME/DATA_FILE_BLOB_NAME.")
    
    # Check if placeholder bucket names are still there; if so, warn user.
    if "your-dev-gcs-bucket" in os.environ.get("DATA_BUCKET_NAME", "") or \
       "your-dev-gcs-bucket" in os.environ.get("RESULTS_BUCKET_NAME", ""):
        print("\nWARNING: Placeholder GCS bucket names detected in the test script.")
        print("Please update them in `_run_mock_api_calls` within `app/main.py` to your actual GCS buckets for the test to work.")
    else:
        _run_mock_api_calls() # Run the mock tests
    
    print("\nLocal mock tests finished.")
