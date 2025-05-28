import uuid
import json
import pathlib
import datetime # Using standard datetime
from typing import Dict, Any, Tuple # For type hinting

from fastapi import FastAPI, BackgroundTasks, HTTPException, Depends
from fastapi.responses import JSONResponse

from app.models import (
    OptimizationParams,
    OptimizationRunResponse,
    ProductionTask,
    DailyInventory,
    OptimizationResult,
    ErrorResponse
)
from app.optimization_core.data_parser import (
    load_production_data, 
    WORKING_DAYS_PER_YEAR, # Import necessary global constants
    TOOL_CHANGE_TIME_PER_TOOL, 
    RAW_MATERIAL_CHANGE_TIME
)
from app.optimization_core.solver import build_and_solve_advanced_model

# --- App Initialization ---
app = FastAPI(
    title="Production Planning Optimizer API",
    description="API for running and retrieving production optimization plans.",
    version="0.3.0" # Version updated for this implementation step
)

# --- Global Variables & Startup ---
# Store loaded data globally. For production, consider a more robust caching/sharing mechanism.
PRODUCTION_DATA_CACHE: Dict[str, Any] = {}
# This will store the tuple (items_data, machines_data, raw_materials_data, tools_data, global_constants_dict)
OPTIMIZATION_INPUT_DATA_TUPLE: Optional[Tuple[Dict, Dict, Dict, Dict, Dict]] = None


RUN_RESULTS_DIR = pathlib.Path("run_results")
SERVER_INSTANCE_START_TIME_UTC = datetime.datetime.utcnow() # Fallback reference time

def get_optimization_input_data() -> Tuple[Dict, Dict, Dict, Dict, Dict]:
    """Dependency to get the loaded production data tuple."""
    if OPTIMIZATION_INPUT_DATA_TUPLE is None:
        # This indicates an issue during startup data loading
        raise HTTPException(status_code=503, detail="Production data not loaded or loading failed. Check server logs.")
    return OPTIMIZATION_INPUT_DATA_TUPLE

@app.on_event("startup")
async def startup_event():
    global PRODUCTION_DATA_CACHE, OPTIMIZATION_INPUT_DATA_TUPLE
    print("INFO: Loading production data at startup...")
    try:
        items, machines, raw_materials, tools = load_production_data("turning-data.xlsx")
        
        if not items or not machines:
            print("ERROR: Failed to load essential data (items or machines). API may not function correctly.")
            PRODUCTION_DATA_CACHE = {"error": "Failed to load essential items or machines data."}
            OPTIMIZATION_INPUT_DATA_TUPLE = None # Ensure it's None if loading fails
        else:
            global_constants_for_solver = {
                "tool_change_time_per_tool": TOOL_CHANGE_TIME_PER_TOOL,
                "raw_material_change_time": RAW_MATERIAL_CHANGE_TIME,
                "working_days_per_year": WORKING_DAYS_PER_YEAR
            }
            PRODUCTION_DATA_CACHE = { # Keep this for potential other uses if needed
                "items_data": items,
                "machines_data": machines,
                "raw_materials_data": raw_materials,
                "tools_data": tools,
                "global_constants": global_constants_for_solver
            }
            # This is the primary data structure passed to the solver
            OPTIMIZATION_INPUT_DATA_TUPLE = (items, machines, raw_materials, tools, global_constants_for_solver)
            print("INFO: Production data loaded successfully.")

    except FileNotFoundError:
        print("ERROR: 'turning-data.xlsx' not found. API will not be able to run optimizations.")
        PRODUCTION_DATA_CACHE = {"error": "'turning-data.xlsx' not found."}
        OPTIMIZATION_INPUT_DATA_TUPLE = None
    except Exception as e:
        print(f"ERROR: Error loading production data: {e}")
        PRODUCTION_DATA_CACHE = {"error": f"An unexpected error occurred during data loading: {str(e)}"}
        OPTIMIZATION_INPUT_DATA_TUPLE = None

    # Create results directory
    RUN_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"INFO: Results will be stored in: {RUN_RESULTS_DIR.resolve()}")


# --- Helper Function for Background Task ---
async def run_optimization_task_background(
    run_id: str, 
    params: OptimizationParams, 
    # The dependency ensures data is loaded, but we pass the tuple directly
    input_data_tuple: Tuple[Dict, Dict, Dict, Dict, Dict] 
):
    """
    Runs the optimization model in the background and saves results.
    """
    run_file_path = RUN_RESULTS_DIR / f"{run_id}.json"
    # Record the actual start time for this specific optimization run
    optimization_request_time_utc_iso = datetime.datetime.utcnow().isoformat()

    # Update the status file with the precise optimization start time
    # This overwrites the initial "processing" status file but adds the specific start time
    with open(run_file_path, "w") as f:
        json.dump({
            "run_id": run_id, 
            "status": "processing", 
            "message": "Solver initiated.",
            "optimization_start_datetime_utc": optimization_request_time_utc_iso # Time when processing actually starts
        }, f, indent=4)

    try:
        print(f"INFO: [{run_id}] Background task: Starting advanced model build and solve...")
        
        # Ensure initial_inventory is correctly passed (empty dict if None, as Pydantic model does)
        initial_inv = params.initial_inventory

        # Call the solver function
        # The input_data_tuple already contains the global_constants dict as its 5th element
        solver_results_dict = build_and_solve_advanced_model(
            parsed_data=input_data_tuple, # This is the tuple (items, machines, raw_mats, tools, global_consts)
            planning_horizon_days=params.planning_horizon_days,
            initial_inventory_dict=initial_inv,
            n_potential_batches_per_item=params.n_potential_batches_per_item,
            max_lot_size_multiplier=params.max_lot_size_multiplier,
            time_limit_seconds=params.solver_time_limit_seconds
        )
        print(f"INFO: [{run_id}] Background task: Solver finished. Status: {solver_results_dict.get('status', 'UNKNOWN')}")

        # Augment results with run_id and the specific optimization start time
        final_results_to_save = solver_results_dict.copy() # Avoid modifying original dict if reused
        final_results_to_save["run_id"] = run_id # Ensure run_id is in the saved file
        final_results_to_save["optimization_start_datetime_utc"] = optimization_request_time_utc_iso

        with open(run_file_path, "w") as f:
            json.dump(final_results_to_save, f, indent=4)
        print(f"INFO: [{run_id}] Background task: Results saved to {run_file_path}")

    except Exception as e:
        print(f"ERROR: [{run_id}] Background task: Error during optimization: {e}")
        # Ensure stack trace is printed for debugging if possible
        import traceback
        traceback.print_exc()
        
        error_result = {
            "run_id": run_id,
            "status": "failed",
            "message": "An unexpected error occurred during optimization processing.",
            "error_message": str(e),
            "optimization_start_datetime_utc": optimization_request_time_utc_iso,
            "objective_value": None, # Ensure all fields from solver output are present
            "active_batches_details": None,
            "inventory_details": None,
            "solver_stats": None
        }
        with open(run_file_path, "w") as f:
            json.dump(error_result, f, indent=4)


# --- API Endpoints ---
@app.post("/optimize/plan", response_model=OptimizationRunResponse, status_code=202)
async def create_optimization_plan_endpoint(
    params: OptimizationParams,
    background_tasks: BackgroundTasks,
    # Use the dependency to ensure data is loaded and get it
    input_data_for_solver: Tuple[Dict, Dict, Dict, Dict, Dict] = Depends(get_optimization_input_data) 
):
    """
    Initiates a new production optimization plan.
    The optimization runs in the background. Results are stored and can be retrieved via GET endpoint.
    """
    run_id = str(uuid.uuid4())
    run_file_path = RUN_RESULTS_DIR / f"{run_id}.json"
    
    # Store initial status file immediately
    # The background task will overwrite this with more details (like specific opt_start_time)
    initial_status_payload = {
        "run_id": run_id,
        "status": "queued", # Changed to queued, background task will set to 'processing'
        "message": "Optimization task accepted and queued for processing.",
        # optimization_start_datetime_utc will be set by the background task when it actually starts
    }
    with open(run_file_path, "w") as f:
        json.dump(initial_status_payload, f, indent=4)

    print(f"INFO: Queuing optimization run: {run_id} with params: {params.model_dump_json(indent=2)}")
    background_tasks.add_task(
        run_optimization_task_background, 
        run_id, 
        params, 
        input_data_for_solver # Pass the loaded data tuple
    )

    return OptimizationRunResponse(
        run_id=run_id,
        status="queued", # Return "queued" status
        message="Optimization task queued for background processing."
    )

@app.get("/optimize/plan/results/{run_id_str}", response_model=OptimizationResult, responses={404: {"model": ErrorResponse}})
async def get_optimization_results_endpoint(run_id_str: str):
    """
    Retrieves the results of a specific optimization run.
    """
    run_file_path = RUN_RESULTS_DIR / f"{run_id_str}.json"

    if not run_file_path.exists():
        raise HTTPException(status_code=404, detail=f"Results for run_id '{run_id_str}' not found. "
                                                     "It may still be processing or the ID is invalid.")

    try:
        with open(run_file_path, "r") as f:
            raw_results = json.load(f)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail=f"Error decoding results for run_id '{run_id_str}'. File may be corrupted or incomplete.")

    # Reference datetime for this specific optimization run
    # This should be saved by `run_optimization_task_background`
    opt_start_utc_iso_str = raw_results.get("optimization_start_datetime_utc")
    if opt_start_utc_iso_str:
        reference_datetime_utc = datetime.datetime.fromisoformat(opt_start_utc_iso_str)
    else:
        # Fallback if optimization_start_datetime_utc is somehow missing (e.g. very early error)
        # Or if status is "queued" and task hasn't started to write its own start time.
        if raw_results.get("status") == "queued":
             # For queued status, it's okay not to have specific start time yet.
             # We can return the basic status without trying to process tasks.
            return OptimizationResult(
                run_id=raw_results.get("run_id", run_id_str),
                status=raw_results.get("status"),
                message=raw_results.get("message"),
                optimization_start_datetime_utc=None # Not started yet
            )
        # If not queued and still missing, use server start as a last resort and log warning.
        print(f"WARN: optimization_start_datetime_utc missing for run {run_id_str}. Using server start time as fallback.")
        reference_datetime_utc = SERVER_INSTANCE_START_TIME_UTC 

    # Transform raw solver output to OptimizationResult Pydantic model
    transformed_plan_per_day: Dict[int, List[ProductionTask]] = {}
    active_batches = raw_results.get("active_batches_details") # This is a list of dicts
    
    if isinstance(active_batches, list): # Ensure it's a list before iterating
        for task_data in active_batches:
            start_mins = task_data["start_time_minutes"]
            end_mins = task_data["end_time_minutes"]
            
            start_dt_utc = reference_datetime_utc + datetime.timedelta(minutes=start_mins)
            end_dt_utc = reference_datetime_utc + datetime.timedelta(minutes=end_mins)

            # Day index for grouping: 0 for first day, 1 for second, etc.
            day_index = start_mins // (24 * 60) 
            
            prod_task = ProductionTask(
                item_id=task_data["item_id"],
                lot_size=task_data["lot_size"],
                machine_id=task_data["machine_id"],
                start_time_minutes=start_mins,
                end_time_minutes=end_mins,
                duration_minutes=task_data.get("duration_minutes", end_mins - start_mins), # Calculate if missing
                start_datetime_utc=start_dt_utc.isoformat(),
                end_datetime_utc=end_dt_utc.isoformat()
            )
            if day_index not in transformed_plan_per_day:
                transformed_plan_per_day[day_index] = []
            transformed_plan_per_day[day_index].append(prod_task)
    
    transformed_inventory_projection: List[DailyInventory] = []
    inventory_details_list = raw_results.get("inventory_details")
    if isinstance(inventory_details_list, list):
        for inv_data in inventory_details_list:
            transformed_inventory_projection.append(DailyInventory(
                item_id=inv_data["item_id"],
                day=inv_data["day"],
                inventory_level=inv_data["inventory_level"]
            ))
            
    # Map solver status (e.g. "OPTIMAL") to a more generic API status ("completed", "failed")
    solver_status_str = raw_results.get("status", "unknown").upper() # Normalize to uppercase
    api_status = "unknown"
    if solver_status_str in ["OPTIMAL", "FEASIBLE"]:
        api_status = "completed"
    elif solver_status_str in ["INFEASIBLE", "MODEL_INVALID", "ERROR", "ABORTED"]: # Consider other CP-SAT non-success statuses
        api_status = "failed"
    elif solver_status_str in ["PROCESSING", "QUEUED"]: # if file is read while solver is running or just queued
        api_status = solver_status_str.lower() 

    return OptimizationResult(
        run_id=raw_results.get("run_id", run_id_str),
        status=api_status,
        total_cost=raw_results.get("objective_value"), # Directly from solver output
        plan_per_day=transformed_plan_per_day if transformed_plan_per_day else None,
        inventory_projection=transformed_inventory_projection if transformed_inventory_projection else None,
        solver_statistics=raw_results.get("solver_stats"),
        message=raw_results.get("message"),
        optimization_start_datetime_utc=opt_start_utc_iso_str, # Pass this through
        error_message=raw_results.get("error_message") 
    )

# Root endpoint for basic API info
@app.get("/")
async def root():
    return {"message": "Welcome to the Production Planning Optimizer API. See /docs for API documentation."}

# To run the app (if this file is executed directly, though `uvicorn app.main:app --reload` is preferred)
if __name__ == "__main__":
    import uvicorn
    print("INFO: Running Uvicorn server directly. For development, prefer: `uvicorn app.main:app --reload`")
    uvicorn.run(app, host="0.0.0.0", port=8000)
