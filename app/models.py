from pydantic import BaseModel, Field
from typing import Optional, List, Dict

class OptimizationParams(BaseModel):
    planning_horizon_days: int = Field(365, gt=0)
    initial_inventory: Optional[Dict[str, int]] = Field({}, description="Initial inventory levels for items {item_id: quantity}")
    # Parameters for the advanced model, matching the solver's expectations
    n_potential_batches_per_item: int = Field(3, gt=0, description="Max number of batches an item can be split into")
    max_lot_size_multiplier: int = Field(10, gt=0, description="Multiplier for max lot size based on total demand for horizon")
    solver_time_limit_seconds: int = Field(60, gt=0, description="Solver time limit in seconds")

class OptimizationRunResponse(BaseModel):
    run_id: str
    status: str # e.g., "processing", "completed", "failed"
    message: str

class ProductionTask(BaseModel):
    item_id: str
    lot_size: int
    machine_id: str
    start_time_minutes: int
    end_time_minutes: int
    duration_minutes: int # Added as per solver output structure
    start_datetime_utc: str # Will be ISO format string
    end_datetime_utc: str   # Will be ISO format string

class DailyInventory(BaseModel):
    item_id: str
    day: int # Day index, 0 to planning_horizon_days-1
    inventory_level: int

class OptimizationResult(BaseModel):
    run_id: str
    status: str # e.g., "processing", "completed", "failed"
    total_cost: Optional[float] = None # Corresponds to objective_value
    plan_per_day: Optional[Dict[int, List[ProductionTask]]] = None
    inventory_projection: Optional[List[DailyInventory]] = None
    solver_statistics: Optional[Dict] = None # To store wall_time, num_conflicts etc.
    message: Optional[str] = None
    # Added to store the reference start datetime for interpreting minute offsets for this run
    optimization_start_datetime_utc: Optional[str] = None 
    error_message: Optional[str] = None # For more specific errors

class ErrorResponse(BaseModel):
    detail: str
