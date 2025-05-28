from ortools.sat.python import cp_model
from app.optimization_core.data_parser import load_production_data

def build_and_solve_basic_model(parsed_data, planning_horizon_days, initial_inventory_dict):
    """
    Builds and solves a basic CP-SAT model for production planning.
    """
    items_data, machines_data, _, _ = parsed_data 
    model = cp_model.CpModel()

    # --- Decision Variables ---
    produced_qty = {} # (item_id, day) -> quantity
    inventory = {}    # (item_id, day) -> quantity
    assign_item_day_to_machine = {} # (item_id, day, machine_id) -> Boolean

    all_item_ids = list(items_data.keys())
    all_machine_ids = list(machines_data.keys())
    
    # Max daily demand for upper bound on production
    # This is a simplification; a tighter bound might be needed for complex scenarios.
    max_possible_daily_production = 0
    for item_id in all_item_ids:
        # Ensure daily_demand is a number, handle potential errors if data is missing/incorrect
        daily_demand = items_data[item_id].get('daily_demand', 0)
        if not isinstance(daily_demand, (int, float)):
            print(f"Warning: Invalid or missing daily_demand for item {item_id}. Defaulting to 0 for bounds.")
            daily_demand = 0
        max_possible_daily_production = max(max_possible_daily_production, daily_demand * 2) # Produce at most twice the daily demand in one day

    if not all_item_ids:
        print("Error: No items found in parsed_data. Cannot build model.")
        return None
    if not all_machine_ids:
        print("Error: No machines found in parsed_data. Cannot build model.")
        return None

    for item_id in all_item_ids:
        for day in range(planning_horizon_days):
            produced_qty[item_id, day] = model.NewIntVar(0, int(max_possible_daily_production) if max_possible_daily_production > 0 else 1000, f"produced_{item_id}_d{day}")
            # Inventory can be large, but not negative.
            inventory[item_id, day] = model.NewIntVar(0, 1000000, f"inventory_{item_id}_d{day}") # Arbitrary large number for max inventory
            for machine_id in all_machine_ids:
                assign_item_day_to_machine[item_id, day, machine_id] = model.NewBoolVar(f"assign_{item_id}_d{day}_m{machine_id}")

    # --- Constraints ---

    # Inventory Balance
    for item_id in all_item_ids:
        for day in range(planning_horizon_days):
            prev_inventory = initial_inventory_dict.get(item_id, 0) if day == 0 else inventory[item_id, day - 1]
            daily_demand_val = items_data[item_id]['daily_demand']
            
            model.Add(inventory[item_id, day] == prev_inventory + produced_qty[item_id, day] - int(daily_demand_val))
            model.Add(inventory[item_id, day] >= 0) # Ensure inventory does not go negative (implicitly handled by daily_demand being int)

    # Machine Assignment
    for item_id in all_item_ids:
        for day in range(planning_horizon_days):
            # If item is produced on a given day, it must be assigned to exactly one machine.
            model.Add(sum(assign_item_day_to_machine[item_id, day, m_id] for m_id in all_machine_ids) == 1).OnlyEnforceIf(produced_qty[item_id, day] > 0)
            # If item is NOT produced on a given day, it must not be assigned to any machine.
            model.Add(sum(assign_item_day_to_machine[item_id, day, m_id] for m_id in all_machine_ids) == 0).OnlyEnforceIf(produced_qty[item_id, day] == 0)


    # Simplified Machine Capacity
    for machine_id in all_machine_ids:
        available_time = machines_data[machine_id]['available_minutes_per_day']
        for day in range(planning_horizon_days):
            daily_machine_load = []
            for item_id in all_item_ids:
                operation_time = items_data[item_id]['operation_time_minutes']
                # LinearExpr: produced_qty * operation_time, only if assigned to this machine
                # We create an intermediate variable for the production time of an item on a specific machine
                prod_time_on_machine = model.NewIntVar(0, available_time, f"prod_time_{item_id}_m{machine_id}_d{day}")
                model.AddMultiplicationEquality(prod_time_on_machine, [produced_qty[item_id, day], assign_item_day_to_machine[item_id, day, machine_id]])
                
                # To correctly link this, we need to ensure that if not assigned, prod_time_on_machine is 0.
                # And if assigned, it's produced_qty * operation_time.
                # This is tricky because assign_item_day_to_machine is already linked to produced_qty > 0.
                # Let's simplify: the sum is over items *assigned* to the machine.
                # We need to express: (produced_qty[i,d] * operation_time[i]) * assign_item_day_to_machine[i,d,m]
                # This requires multiplication of an IntVar (produced_qty) and a BoolVar (assign).
                # CP-SAT handles this by multiplying with the boolean var directly in the sum if the int var is bounded.

                # Simpler approach for now, assuming assign_item_day_to_machine correctly gates the production quantity
                # The term should only contribute if assign_item_day_to_machine[item_id, day, machine_id] is true.
                # We can create an expression for the time taken by item_id on machine_id if assigned.
                
                # Create an expression for time taken by an item if produced on this machine
                # This will be operation_time * produced_qty if assigned, else 0
                # We can use an intermediate variable for this, or rely on OnlyEnforceIf logic if possible.

                # Let's use the direct sum with multiplication, CP-SAT should handle it.
                # term = produced_qty[item_id, day] * items_data[item_id]['operation_time_minutes']
                # daily_machine_load.append(term * assign_item_day_to_machine[item_id, day, machine_id])
                # The above direct multiplication is not directly supported for the objective/constraints in the way Addsum works.
                # Instead, create product variables
                
                # Time consumed by item 'i' on machine 'm' on day 'd'
                time_consumed_var = model.NewIntVar(0, available_time, f"time_consumed_{item_id}_{machine_id}_d{day}")
                
                # if assign_item_day_to_machine[i,d,m] is 0, time_consumed_var must be 0
                model.Add(time_consumed_var == 0).OnlyEnforceIf(assign_item_day_to_machine[item_id, day, machine_id].Not())
                
                # if assign_item_day_to_machine[i,d,m] is 1, time_consumed_var must be produced_qty[i,d] * operation_time_minutes
                # This means: (produced_qty[i,d] * operation_time_minutes) - time_consumed_var == 0
                # We need to use AddMultiplicationEquality or linearize.
                # For now, let's use a simpler form that might be less strict or rely on the assignment constraint.
                # This constraint is the most complex part of this basic model.
                # A common way is to link production to assignment:
                # produced_qty[i,d] <= M * sum(assign_item_day_to_machine[i,d,m] for m in machines)
                # And then the capacity constraint:
                # Sum ( produced_qty_on_machine[i,d,m] * op_time[i] ) <= capacity[m]
                # Where produced_qty_on_machine[i,d,m] is another variable.
                
                # Let's try to stick to the formulation: Sum(produced_qty[i, day] * op_time[i] * assign[i,day,m] for i in items)
                # This needs an intermediate variable for the product of (produced_qty * assign)
                
                # Intermediate variable: production_on_machine[item_id, day, machine_id]
                # This var will be equal to produced_qty[item_id, day] if assign_item_day_to_machine is true, else 0.
                prod_on_machine_var = model.NewIntVar(0, int(max_possible_daily_production) if max_possible_daily_production > 0 else 1000, f"prod_on_{item_id}_{machine_id}_d{day}")
                
                # If assigned, prod_on_machine_var == produced_qty[item_id, day]
                model.Add(prod_on_machine_var == produced_qty[item_id, day]).OnlyEnforceIf(assign_item_day_to_machine[item_id, day, machine_id])
                # If not assigned, prod_on_machine_var == 0
                model.Add(prod_on_machine_var == 0).OnlyEnforceIf(assign_item_day_to_machine[item_id, day, machine_id].Not())
                
                daily_machine_load.append(prod_on_machine_var * items_data[item_id]['operation_time_minutes'])

            model.Add(sum(daily_machine_load) <= available_time)

    # --- Objective Function ---
    objective_terms = []
    for item_id in all_item_ids:
        for day in range(planning_horizon_days):
            objective_terms.append(inventory[item_id, day] * items_data[item_id]['inventory_cost_euros'])
            # The problem asks to minimize inventory and production costs (using inventory_cost_euros as a proxy for production cost for now)
            objective_terms.append(produced_qty[item_id, day] * items_data[item_id]['inventory_cost_euros'])


    model.Minimize(sum(objective_terms))

    # --- Solve ---
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30.0
    status = solver.Solve(model)

    # --- Output Results ---
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print(f"Solution found in {solver.WallTime()} seconds")
        print(f"Objective value: {solver.ObjectiveValue()}")
        for item_id in all_item_ids:
            print(f"\nItem: {item_id}")
            for day in range(planning_horizon_days):
                prod_val = solver.Value(produced_qty[item_id, day])
                inv_val = solver.Value(inventory[item_id, day])
                if prod_val > 0: # Only print if there's production
                    print(f"  Day {day}: Produced = {prod_val}, Inventory = {inv_val}")
                    for m_id in all_machine_ids:
                        if solver.Value(assign_item_day_to_machine[item_id, day, m_id]):
                            print(f"    Assigned to Machine: {m_id}")
            if sum(solver.Value(produced_qty[item_id,d]) for d in range(planning_horizon_days)) == 0:
                 print(f"  No production planned for this item during the horizon.")


    elif status == cp_model.INFEASIBLE:
        print("Model is infeasible.")
        # You can try to find the reasons for infeasibility here if needed
        # For example, by using solver.SufficientAssumptionsForInfeasibility()
    elif status == cp_model.MODEL_INVALID:
        print("Model is invalid.")
    else:
        print(f"Solver status: {status}")
        
    return solver, produced_qty, inventory, assign_item_day_to_machine


if __name__ == "__main__":
    excel_file_path = 'turning-data.xlsx'  # Make sure this file is in the root or provide correct path
    parsed_data_tuple = load_production_data(excel_file_path)

    if parsed_data_tuple and parsed_data_tuple[0] and parsed_data_tuple[1]: # Check if items and machines data were loaded
        items_data, machines_data, raw_materials_data, tools_data = parsed_data_tuple
        
        # Test Basic Model (optional)
        # planning_horizon_days_basic = 3
        # initial_inventory_basic = {item_id: 0 for item_id in items_data.keys()}
        # print(f"\n--- Running Basic CP-SAT Model for {planning_horizon_days_basic} days ---")
        # build_and_solve_basic_model(parsed_data_tuple, planning_horizon_days_basic, initial_inventory_basic)

        print("\n" + "="*50)
        print("--- Running Advanced CP-SAT Model ---")
        print("="*50 + "\n")

        # Parameters for the advanced model
        planning_horizon_days_advanced = 7 # For inventory tracking
        N_POTENTIAL_BATCHES_PER_ITEM = 3   # Max number of batches an item can be split into
        max_lot_size_multiplier = 10 # Max lot size can be X times total demand for the horizon.
                                     # This needs to be carefully set. For 7 days, maybe 7*avg_daily_demand
        solver_time_limit_seconds = 60


        # Use a subset of items and machines for focused testing if needed
        # items_to_test = list(items_data.keys())[:2] # Test with first 2 items
        # machines_to_test = list(machines_data.keys())[:1] # Test with first 1 machine
        # test_items_data = {k: items_data[k] for k in items_to_test if k in items_data}
        # test_machines_data = {k: machines_data[k] for k in machines_to_test if k in machines_data}
        # test_parsed_data = (test_items_data, test_machines_data, raw_materials_data, tools_data)
        # initial_inventory_advanced = {item_id: 0 for item_id in test_items_data.keys()}


        initial_inventory_advanced = {item_id: 0 for item_id in items_data.keys()}


        build_and_solve_advanced_model(
            parsed_data_tuple, # Use full data for now
            planning_horizon_days_advanced,
            initial_inventory_advanced,
            N_POTENTIAL_BATCHES_PER_ITEM,
            max_lot_size_multiplier,
            solver_time_limit_seconds
        )

    else:
        print("\nFailed to load data, so CP-SAT model execution is skipped.")
        if not parsed_data_tuple or not parsed_data_tuple[0]:
            print("Reason: No items data loaded.")
        if not parsed_data_tuple or not parsed_data_tuple[1]:
            print("Reason: No machines data loaded.")


def build_and_solve_advanced_model(parsed_data, 
                                   planning_horizon_days, 
                                   initial_inventory_dict,
                                   n_potential_batches_per_item,
                                   max_lot_size_multiplier,
                                   time_limit_seconds=60):
    """
    Builds and solves an advanced CP-SAT model with batch production,
    sequence-dependent changeovers, and detailed costs.
    """
    items_data, machines_data, raw_materials_data, tools_data = parsed_data
    model = cp_model.CpModel()

    all_item_ids = list(items_data.keys())
    all_machine_ids = list(machines_data.keys())
    
    if not all_item_ids:
        print("Error: No items to schedule.")
        return None
    if not all_machine_ids:
        print("Error: No machines available for scheduling.")
        return None

    planning_horizon_minutes = planning_horizon_days * 24 * 60

    # --- Decision Variables for Batch Production ---
    # For each item 'i' and potential batch 'b'
    lot_size_vars = {}          # (item_id, batch_idx) -> int: quantity in the batch
    task_active_vars = {}       # (item_id, batch_idx) -> bool: if this batch is produced
    # assigned_machine_vars = {}  # (item_id, batch_idx) -> int: machine_id (This will be part of interval creation)
    start_time_vars = {}        # (item_id, batch_idx) -> int: start time in minutes from horizon start
    duration_vars = {}          # (item_id, batch_idx) -> int: duration in minutes
    end_time_vars = {}          # (item_id, batch_idx) -> int: end time in minutes
    interval_vars = {}          # (item_id, batch_idx, machine_id) -> OptionalIntervalVar
                                # OR interval_vars_per_machine[machine_id] = list of OptionalIntervalVar for tasks on that machine

    # Inventory variables (daily)
    inventory_vars = {}         # (item_id, day) -> int: quantity at the end of the day

    # Maximum possible lot size for an item (e.g., total demand over horizon * multiplier)
    max_lot_sizes = {}
    for item_id in all_item_ids:
        total_demand_horizon = sum(items_data[item_id]['daily_demand'] for _ in range(planning_horizon_days))
        max_lot_sizes[item_id] = int(total_demand_horizon * max_lot_size_multiplier)
        if max_lot_sizes[item_id] == 0 : # Handle items with zero demand, allow some production
            max_lot_sizes[item_id] = 100 # Default small max if no demand (e.g. for new products)


    task_machine_assignment_literals = {} # (item_id, batch_idx, machine_id) -> BoolVar, true if task (i,b) is on machine m

    for item_id in all_item_ids:
        op_time_minutes = items_data[item_id]['operation_time_minutes']
        if op_time_minutes <= 0: # Skip items with non-positive operation time
            print(f"Warning: Item {item_id} has non-positive operation time ({op_time_minutes}). Skipping this item for scheduling batches.")
            continue

        for batch_idx in range(n_potential_batches_per_item):
            task_key = (item_id, batch_idx)
            
            lot_size_vars[task_key] = model.NewIntVar(0, max_lot_sizes[item_id], f"lot_{item_id}_b{batch_idx}")
            task_active_vars[task_key] = model.NewBoolVar(f"active_{item_id}_b{batch_idx}")
            
            start_time_vars[task_key] = model.NewIntVar(0, planning_horizon_minutes, f"start_{item_id}_b{batch_idx}")
            # Duration is not fixed if lot size is 0. If active, duration = lot_size * op_time. If not, duration = 0.
            # Max duration can be max_lot_size * op_time
            max_duration = max_lot_sizes[item_id] * op_time_minutes
            actual_duration_var = model.NewIntVar(0, max_duration if max_duration > 0 else 1, f"actual_duration_{item_id}_b{batch_idx}") # Ensure positive upper bound

            # Link task_active to lot_size
            model.Add(lot_size_vars[task_key] > 0).OnlyEnforceIf(task_active_vars[task_key])
            model.Add(lot_size_vars[task_key] == 0).OnlyEnforceIf(task_active_vars[task_key].Not())

            # Calculate actual duration: duration_vars = lot_size_vars * op_time_minutes if active, else 0
            # Need to use AddMultiplicationEquality for lot_size * op_time
            # We also need to make it zero if not active.
            # Create an intermediate var for lot_size * op_time for active tasks
            intermediate_active_duration = model.NewIntVar(0, max_duration if max_duration > 0 else 1, f"int_active_dur_{item_id}_b{batch_idx}")
            if op_time_minutes > 0 : # Only add if op_time is valid
                 model.AddMultiplicationEquality(intermediate_active_duration, [lot_size_vars[task_key], model.NewConstant(op_time_minutes)])
            else: # Should not happen due to check above, but as safeguard
                 model.Add(intermediate_active_duration == 0)

            model.Add(actual_duration_var == intermediate_active_duration).OnlyEnforceIf(task_active_vars[task_key])
            model.Add(actual_duration_var == 0).OnlyEnforceIf(task_active_vars[task_key].Not())
            duration_vars[task_key] = actual_duration_var
            
            end_time_vars[task_key] = model.NewIntVar(0, planning_horizon_minutes, f"end_{item_id}_b{batch_idx}")
            model.Add(end_time_vars[task_key] == start_time_vars[task_key] + duration_vars[task_key])

            # Literals for machine assignment for this task (item_id, batch_idx)
            machine_literals_for_task = []
            for machine_id in all_machine_ids:
                lit = model.NewBoolVar(f"assign_{item_id}_b{batch_idx}_to_m{machine_id}")
                task_machine_assignment_literals[item_id, batch_idx, machine_id] = lit
                machine_literals_for_task.append(lit)
                
                # Create OptionalIntervalVar for this task on this specific machine
                interval_vars[item_id, batch_idx, machine_id] = model.NewOptionalIntervalVar(
                    start_time_vars[task_key],
                    duration_vars[task_key], # This duration must be 0 if task_active is false & lit is false.
                                             # The actual_duration_var handles the task_active part.
                                             # If lit is false, this interval should not be considered.
                    end_time_vars[task_key],
                    lit, # This interval is active if this task (i,b) is assigned to this machine 'm'
                    f"interval_{item_id}_b{batch_idx}_m{machine_id}"
                )

            # A task, if active, must be assigned to exactly one machine.
            model.Add(sum(machine_literals_for_task) == 1).OnlyEnforceIf(task_active_vars[task_key])
            # If task is not active, it's not assigned to any machine (all literals must be false).
            model.Add(sum(machine_literals_for_task) == 0).OnlyEnforceIf(task_active_vars[task_key].Not())


    # --- Inventory Variables ---
    for item_id in all_item_ids:
        for day in range(planning_horizon_days):
            # Max inventory could be total demand over horizon or more.
            max_inv_val = sum(items_data[item_id]['daily_demand'] * planning_horizon_days for _ in range(planning_horizon_days)) * 2
            if max_inv_val == 0: max_inv_val = 10000 # Default if no demand
            inventory_vars[item_id, day] = model.NewIntVar(0, max_inv_val if max_inv_val > 0 else 10000, f"inv_{item_id}_d{day}")


    # --- Constraints ---

    # 1. Machine Capacity: AddNoOverlap
    # Collect all intervals assigned to each machine.
    machine_specific_intervals = {m_id: [] for m_id in all_machine_ids}
    for (i_id, b_idx, m_id), interval_v in interval_vars.items():
        if i_id not in items_data or items_data[i_id]['operation_time_minutes'] <=0:
            continue
        machine_specific_intervals[m_id].append(interval_v)

    for machine_id, intervals_on_machine in machine_specific_intervals.items():
        if intervals_on_machine:
            model.AddNoOverlap(intervals_on_machine)
        # else:
            # print(f"Note: No intervals assigned to machine {machine_id} prior to solving. This is normal if items don't use it.")


    # 2. Inventory Balance
    produced_on_day_vars = {} # (item_id, day) -> sum of lot_sizes ending on that day
    for item_id in all_item_ids:
        if item_id not in items_data or items_data[item_id]['operation_time_minutes'] <=0: # Skip items not processed
            for day in range(planning_horizon_days): # Still need inventory vars for them
                 prev_inventory = initial_inventory_dict.get(item_id, 0) if day == 0 else inventory_vars[item_id, day - 1]
                 daily_demand_val = items_data[item_id].get('daily_demand', 0)
                 model.Add(inventory_vars[item_id, day] == prev_inventory - int(daily_demand_val))
                 model.Add(inventory_vars[item_id, day] >= 0) # Ensure non-negative inventory
            continue

        for day in range(planning_horizon_days):
            batches_ending_on_day_d = []
            for batch_idx in range(n_potential_batches_per_item):
                task_key = (item_id, batch_idx)
                
                # end_day_var = model.NewIntVar(0, planning_horizon_days -1, f"end_day_{item_id}_b{batch_idx}")
                # model.AddDivisionEquality(end_day_var, end_time_vars[task_key], 24*60) # end_day = end_time // (mins_in_day)
                # Note: AddDivisionEquality requires dividend to be non-negative. End_time is always >=0.
                # For CP-SAT, it's often better to create an expression for end_day.
                # end_time_vars[task_key] / (24*60) can be tricky with integer division if not careful.
                # A robust way for "falls on day d": start_of_day_d <= end_time < start_of_day_d_plus_1
                
                # Create a boolean var: is_batch_ending_on_day_d
                is_ending_on_day_d = model.NewBoolVar(f"is_end_{item_id}_b{batch_idx}_d{day}")
                
                # Condition: (day * 24 * 60) < end_time_vars[task_key] <= ((day + 1) * 24 * 60)
                # And task must be active. If not active, its contribution is 0.
                # end_time_vars already correctly linked to 0 if task_active is false.
                
                # Lower bound: end_time > day * minutes_in_day
                model.Add(end_time_vars[task_key] > day * 24 * 60).OnlyEnforceIf(is_ending_on_day_d)
                # Upper bound: end_time <= (day + 1) * minutes_in_day
                model.Add(end_time_vars[task_key] <= (day + 1) * 24 * 60).OnlyEnforceIf(is_ending_on_day_d)

                # If not ending on day d, at least one of these must be violated.
                # This requires careful formulation for the Not() case.
                # Alternative: create an auxiliary variable for the day number of the end time.
                end_day_of_batch = model.NewIntVar(0, planning_horizon_days, f"end_day_of_{item_id}_b{batch_idx}")
                # end_day = (end_time - 1) // (24*60) if end_time > 0, else can be -1 or 0 depending on convention.
                # If end_time is 0 (inactive task), end_day should not matter or be some value that doesn't match.
                # For active tasks: end_day = floor((end_time_vars[task_key] - 1) / (24 * 60))
                # This ensures that if a task finishes exactly at midnight (e.g. end of day 0), it counts for day 0.
                # If end_time_vars[task_key] is 0 (inactive), then (0-1)//(24*60) = -1, which is fine.
                
                # Simpler for now: if a task is active, its end_day_of_batch is calculated.
                # We need to handle end_time = 0 for inactive tasks carefully.
                # Let end_day_of_batch be planning_horizon_days + 1 if not active (won't match any day)
                # This requires task_active_vars to gate it.
                
                # Let's use a direct comparison with day for is_ending_on_day_d
                # To make is_ending_on_day_d true if end_day_of_batch == day
                # For active tasks:
                # end_day_of_batch = (end_time_vars[task_key] -1) // (24*60)
                # model.Add(end_day_of_batch == day).OnlyEnforceIf(is_ending_on_day_d).OnlyEnforceIf(task_active_vars[task_key])
                # model.Add(end_day_of_batch != day).OnlyEnforceIf(is_ending_on_day_d.Not()).OnlyEnforceIf(task_active_vars[task_key])
                # model.Add(is_ending_on_day_d == False).OnlyEnforceIf(task_active_vars[task_key].Not()) # If not active, not ending today

                # Revisit: is_ending_on_day_d
                # True if task_active AND (day * 24 * 60) < end_time_vars[task_key] <= ((day + 1) * 24 * 60)
                # Let's define active_and_ending_today = model.NewBoolVar()
                # active_and_ending_today implies task_active_vars[task_key]
                # active_and_ending_today implies (day * 24 * 60) < end_time_vars[task_key]
                # active_and_ending_today implies end_time_vars[task_key] <= ((day + 1) * 24 * 60)
                # This is equivalent to:
                # model.AddBoolOr([task_active_vars[task_key].Not(), is_ending_on_day_d.Not()]) # if not active, then not active_and_ending_today
                # model.Add(end_time_vars[task_key] > day * 24 * 60).OnlyEnforceIf(is_ending_on_day_d)
                # model.Add(end_time_vars[task_key] <= (day + 1) * 24 * 60).OnlyEnforceIf(is_ending_on_day_d)
                # And if is_ending_on_day_d is false, then one of the above must be false.
                # This is tricky with OnlyEnforceIf.

                # A simpler approach for produced_on_day[i,d]:
                # Sum (lot_size[i,b] * literal_is_task_active[i,b] * literal_ends_on_day_d[i,b])
                # Where literal_ends_on_day_d is true if end_time_vars[i,b] is in (day*mins, (day+1)*mins]
                
                # Create boolean: task (i,b) ends on day 'd'
                task_ends_on_day_d_lit = model.NewBoolVar(f"task_{item_id}_{batch_idx}_ends_d{day}")
                # If task is not active, it cannot end on day d.
                model.AddImplication(task_ends_on_day_d_lit, task_active_vars[task_key])
                
                # If task is active:
                # task_ends_on_day_d_lit is true if start_of_day_d < end_time <= start_of_day_d_plus_1
                # Note: end_time can be 0 if duration is 0 (e.g. lot size is 0, but task somehow active).
                # Duration is already 0 if task_active is false.
                # End_time is start_time + duration. If task_active is false, duration is 0, end_time=start_time.
                # This means an inactive task's end_time could fall in a day window if its start_time does.
                # The task_active_vars[task_key] will gate the contribution of lot_size.

                # If task_ends_on_day_d_lit is true:
                model.Add(end_time_vars[task_key] > day * 24 * 60).OnlyEnforceIf(task_ends_on_day_d_lit)
                model.Add(end_time_vars[task_key] <= (day + 1) * 24 * 60).OnlyEnforceIf(task_ends_on_day_d_lit)
                
                # If task_ends_on_day_d_lit is false (and task is active):
                # then end_time_vars[task_key] <= day * 24 * 60 OR end_time_vars[task_key] > (day+1) * 24*60
                # This is equivalent to:
                # (end_time_vars[task_key] <= day * 24 * 60) OR (end_time_vars[task_key] - (day+1)*24*60 > 0)
                # Create two bools for these conditions and use AddBoolOr.
                cond1 = model.NewBoolVar(f"cond1_{item_id}_{batch_idx}_d{day}")
                model.Add(end_time_vars[task_key] <= day * 24 * 60).OnlyEnforceIf(cond1)
                model.Add(end_time_vars[task_key] > day * 24 * 60).OnlyEnforceIf(cond1.Not())
                
                cond2 = model.NewBoolVar(f"cond2_{item_id}_{batch_idx}_d{day}")
                model.Add(end_time_vars[task_key] > (day + 1) * 24 * 60).OnlyEnforceIf(cond2)
                model.Add(end_time_vars[task_key] <= (day + 1) * 24 * 60).OnlyEnforceIf(cond2.Not())
                
                model.AddBoolOr([cond1, cond2]).OnlyEnforceIf(task_ends_on_day_d_lit.Not()).OnlyEnforceIf(task_active_vars[task_key])
                
                # Contribution to production on day d: lot_size * task_active * task_ends_on_day_d_lit
                # Since task_active is implied by task_ends_on_day_d_lit, we can simplify.
                # If task_ends_on_day_d_lit is true, lot_size must be >0 (due to task_active implication).
                
                # Create an intermediate variable for the production of this batch on this day
                batch_prod_on_day_d = model.NewIntVar(0, max_lot_sizes[item_id], f"bprod_{item_id}_b{batch_idx}_d{day}")
                model.Add(batch_prod_on_day_d == lot_size_vars[task_key]).OnlyEnforceIf(task_ends_on_day_d_lit)
                model.Add(batch_prod_on_day_d == 0).OnlyEnforceIf(task_ends_on_day_d_lit.Not())
                batches_ending_on_day_d.append(batch_prod_on_day_d)

            # Sum of all batch productions ending on day d for item_id
            daily_production_var = model.NewIntVar(0, max_lot_sizes[item_id] * n_potential_batches_per_item, f"prod_{item_id}_d{day}")
            model.Add(daily_production_var == sum(batches_ending_on_day_d))
            produced_on_day_vars[item_id, day] = daily_production_var
            
            # Inventory balance constraint
            prev_inventory = initial_inventory_dict.get(item_id, 0) if day == 0 else inventory_vars[item_id, day - 1]
            daily_demand_val = items_data[item_id].get('daily_demand', 0) # Already an int from parser
            
            model.Add(inventory_vars[item_id, day] == prev_inventory + produced_on_day_vars[item_id, day] - int(daily_demand_val))
            model.Add(inventory_vars[item_id, day] >= 0) # Ensure non-negative inventory


    # 3. Changeover Constraints using AddCircuit
    # For each machine, model sequence-dependent changeovers.
    # This requires defining arcs between all potential tasks (item, batch_idx) on that machine.
    # Global constants for changeover calculation (assuming they are in `items_data` or `parsed_data` top level)
    # These should ideally come from `parsed_data` if they were loaded there, or from `machines_data` if specific.
    # For now, assume they are accessible via items_data or globally.
    # The data_parser.py has TOOL_CHANGE_TIME_PER_TOOL and RAW_MATERIAL_CHANGE_TIME as global.
    # These should be passed in `parsed_data` ideally. For now, hardcode or assume access.
    # Let's assume they are part of `tools_data` or a new structure in `parsed_data`.
    # For this step, I'll use the helper `get_changeover_time` which expects them as args.
    # I'll need to fetch these from the `parsed_data` structure.
    # The data_parser puts them as global variables. This is not ideal.
    # Let's assume they are available in `items_data` or `tools_data` or passed in.
    # For now, I'll define them locally here, assuming they would be passed.
    TOOL_CHANGE_TIME_PER_TOOL_CONST = 5 # minutes, from problem desc
    RAW_MATERIAL_CHANGE_TIME_CONST = 20 # minutes, from problem desc

    total_changeover_time_cost_terms = [] # For objective function later, if needed

    for machine_id in all_machine_ids:
        tasks_on_machine = [] # List of (item_id, batch_idx) tuples for this machine
        task_to_lit_map = {} # Maps (item_id, batch_idx) to its assignment literal for this machine

        for i_id in all_item_ids:
            if i_id not in items_data or items_data[i_id]['operation_time_minutes'] <=0:
                continue
            for b_idx in range(n_potential_batches_per_item):
                # if task_active_vars[(i_id, b_idx)] is potentially true for this machine:
                # We consider all (item_id, batch_idx) that *could* be on this machine.
                # The task_machine_assignment_literals[(i_id, b_idx, machine_id)] determines if it *is* on this machine.
                tasks_on_machine.append((i_id, b_idx))
                task_to_lit_map[(i_id, b_idx)] = task_machine_assignment_literals[i_id, b_idx, machine_id]

        if not tasks_on_machine:
            continue

        num_tasks_on_machine = len(tasks_on_machine)
        if num_tasks_on_machine <= 1: # No sequence if 0 or 1 task
            continue
            
        arcs = []
        # For AddCircuit, we need a dummy node (e.g., index num_tasks_on_machine) to represent start/end of sequence on machine.
        dummy_node_idx = num_tasks_on_machine 
        
        # Map (item_id, batch_idx) to a simple integer index for AddCircuit
        task_to_node_idx = {task_key: i for i, task_key in enumerate(tasks_on_machine)}

        for i in range(num_tasks_on_machine):
            task_i_key = tasks_on_machine[i] # (item_id_i, batch_idx_i)
            task_i_node_idx = task_to_node_idx[task_i_key]
            
            # Arc from dummy to task_i: represents task_i starting the sequence
            # Literal for this arc is task_to_lit_map[task_i_key] (task is active on this machine)
            # Cost of this arc is 0 (or initial setup cost if any)
            arcs.append((dummy_node_idx, task_i_node_idx, task_to_lit_map[task_i_key]))

            # Arc from task_i to dummy: represents task_i ending the sequence
            # Literal is also task_to_lit_map[task_i_key]
            # Cost is 0
            arcs.append((task_i_node_idx, dummy_node_idx, task_to_lit_map[task_i_key]))
            
            for j in range(num_tasks_on_machine):
                if i == j:
                    continue
                task_j_key = tasks_on_machine[j] # (item_id_j, batch_idx_j)
                task_j_node_idx = task_to_node_idx[task_j_key]

                # Literal for arc task_i -> task_j
                # This arc is active if both task_i and task_j are on this machine, AND task_i is followed by task_j.
                # The AddCircuit handles the "followed by" part.
                # The literal itself should represent that this specific transition is chosen.
                # Both task_i and task_j must be active on this machine for this transition to be meaningful.
                
                # lit_i_precedes_j needs to imply that task_i is active on machine and task_j is active on machine.
                lit_i_precedes_j = model.NewBoolVar(f"arc_{task_i_key[0]}_{task_i_key[1]}_to_{task_j_key[0]}_{task_j_key[1]}_on_m{machine_id}")
                
                # This literal implies both tasks are active on this machine
                model.AddImplication(lit_i_precedes_j, task_to_lit_map[task_i_key])
                model.AddImplication(lit_i_precedes_j, task_to_lit_map[task_j_key])
                
                arcs.append((task_i_node_idx, task_j_node_idx, lit_i_precedes_j))

                # Add changeover constraint:
                # start_time[task_j] >= end_time[task_i] + changeover_time(task_i, task_j)
                # This must only be enforced if lit_i_precedes_j is true.
                
                item1_id = task_i_key[0]
                item2_id = task_j_key[0]
                
                changeover_duration = get_changeover_time(
                    item1_id, item2_id, items_data, raw_materials_data, 
                    TOOL_CHANGE_TIME_PER_TOOL_CONST, RAW_MATERIAL_CHANGE_TIME_CONST
                )
                
                # If changeover_duration is 0, constraint is start_j >= end_i, which is already handled by NoOverlap if no gap.
                # NoOverlap ensures start_j >= end_i OR start_i >= end_j.
                # AddCircuit determines the actual sequence.
                if changeover_duration > 0:
                    model.Add(start_time_vars[task_j_key] >= end_time_vars[task_i_key] + changeover_duration).OnlyEnforceIf(lit_i_precedes_j)
                else:
                    # If no changeover, NoOverlap is sufficient. We still need the arc for the circuit.
                    # We could enforce start_j >= end_i if lit_i_precedes_j is true, to be explicit.
                    model.Add(start_time_vars[task_j_key] >= end_time_vars[task_i_key]).OnlyEnforceIf(lit_i_precedes_j)

        if arcs:
            model.AddCircuit(arcs)
        else:
             print(f"Warning: No arcs for AddCircuit on machine {machine_id}. Num tasks: {num_tasks_on_machine}")


    # --- Objective Function ---
    total_machining_cost_terms = []
    total_inventory_holding_cost_terms = []

    # Machining Cost
    for item_id in all_item_ids:
        if item_id not in items_data or items_data[item_id]['operation_time_minutes'] <=0:
            continue
        for batch_idx in range(n_potential_batches_per_item):
            task_key = (item_id, batch_idx)
            # Cost is duration_vars[task_key] * (machine_cost_per_hour / 60)
            # This needs to be associated with the assigned machine.
            # We need to sum: duration_vars[task_key] * (cost_per_min_for_assigned_machine)
            
            for machine_id in all_machine_ids:
                machine_cost_per_hour = machines_data[machine_id].get('cost_per_hour', 20.0) # Default from parser
                machine_cost_per_minute = machine_cost_per_hour / 60.0
                
                # Cost term if task (i,b) is assigned to machine_id
                # cost = duration_vars[task_key] * machine_cost_per_minute
                # This needs to be multiplied by task_machine_assignment_literals[item_id, batch_idx, machine_id]
                
                cost_if_assigned = model.NewIntVar(0, int(planning_horizon_minutes * machine_cost_per_minute * 100), # Scaled cost
                                                   f"cost_{item_id}_b{batch_idx}_m{machine_id}")
                
                # If assigned to this machine: cost_if_assigned = duration_vars * cost_per_min (approx)
                # Need to use AddMultiplicationEquality if cost_per_minute is not integer.
                # Let's assume cost_per_minute can be scaled to an integer or use floating point coefficients if solver supports.
                # CP-SAT objective must be linear sum of IntVars or constants.
                # So, machine_cost_per_minute must be an integer or scaled.
                # For now, scale it: cost_per_100_minute_parts (e.g. EUR cents per minute part)
                
                # Let cost be in "cost units" e.g. cents.
                machine_cost_per_minute_cents = int(machine_cost_per_minute * 100) # Convert EUR to cents

                # Max possible duration is planning_horizon_minutes
                max_cost_val = planning_horizon_minutes * machine_cost_per_minute_cents
                
                cost_var_on_machine = model.NewIntVar(0, max_cost_val if max_cost_val > 0 else 1, f"mcost_{item_id}_b{batch_idx}_m{machine_id}")

                # cost_var_on_machine = duration_vars[task_key] * machine_cost_per_minute_cents
                model.AddMultiplicationEquality(cost_var_on_machine, [duration_vars[task_key], model.NewConstant(machine_cost_per_minute_cents)])
                
                # This cost term is only added if task is assigned to this machine
                final_machining_cost_for_task_on_machine = model.NewIntVar(0, max_cost_val if max_cost_val > 0 else 1, f"final_mcost_{item_id}_b{batch_idx}_m{machine_id}")
                model.Add(final_machining_cost_for_task_on_machine == cost_var_on_machine).OnlyEnforceIf(task_machine_assignment_literals[item_id, batch_idx, machine_id])
                model.Add(final_machining_cost_for_task_on_machine == 0).OnlyEnforceIf(task_machine_assignment_literals[item_id, batch_idx, machine_id].Not())
                
                total_machining_cost_terms.append(final_machining_cost_for_task_on_machine)

    # Inventory Holding Cost
    # inventory_cost_euros_daily_rate = inventory_cost_euros * (0.10 / 365)
    # Scale this to integer cents for the objective
    for item_id in all_item_ids:
        # inventory_cost_euros is per piece, per year (implicit from problem, or needs clarification)
        # Assuming items_data[item_id]['inventory_cost_euros'] is "per unit per year at 10% interest"
        # So, cost per unit per day = (items_data[item_id]['inventory_cost_euros'] * 0.10) / 365
        # Let's assume the 'inventory_cost_euros' in items_data is already the per-unit, per-day cost for simplicity
        # Or, if it's per piece (value of piece), then apply rate: value * (annual_rate / days_in_year)
        
        # The problem states: "inventory_cost_euros": round(random.uniform(1.5, 3.0), 2) in data_parser
        # This is likely a placeholder value. Let's assume it's per piece per day for now.
        # And objective: total_inventory_holding_cost = Sum(inventory[i, d] * parsed_data['items_data'][i]['inventory_cost_euros'] * (0.10 / 365) for all i,d)
        # This implies 'inventory_cost_euros' is the value of the item.
        
        item_value_euros = items_data[item_id].get('inventory_cost_euros', 0) # This is the value from random assignment.
        # Daily holding cost per unit = item_value_euros * (0.10 / 365)
        # Convert to cents: item_value_cents * (10 / 36500) = item_value_cents / 3650
        daily_holding_cost_per_unit_cents_numerator = int(item_value_euros * 100) # item_value_cents
        daily_holding_cost_per_unit_cents_denominator = 365 # Assuming 10% annual rate, so daily rate is (value * 0.10)/365
                                                            # If objective is value * rate_factor, then rate_factor = (0.10/365)
                                                            # Cost = inventory_vars * item_value_cents * 10 / 365
                                                            # To make it integer, multiply objective by 365.
                                                            # Objective term: inventory_vars * item_value_cents * 10

        # For precision with integer arithmetic, it's common to scale the objective function.
        # Let's use a large scaling factor for costs, e.g., 10000.
        # Objective = sum ( inventory_vars * item_value_euros * 100 * 10 ) -> sum (inv_var * item_value_cents * 10)
        # This is inventory_vars * (item_value_euros * 1000) effectively.
        # The prompt specifies `* (0.10 / 365)`. Let this be `RATE_FACTOR`.
        # Cost = inventory_var * item_value_euro * RATE_FACTOR
        # Cost_cents = inventory_var * item_value_euro * RATE_FACTOR * 100
        # To keep it integer, we can multiply the entire objective by 365 * 100 (for example)
        # Then this term becomes: inventory_var * item_value_euro * 0.10 * 100 = inventory_var * item_value_euro * 10
        
        # Let's use a scaled integer cost for inventory holding.
        # Assume items_data[item_id]['inventory_cost_euros'] is the per-unit, per-day holding cost in EUR.
        # Convert to cents:
        daily_inv_cost_cents = int(items_data[item_id]['inventory_cost_euros'] * 100) # If it's already daily cost.
        # If it's item value, then:
        # daily_inv_cost_cents = int(items_data[item_id]['inventory_cost_euros'] * (0.10 / 365) * 10000) # Scaled by 10000 for precision
        # For now, let's use prompt: inventory_cost_euros * (0.10 / 365)
        # To make this integer, we need to be careful.
        # Let cost_coeff = items_data[item_id]['inventory_cost_euros'] * 10. (Scaled by 100 for cents, and by 0.10 for rate)
        # Then divide the final objective by 365. This is not possible with CP-SAT integer objectives.
        # So, we must ensure the coefficient is integer.
        # (value_in_cents * 10) / 365. This is still not integer.
        # We have to scale the whole objective by 365.
        # Then inventory cost term is: inventory_vars[i,d] * (value_in_cents * 10)
        
        # Let's assume 'inventory_cost_euros' is the per-unit value.
        # Cost per day = inventory_vars[item_id, day] * (items_data[item_id]['inventory_cost_euros'] * 100 * 0.10 / 365)
        # We will have to use integer scaling for the objective.
        # Let objective be scaled by 36500 (to handle 0.10 and 365 and convert to cents)
        # Original term: inventory_vars * value_eur * (0.10/365)
        # Scaled term: inventory_vars * value_eur * (0.10/365) * 36500
        #            = inventory_vars * value_eur * 10 * 100
        #            = inventory_vars * value_eur * 1000
        
        # For now, let's use a simplified integer cost, assuming 'inventory_cost_euros' is a daily holding cost in cents.
        # This part needs to be confirmed based on precise meaning of 'inventory_cost_euros'.
        # Based on data_parser, it's random(1.5, 3.0). This is likely EUR value of item.
        # So, daily holding cost in cents for item i, day d is:
        # inventory_vars[item_id, day] * int(items_data[item_id]['inventory_cost_euros'] * 100 * 0.10 / 365 * SCALE_FACTOR_OBJ)
        # Let SCALE_FACTOR_OBJ = 365 (to remove division)
        # Cost term: inventory_vars[item_id, day] * int(items_data[item_id]['inventory_cost_euros'] * 10) # value_in_eur * 100 * 0.10 = value_in_eur * 10
        
        item_value_x10_cents = int(items_data[item_id]['inventory_cost_euros'] * 10) # (value in EUR * 100 cents/EUR * 0.10 rate factor)
                                                                                # This is effectively item_value_in_EUR * 10
        
        for day in range(planning_horizon_days):
            # Cost for this item, this day: inventory_vars[item_id, day] * item_value_x10_cents
            # This term will be part of an objective that is implicitly scaled by 365.
            inv_holding_term = model.NewIntVar(0, 1000000 * item_value_x10_cents, # Max inventory * cost_coeff
                                              f"inv_cost_{item_id}_d{day}")
            model.AddMultiplicationEquality(inv_holding_term, [inventory_vars[item_id, day], model.NewConstant(item_value_x10_cents)])
            total_inventory_holding_cost_terms.append(inv_holding_term)

    # Total objective (scaled by 365 for inventory, and by 100 for machining costs to be in cents)
    # All machining costs are already in cents.
    # Inventory costs are (value_eur * 10), effectively (value_cents * 0.10).
    # If we want the objective in a consistent unit (e.g. "scaled cents days")
    # Machining: cost_cents * duration_minutes.
    # Inventory: inv_level * value_cents_annual_rate_scaled_by_365_and_10_percent.
    # The scaling needs to be consistent or documented.
    # For now, objective is sum of all terms as calculated.
    # It represents: sum(machining_cost_in_cents) + sum(inventory_holding_cost_scaled)
    # where inventory_holding_cost_scaled = inventory_level * item_value_in_EUR * 10
    # This means the inventory part is implicitly (cost per year / 36.5) approx.
    # The prompt asks for (cost_per_hour / 60.0) for machining and (value * 0.10 / 365) for inventory.
    # My machining cost is: duration_minutes * cost_per_hour / 60 * 100 (in cents)
    # My inventory cost is: inventory_level * value_eur * 0.10 * 100 (scaled by 365, in cents)
    # So the relative weighting should be correct if I sum these terms.
    
    total_objective_terms = total_machining_cost_terms + total_inventory_holding_cost_terms
    if not total_objective_terms:
        print("Warning: No terms in the objective function. Model might be trivial or misconfigured.")
        # Add a dummy objective to prevent errors if needed, though CP-SAT might handle empty sum.
        model.Minimize(model.NewConstant(0)) # Minimize a constant if no terms.
    else:
        model.Minimize(sum(total_objective_terms))


    # --- Solve ---
    solver = cp_model.CpSolver()
    solver.parameters.log_search_progress = True # Good for seeing progress
    solver.parameters.max_time_in_seconds = float(time_limit_seconds)
    
    print(f"\nStarting solver for advanced model (time limit: {time_limit_seconds}s)...")
    status = solver.Solve(model)

    # --- Output Results ---
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print(f"\nSolution found in {solver.WallTime():.2f} seconds")
        print(f"Objective value: {solver.ObjectiveValue():.2f}" if status == cp_model.OPTIMAL else "Feasible solution (no objective yet or not optimal)")
        
        print("\n--- Production Schedule (Active Batches) ---")
        for item_id in all_item_ids:
            if item_id not in items_data or items_data[item_id]['operation_time_minutes'] <=0:
                continue
            for batch_idx in range(n_potential_batches_per_item):
                task_key = (item_id, batch_idx)
                if solver.Value(task_active_vars[task_key]):
                    lot_sz = solver.Value(lot_size_vars[task_key])
                    s_time = solver.Value(start_time_vars[task_key])
                    e_time = solver.Value(end_time_vars[task_key])
                    dur = solver.Value(duration_vars[task_key])
                    
                    assigned_m = "None"
                    for machine_id in all_machine_ids:
                        if solver.Value(task_machine_assignment_literals[item_id, batch_idx, machine_id]):
                            assigned_m = machine_id
                            break
                    
                    print(f"  Item {item_id}, Batch {batch_idx}: Lot Size = {lot_sz}, Machine = {assigned_m}, "
                          f"Start = {s_time} min (Day {s_time//(24*60)}), End = {e_time} min (Day {e_time//(24*60)}), Duration = {dur} min")

        # Print Inventory (once implemented)
        # print("\n--- Inventory Levels (End of Day) ---")
        # for item_id in all_item_ids:
        #     print(f"  Item {item_id}:")
        #     for day in range(planning_horizon_days):
        #         inv_val = solver.Value(inventory_vars[item_id, day])
        #         print(f"    Day {day}: {inv_val}")

    elif status == cp_model.MODEL_INVALID:
        print("Model is invalid. Validating model constraints...")
        print(model.Validate()) # Print details about the invalidity
    elif status == cp_model.INFEASIBLE:
        print("Model is infeasible.")
        # To help debug infeasibility:
        # print("Sufficient assumptions for infeasibility (may take time):")
        # print(solver.SufficientAssumptionsForInfeasibility())
    else:
        print(f"Solver status: {status} (No solution found or error)")

    return solver #, other result vars if needed

# Global constants from data_parser, if needed at this level (though better passed via parsed_data)
# TOOL_CHANGE_TIME_PER_TOOL = 5
# RAW_MATERIAL_CHANGE_TIME = 20
# WORKING_DAYS_PER_YEAR = 52 * 5
# These should be accessed from parsed_data to keep solver self-contained with its inputs.
# For example: parsed_data.tool_change_time_constant

# Helper function to get changeover time (to be used later)
def get_changeover_time(item1_id, item2_id, items_data, raw_materials_data, tool_change_const, raw_material_change_const):
    if item1_id == item2_id:
        return 0

    tool_change_time = 0
    material_change_time = 0

    tools1_ids = set(items_data[item1_id]['required_tools'])
    tools2_ids = set(items_data[item2_id]['required_tools'])
    
    # Symmetric difference for tool swaps
    num_tool_swaps = len(tools1_ids.symmetric_difference(tools2_ids))
    tool_change_time = num_tool_swaps * tool_change_const

    if items_data[item1_id]['raw_material_id'] != items_data[item2_id]['raw_material_id']:
        material_change_time = raw_material_change_const
    
    return tool_change_time + material_change_time
