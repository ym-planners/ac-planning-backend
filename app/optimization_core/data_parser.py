import pandas as pd
import random

# Global parameters (can be adjusted or moved to a config file later)
TOOL_CHANGE_TIME_PER_TOOL = 5  # minutes
RAW_MATERIAL_CHANGE_TIME = 20  # minutes
WORKING_DAYS_PER_YEAR = 52 * 5

def load_production_data(excel_filepath):
    """
    Loads and parses production data from the specified Excel file.
    """
    try:
        xls = pd.ExcelFile(excel_filepath)
    except FileNotFoundError:
        print(f"Error: The file '{excel_filepath}' was not found.")
        return None, None, None, None

    items_data = {}
    machines_data = {}
    raw_materials_data = {}
    # Tools data will be derived from items_data for now

    # Parse 'Items' tab
    if 'Items' in xls.sheet_names:
        df_items = pd.read_excel(xls, 'Items')
        for _, row in df_items.iterrows():
            item_id = row['Item ID']
            yearly_forecast = row['Yearly Forecast (pieces)']
            
            # Operation Time: Look for 'Operation Time (min/piece)' column
            # If not found, use a placeholder and print a warning.
            operation_time_minutes = row.get('Operation Time (min/piece)', 10) 
            if 'Operation Time (min/piece)' not in row:
                print(f"Warning: 'Operation Time (min/piece)' not found for item {item_id}. Using default value: 10 minutes.")

            required_tools_str = str(row.get('Required Tools', '')) # Ensure it's a string
            required_tools = [tool.strip() for tool in required_tools_str.split(',') if tool.strip()] if required_tools_str else []


            items_data[item_id] = {
                "yearly_forecast": yearly_forecast,
                "operation_time_minutes": operation_time_minutes,
                "required_tools": required_tools,
                "raw_material_id": row['Raw Material ID'],
                "daily_demand": yearly_forecast / WORKING_DAYS_PER_YEAR,
                "inventory_cost_euros": round(random.uniform(1.5, 3.0), 2)
            }
    else:
        print("Warning: 'Items' sheet not found in the Excel file.")

    # Parse 'Raw Materials' tab
    if 'Raw Materials' in xls.sheet_names:
        df_raw_materials = pd.read_excel(xls, 'Raw Materials')
        for _, row in df_raw_materials.iterrows():
            raw_material_id = row['Material ID']
            raw_materials_data[raw_material_id] = {
                "cost_per_unit": row['Cost per Unit (EUR)']
            }
    else:
        print("Warning: 'Raw Materials' sheet not found in the Excel file.")

    # Parse 'Capacity' (Machines) tab
    if 'Capacity' in xls.sheet_names:
        df_capacity = pd.read_excel(xls, 'Capacity')
        for _, row in df_capacity.iterrows():
            machine_id = row['Machine ID']
            # Turret capacity: Look for 'Turret Capacity' column. If not, assume 10.
            turret_capacity = row.get('Turret Capacity', 10)
            if 'Turret Capacity' not in row:
                 print(f"Warning: 'Turret Capacity' not found for machine {machine_id}. Using default value: 10.")

            machines_data[machine_id] = {
                "type": row['Machine Type'],
                "turret_capacity": turret_capacity,
                "cost_per_hour": row.get('Cost per Hour (EUR)', 20.0), # Assume 20.0 if not specified
                "available_minutes_per_day": 24 * 60 # Assuming 24/7 availability for now
            }
            if 'Cost per Hour (EUR)' not in row:
                print(f"Warning: 'Cost per Hour (EUR)' not found for machine {machine_id}. Using default value: 20.0 EUR.")

    else:
        print("Warning: 'Capacity' sheet not found in the Excel file.")
        
    # Parse 'Tools' tab - this might not be directly used if tools are listed per item
    # For now, we assume tools are implicitly defined by their usage in 'Items'
    tools_data = {}
    if 'Tools' in xls.sheet_names:
        df_tools = pd.read_excel(xls, 'Tools')
        for _, row in df_tools.iterrows():
            tool_id = row['Tool ID']
            # Add any other tool-specific data if needed, e.g., lifetime, cost
            tools_data[tool_id] = {
                "setup_time_minutes": row.get('Setup Time (minutes)', TOOL_CHANGE_TIME_PER_TOOL) 
            }
            if 'Setup Time (minutes)' not in row:
                 print(f"Warning: 'Setup Time (minutes)' not found for tool {tool_id}. Using default value: {TOOL_CHANGE_TIME_PER_TOOL} minutes.")
    else:
        print("Warning: 'Tools' sheet not found in the Excel file. Tool setup times will use the global default.")


    return items_data, machines_data, raw_materials_data, tools_data

if __name__ == "__main__":
    # This block is for testing the parser directly
    # Ensure 'turning-data.xlsx' is in the root directory relative to where this script is run from
    # Or provide the correct path.
    
    # Attempt to load from a relative path assuming the script is run from the project root
    excel_file_path = 'turning-data.xlsx' 
    
    items, machines, raw_materials, tools = load_production_data(excel_file_path)

    if items:
        print("\n--- Items Data ---")
        for item_id, data in list(items.items())[:2]: # Print first 2 for brevity
            print(f"{item_id}: {data}")
    
    if machines:
        print("\n--- Machines Data ---")
        for machine_id, data in list(machines.items())[:2]:
            print(f"{machine_id}: {data}")

    if raw_materials:
        print("\n--- Raw Materials Data ---")
        for rm_id, data in list(raw_materials.items())[:2]:
            print(f"{rm_id}: {data}")
    
    if tools:
        print("\n--- Tools Data ---")
        for tool_id, data in list(tools.items())[:2]:
            print(f"{tool_id}: {data}")

    print(f"\nGlobal TOOL_CHANGE_TIME_PER_TOOL: {TOOL_CHANGE_TIME_PER_TOOL}")
    print(f"Global RAW_MATERIAL_CHANGE_TIME: {RAW_MATERIAL_CHANGE_TIME}")
