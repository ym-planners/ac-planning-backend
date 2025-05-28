import pandas as pd
import random
import io # For BytesIO
from google.cloud import storage

# Global parameters (can be adjusted or moved to a config file later)
TOOL_CHANGE_TIME_PER_TOOL = 5  # minutes
RAW_MATERIAL_CHANGE_TIME = 20  # minutes
WORKING_DAYS_PER_YEAR = 52 * 5 # Default, can be overridden by data if needed

def load_production_data(bucket_name: str, blob_name: str):
    """
    Loads and parses production data from a blob in Google Cloud Storage.
    Returns a 5-tuple: (items_data, machines_data, raw_materials_data, tools_data, global_constants_dict)
    """
    print(f"Attempting to load data from GCS: gs://{bucket_name}/{blob_name}")
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        # Download blob content into an in-memory bytes buffer
        in_memory_file = io.BytesIO()
        blob.download_to_file(in_memory_file)
        in_memory_file.seek(0) # Reset stream position to the beginning
        
        print(f"Successfully downloaded gs://{bucket_name}/{blob_name} to memory.")
        # Now use this in-memory file with pd.ExcelFile
        xls = pd.ExcelFile(in_memory_file)
        
    except Exception as e: # Catch a broader range of GCS/permission errors
        print(f"Error: Failed to download or access '{blob_name}' from bucket '{bucket_name}'. Exception: {e}")
        # To ensure a consistent return type for the caller, even on error
        return None, None, None, None, {}


    items_data = {}
    machines_data = {}
    raw_materials_data = {}
    tools_data = {} # Initialize tools_data

    # Parse 'Items' tab
    if 'Items' in xls.sheet_names:
        df_items = pd.read_excel(xls, 'Items')
        for _, row in df_items.iterrows():
            item_id = row['Item ID']
            yearly_forecast = row['Yearly Forecast (pieces)']
            
            operation_time_minutes = row.get('Operation Time (min/piece)', 10) 
            if 'Operation Time (min/piece)' not in row or pd.isna(operation_time_minutes):
                print(f"Warning: 'Operation Time (min/piece)' not found or NaN for item {item_id}. Using default: 10 minutes.")
                operation_time_minutes = 10

            required_tools_str = str(row.get('Required Tools', '')) 
            required_tools = [tool.strip() for tool in required_tools_str.split(',') if tool.strip()] if required_tools_str else []

            items_data[item_id] = {
                "yearly_forecast": yearly_forecast,
                "operation_time_minutes": operation_time_minutes,
                "required_tools": required_tools,
                "raw_material_id": row['Raw Material ID'],
                "daily_demand": yearly_forecast / WORKING_DAYS_PER_YEAR, # Uses global WORKING_DAYS_PER_YEAR
                "inventory_cost_euros": round(random.uniform(1.5, 3.0), 2) # Placeholder, as per original
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
            turret_capacity = row.get('Turret Capacity', 10)
            if 'Turret Capacity' not in row or pd.isna(turret_capacity):
                 print(f"Warning: 'Turret Capacity' not found or NaN for machine {machine_id}. Using default: 10.")
                 turret_capacity = 10

            cost_per_hour = row.get('Cost per Hour (EUR)', 20.0)
            if 'Cost per Hour (EUR)' not in row or pd.isna(cost_per_hour):
                print(f"Warning: 'Cost per Hour (EUR)' not found or NaN for machine {machine_id}. Using default: 20.0 EUR.")
                cost_per_hour = 20.0
            
            machines_data[machine_id] = {
                "type": row['Machine Type'],
                "turret_capacity": int(turret_capacity), # Ensure integer
                "cost_per_hour": float(cost_per_hour), # Ensure float
                "available_minutes_per_day": 24 * 60 
            }
    else:
        print("Warning: 'Capacity' sheet not found in the Excel file.")
        
    # Parse 'Tools' tab
    if 'Tools' in xls.sheet_names:
        df_tools = pd.read_excel(xls, 'Tools')
        for _, row in df_tools.iterrows():
            tool_id = row['Tool ID']
            setup_time = row.get('Setup Time (minutes)', TOOL_CHANGE_TIME_PER_TOOL) # Use global as default
            if 'Setup Time (minutes)' not in row or pd.isna(setup_time):
                 print(f"Warning: 'Setup Time (minutes)' not found or NaN for tool {tool_id}. Using default: {TOOL_CHANGE_TIME_PER_TOOL} minutes.")
                 setup_time = TOOL_CHANGE_TIME_PER_TOOL
            tools_data[tool_id] = {
                "setup_time_minutes": setup_time
            }
    else:
        print("Warning: 'Tools' sheet not found in the Excel file. Tool setup times will use the global default.")

    # Prepare the global constants dictionary to be returned
    global_constants_dict = {
        "tool_change_time_per_tool": TOOL_CHANGE_TIME_PER_TOOL,
        "raw_material_change_time": RAW_MATERIAL_CHANGE_TIME,
        "working_days_per_year": WORKING_DAYS_PER_YEAR
    }
    
    # Check if essential data is missing and return None for data elements if so
    if not items_data or not machines_data:
        print("Error: Critical data (items or machines) could not be parsed. Returning None for data elements.")
        return None, None, None, None, global_constants_dict # Still return constants

    return items_data, machines_data, raw_materials_data, tools_data, global_constants_dict


if __name__ == "__main__":
    # This block is for local testing of GCS loading.
    # You need to have GOOGLE_APPLICATION_CREDENTIALS set in your environment,
    # or be running in an environment with default credentials (e.g., a GCE VM).
    print("Testing GCS data loader...")
    # Replace with your actual bucket and blob name for testing
    # test_bucket_name = "your-gcs-bucket-name"
    # test_blob_name = "path/to/your/turning-data.xlsx"
    
    # For the test to pass without actual GCS, we'll skip if these are not set
    import os
    test_bucket_name = os.environ.get("TEST_GCS_BUCKET_NAME")
    test_blob_name = os.environ.get("TEST_GCS_BLOB_NAME")

    if test_bucket_name and test_blob_name:
        print(f"Using GCS bucket: {test_bucket_name}, blob: {test_blob_name} from environment variables for testing.")
        items, machines, raw_materials, tools, constants = load_production_data(test_bucket_name, test_blob_name)

        if items:
            print("\n--- Items Data (First 2) ---")
            for item_id, data in list(items.items())[:2]:
                print(f"{item_id}: {data}")
        else:
            print("\nNo items data loaded from GCS test.")
        
        if machines:
            print("\n--- Machines Data (First 2) ---")
            for machine_id, data in list(machines.items())[:2]:
                print(f"{machine_id}: {data}")
        else:
            print("\nNo machines data loaded from GCS test.")

        if raw_materials:
            print("\n--- Raw Materials Data (First 2) ---")
            for rm_id, data in list(raw_materials.items())[:2]:
                print(f"{rm_id}: {data}")
        else:
            print("\nNo raw materials data loaded from GCS test.")
        
        if tools:
            print("\n--- Tools Data (First 2) ---")
            for tool_id, data in list(tools.items())[:2]:
                print(f"{tool_id}: {data}")
        else:
            print("\nNo tools data loaded from GCS test.")
            
        print("\n--- Global Constants ---")
        print(constants)
    else:
        print("\nSkipping GCS load test as TEST_GCS_BUCKET_NAME or TEST_GCS_BLOB_NAME env vars are not set.")
        print("To run this test, set these environment variables to point to your test Excel file on GCS.")
        print("Example: export TEST_GCS_BUCKET_NAME=my-bucket")
        print("         export TEST_GCS_BLOB_NAME=data/turning-data.xlsx")
