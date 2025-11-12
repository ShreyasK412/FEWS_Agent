"""
Convert ipcFic_data.csv to the format expected by the system.
Maps FEWS NET IPC data format to: date, admin2, ipc_phase, population_affected
"""
import pandas as pd
from pathlib import Path
from typing import Optional

def convert_ipc_data(
    input_file: str = "ipcFic_data.csv",
    output_file: Optional[str] = None,
    scenario_filter: Optional[str] = "CS",  # CS=Current Situation, ML1=Near Term, ML2=Medium Term, None=All
    country_filter: str = "Ethiopia"
):
    """
    Convert FEWS NET IPC data to system format.
    
    Args:
        input_file: Path to ipcFic_data.csv
        output_file: Output path (default: data/raw/ipc/ethiopia/ipc_phases.csv)
        scenario_filter: Filter by scenario ('CS', 'ML1', 'ML2', or None for all)
        country_filter: Filter by country (default: 'Ethiopia')
    """
    input_path = Path(input_file)
    
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return None
    
    print(f"📖 Reading IPC data from: {input_path}")
    df = pd.read_csv(input_path)
    
    print(f"   Original records: {len(df)}")
    
    # Filter by country
    if 'country' in df.columns:
        df = df[df['country'] == country_filter].copy()
        print(f"   After country filter ({country_filter}): {len(df)}")
    
    # Filter by scenario if specified
    if scenario_filter and 'scenario' in df.columns:
        df = df[df['scenario'] == scenario_filter].copy()
        print(f"   After scenario filter ({scenario_filter}): {len(df)}")
    
    # Map columns
    required_mappings = {
        'date': 'projection_start',
        'admin2': 'geographic_unit_name',
        'ipc_phase': 'value'
    }
    
    # Check required columns exist
    missing_cols = [src for src in required_mappings.values() if src not in df.columns]
    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        return None
    
    # Drop rows with missing IPC phase values
    df = df.dropna(subset=['value']).copy()
    print(f"   After removing missing IPC phases: {len(df)}")
    
    # Create output dataframe
    output_df = pd.DataFrame()
    output_df['date'] = pd.to_datetime(df['projection_start']).dt.strftime('%Y-%m-%d')
    output_df['admin2'] = df['geographic_unit_name'].str.strip()
    output_df['ipc_phase'] = df['value'].astype(int)
    
    # Add population_affected if available (optional)
    if 'population_affected' in df.columns:
        output_df['population_affected'] = df['population_affected']
    else:
        output_df['population_affected'] = None
        print("   ⚠️  No 'population_affected' column found (optional)")
    
    # Remove duplicates (same date, admin2, ipc_phase)
    original_len = len(output_df)
    output_df = output_df.drop_duplicates(subset=['date', 'admin2', 'ipc_phase'])
    if len(output_df) < original_len:
        print(f"   Removed {original_len - len(output_df)} duplicate records")
    
    # Sort by date and region
    output_df = output_df.sort_values(['date', 'admin2']).reset_index(drop=True)
    
    # Determine output path
    if output_file is None:
        output_dir = Path("data/raw/ipc/ethiopia")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "ipc_phases.csv"
    else:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save CSV
    output_df.to_csv(output_path, index=False)
    
    print(f"\n✅ Converted and saved to: {output_path}")
    print(f"   Total records: {len(output_df)}")
    print(f"   Date range: {output_df['date'].min()} to {output_df['date'].max()}")
    print(f"   Unique regions: {output_df['admin2'].nunique()}")
    print(f"   IPC phases: {sorted(output_df['ipc_phase'].unique())}")
    
    # Show sample
    print(f"\n   Sample data:")
    print(output_df.head(10).to_string(index=False))
    
    return output_df


if __name__ == "__main__":
    import sys
    
    scenario = "CS"  # Default to Current Situation
    if len(sys.argv) > 1:
        scenario = sys.argv[1]  # Can be "CS", "ML1", "ML2", or "all"
    
    if scenario.lower() == "all":
        scenario = None
    
    print("="*80)
    print("CONVERTING IPC DATA")
    print("="*80)
    print(f"Scenario filter: {scenario or 'All scenarios'}")
    print()
    
    convert_ipc_data(scenario_filter=scenario)

