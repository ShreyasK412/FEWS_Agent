"""
Data loader for manual CSV uploads.
Loads structured data from data/raw/ directories.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime


class DataLoader:
    """Load structured data from CSV files."""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.country = "ethiopia"
    
    def load_prices(self) -> Optional[pd.DataFrame]:
        """Load price data from CSV."""
        price_file = self.data_dir / "prices" / self.country / "wfp_prices.csv"
        
        if not price_file.exists():
            print(f"⚠️  Price file not found: {price_file}")
            print("   Expected: data/raw/prices/ethiopia/wfp_prices.csv")
            return None
        
        try:
            df = pd.read_csv(price_file, comment='#')
            print(f"✅ Loaded {len(df)} price records from {price_file.name}")
            return df
        except Exception as e:
            print(f"❌ Error loading prices: {e}")
            return None
    
    def load_ipc_phases(self) -> Optional[pd.DataFrame]:
        """Load IPC phase classifications from CSV."""
        ipc_file = self.data_dir / "ipc" / self.country / "ipc_phases.csv"
        
        if not ipc_file.exists():
            print(f"⚠️  IPC file not found: {ipc_file}")
            print("   Expected: data/raw/ipc/ethiopia/ipc_phases.csv")
            print("   Format: date,admin2,ipc_phase,population_affected")
            return None
        
        try:
            df = pd.read_csv(ipc_file)
            # Normalize column names
            df.columns = df.columns.str.lower().str.strip()
            
            # Map common column name variations
            if 'region' in df.columns and 'admin2' not in df.columns:
                df['admin2'] = df['region']
            if 'phase' in df.columns and 'ipc_phase' not in df.columns:
                df['ipc_phase'] = df['phase']
            
            print(f"✅ Loaded {len(df)} IPC records from {ipc_file.name}")
            return df
        except Exception as e:
            print(f"❌ Error loading IPC data: {e}")
            return None
    
    def load_rainfall(self) -> Optional[pd.DataFrame]:
        """Load rainfall data from CSV (optional)."""
        rainfall_file = self.data_dir / "climate" / self.country / "rainfall.csv"
        
        if not rainfall_file.exists():
            return None  # Optional data
        
        try:
            df = pd.read_csv(rainfall_file)
            df.columns = df.columns.str.lower().str.strip()
            print(f"✅ Loaded {len(df)} rainfall records")
            return df
        except Exception as e:
            print(f"⚠️  Error loading rainfall: {e}")
            return None
    
    def load_conflict(self) -> Optional[pd.DataFrame]:
        """Load conflict data from CSV (optional)."""
        conflict_file = self.data_dir / "acled" / self.country / "conflict.csv"
        
        if not conflict_file.exists():
            return None  # Optional data
        
        try:
            df = pd.read_csv(conflict_file)
            df.columns = df.columns.str.lower().str.strip()
            print(f"✅ Loaded {len(df)} conflict records")
            return df
        except Exception as e:
            print(f"⚠️  Error loading conflict: {e}")
            return None
    
    def load_population(self) -> Optional[pd.DataFrame]:
        """Load population data from CSV (optional)."""
        pop_file = self.data_dir / "population" / self.country / "population.csv"
        
        if not pop_file.exists():
            return None  # Optional data
        
        try:
            df = pd.read_csv(pop_file)
            df.columns = df.columns.str.lower().str.strip()
            print(f"✅ Loaded {len(df)} population records")
            return df
        except Exception as e:
            print(f"⚠️  Error loading population: {e}")
            return None
    
    def load_all(self) -> Dict[str, pd.DataFrame]:
        """Load all available data sources."""
        print("="*80)
        print("LOADING DATA FROM CSV FILES")
        print("="*80)
        print()
        
        data = {}
        
        data['prices'] = self.load_prices()
        data['ipc'] = self.load_ipc_phases()
        data['rainfall'] = self.load_rainfall()
        data['conflict'] = self.load_conflict()
        data['population'] = self.load_population()
        
        print()
        print("="*80)
        print("DATA LOADING SUMMARY")
        print("="*80)
        print(f"✅ Prices: {'Loaded' if data['prices'] is not None else 'Missing'}")
        print(f"{'✅' if data['ipc'] is not None else '❌'} IPC Phases: {'Loaded' if data['ipc'] is not None else 'Missing (REQUIRED)'}")
        print(f"{'✅' if data['rainfall'] is not None else '⚠️ '} Rainfall: {'Loaded' if data['rainfall'] is not None else 'Optional'}")
        print(f"{'✅' if data['conflict'] is not None else '⚠️ '} Conflict: {'Loaded' if data['conflict'] is not None else 'Optional'}")
        print(f"{'✅' if data['population'] is not None else '⚠️ '} Population: {'Loaded' if data['population'] is not None else 'Optional'}")
        print()
        
        return data
    
    def create_unified_dataset(self) -> Optional[pd.DataFrame]:
        """
        Create unified dataset from all loaded sources.
        Returns DataFrame with columns: date, admin2, ipc_phase, prices, rainfall, conflict, etc.
        """
        data = self.load_all()
        
        if data['prices'] is None:
            print("❌ Cannot create unified dataset: price data is required")
            return None
        
        # Start with price data as base
        prices_df = data['prices'].copy()
        
        # Normalize date column
        if 'date' in prices_df.columns:
            prices_df['date'] = pd.to_datetime(prices_df['date'])
        else:
            print("❌ Price data must have 'date' column")
            return None
        
        # Normalize admin2 column
        if 'admin2' not in prices_df.columns:
            if 'region' in prices_df.columns:
                prices_df['admin2'] = prices_df['region']
            else:
                print("❌ Price data must have 'admin2' or 'region' column")
                return None
        
        # Aggregate prices by date and admin2
        price_cols = ['price', 'usdprice'] if 'usdprice' in prices_df.columns else ['price']
        prices_agg = prices_df.groupby(['date', 'admin2'])[price_cols].mean().reset_index()
        prices_agg.rename(columns={'price': 'avg_price', 'usdprice': 'avg_usd_price'}, inplace=True)
        
        # Merge IPC data
        if data['ipc'] is not None:
            ipc_df = data['ipc'].copy()
            if 'date' in ipc_df.columns:
                ipc_df['date'] = pd.to_datetime(ipc_df['date'])
                # Create month-year key for merging (since price dates are mid-month, IPC dates are start-of-month)
                prices_agg['year_month'] = prices_agg['date'].dt.to_period('M')
                ipc_df['year_month'] = ipc_df['date'].dt.to_period('M')
                
                # Merge on year_month and admin2 (case-insensitive)
                prices_agg['admin2_upper'] = prices_agg['admin2'].str.upper().str.strip()
                ipc_df['admin2_upper'] = ipc_df['admin2'].str.upper().str.strip()
                
                # Merge
                prices_agg = prices_agg.merge(
                    ipc_df[['year_month', 'admin2_upper', 'ipc_phase', 'population_affected']],
                    on=['year_month', 'admin2_upper'],
                    how='left'
                )
                
                # Clean up temporary columns
                prices_agg = prices_agg.drop(columns=['year_month', 'admin2_upper'])
        
        # Merge rainfall (optional)
        if data['rainfall'] is not None:
            rainfall_df = data['rainfall'].copy()
            if 'date' in rainfall_df.columns:
                rainfall_df['date'] = pd.to_datetime(rainfall_df['date'])
                prices_agg = prices_agg.merge(
                    rainfall_df[['date', 'admin2', 'rainfall_mm']],
                    on=['date', 'admin2'],
                    how='left'
                )
        
        # Merge conflict (optional)
        if data['conflict'] is not None:
            conflict_df = data['conflict'].copy()
            if 'date' in conflict_df.columns:
                conflict_df['date'] = pd.to_datetime(conflict_df['date'])
                # Aggregate conflict incidents by date and admin2
                conflict_agg = conflict_df.groupby(['date', 'admin2']).size().reset_index(name='conflict_incidents')
                prices_agg = prices_agg.merge(conflict_agg, on=['date', 'admin2'], how='left')
                prices_agg['conflict_incidents'] = prices_agg['conflict_incidents'].fillna(0)
        
        # Add population (optional, static)
        if data['population'] is not None:
            pop_df = data['population'].copy()
            prices_agg = prices_agg.merge(
                pop_df[['admin2', 'population']],
                on='admin2',
                how='left'
            )
        
        # Sort by date and admin2
        prices_agg = prices_agg.sort_values(['admin2', 'date']).reset_index(drop=True)
        
        print(f"✅ Created unified dataset with {len(prices_agg)} records")
        print(f"   Columns: {list(prices_agg.columns)}")
        print(f"   Date range: {prices_agg['date'].min()} to {prices_agg['date'].max()}")
        print(f"   Regions: {prices_agg['admin2'].nunique()}")
        
        return prices_agg


if __name__ == "__main__":
    loader = DataLoader()
    unified_data = loader.create_unified_dataset()
    if unified_data is not None:
        print("\nSample data:")
        print(unified_data.head())

