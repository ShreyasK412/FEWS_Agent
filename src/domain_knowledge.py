"""
Domain Knowledge Layer for FEWS System
Provides structured lookups for livelihoods, rainfall seasons, shocks, and interventions.
"""
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class LivelihoodInfo:
    """Livelihood system information for a region."""
    region: str
    zone: str
    livelihood_system: str
    elevation_category: str
    notes: str


@dataclass
class RainfallSeasonInfo:
    """Rainfall season information for a region."""
    region: str
    zone: str
    dominant_season: str
    secondary_season: Optional[str]
    season_months: str
    notes: str


class DomainKnowledge:
    """Domain knowledge layer providing structured lookups."""
    
    def __init__(self, data_dir: str = "data/domain"):
        self.data_dir = Path(data_dir)
        self.livelihood_df: Optional[pd.DataFrame] = None
        self.rainfall_df: Optional[pd.DataFrame] = None
        self.shock_ontology: Dict = {}
        self.driver_interventions: Dict = {}
        
        self._load_all()
    
    def _load_all(self):
        """Load all domain knowledge files."""
        # Load livelihood systems
        livelihood_file = self.data_dir / "livelihood_systems.csv"
        if livelihood_file.exists():
            self.livelihood_df = pd.read_csv(livelihood_file)
            # Normalize region names for matching
            self.livelihood_df['region_upper'] = self.livelihood_df['region'].str.upper()
            self.livelihood_df['zone_upper'] = self.livelihood_df['zone'].str.upper()
        else:
            print(f"⚠️  Warning: {livelihood_file} not found. Livelihood lookups will use fallback logic.")
        
        # Load rainfall seasons
        rainfall_file = self.data_dir / "rainfall_seasons.csv"
        if rainfall_file.exists():
            self.rainfall_df = pd.read_csv(rainfall_file)
            self.rainfall_df['region_upper'] = self.rainfall_df['region'].str.upper()
            self.rainfall_df['zone_upper'] = self.rainfall_df['zone'].str.upper()
        else:
            print(f"⚠️  Warning: {rainfall_file} not found. Rainfall season lookups will use fallback logic.")
        
        # Load shock ontology
        shock_file = self.data_dir / "shock_ontology.json"
        if shock_file.exists():
            with open(shock_file, 'r') as f:
                self.shock_ontology = json.load(f)
        else:
            print(f"⚠️  Warning: {shock_file} not found. Shock detection will use fallback logic.")
        
        # Load driver-intervention mapping
        intervention_file = self.data_dir / "driver_interventions.json"
        if intervention_file.exists():
            with open(intervention_file, 'r') as f:
                self.driver_interventions = json.load(f)
        else:
            print(f"⚠️  Warning: {intervention_file} not found. Intervention mapping will use fallback logic.")
    
    def get_livelihood_system(self, region: str, zone: Optional[str] = None) -> Optional[LivelihoodInfo]:
        """
        Get livelihood system for a region/zone.
        
        Args:
            region: Region name (e.g., "Tigray", "SNNPR")
            zone: Optional zone name (e.g., "Central Zone", "Burji Special")
        
        Returns:
            LivelihoodInfo if found, None otherwise
        """
        if self.livelihood_df is None:
            return None
        
        region_upper = region.upper()
        
        # Try exact match first
        if zone:
            zone_upper = zone.upper()
            matches = self.livelihood_df[
                (self.livelihood_df['region_upper'] == region_upper) &
                (self.livelihood_df['zone_upper'] == zone_upper)
            ]
            if not matches.empty:
                row = matches.iloc[0]
                return LivelihoodInfo(
                    region=row['region'],
                    zone=row['zone'],
                    livelihood_system=row['livelihood_system'],
                    elevation_category=row['elevation_category'],
                    notes=row['notes']
                )
        
        # Try region-only match
        matches = self.livelihood_df[self.livelihood_df['region_upper'] == region_upper]
        if not matches.empty:
            # If multiple zones, prefer the most common one
            row = matches.iloc[0]
            return LivelihoodInfo(
                region=row['region'],
                zone=row['zone'],
                livelihood_system=row['livelihood_system'],
                elevation_category=row['elevation_category'],
                notes=row['notes']
            )
        
        return None
    
    def get_rainfall_season(self, region: str, zone: Optional[str] = None) -> Optional[RainfallSeasonInfo]:
        """
        Get rainfall season information for a region/zone.
        
        Args:
            region: Region name
            zone: Optional zone name
        
        Returns:
            RainfallSeasonInfo if found, None otherwise
        """
        if self.rainfall_df is None:
            return None
        
        region_upper = region.upper()
        
        # Try exact match first
        if zone:
            zone_upper = zone.upper()
            matches = self.rainfall_df[
                (self.rainfall_df['region_upper'] == region_upper) &
                (self.rainfall_df['zone_upper'] == zone_upper)
            ]
            if not matches.empty:
                row = matches.iloc[0]
                return RainfallSeasonInfo(
                    region=row['region'],
                    zone=row['zone'],
                    dominant_season=row['dominant_season'],
                    secondary_season=row.get('secondary_season'),
                    season_months=row['season_months'],
                    notes=row['notes']
                )
        
        # Try region-only match
        matches = self.rainfall_df[self.rainfall_df['region_upper'] == region_upper]
        if not matches.empty:
            row = matches.iloc[0]
            return RainfallSeasonInfo(
                region=row['region'],
                zone=row['zone'],
                dominant_season=row['dominant_season'],
                secondary_season=row.get('secondary_season'),
                season_months=row['season_months'],
                notes=row['notes']
            )
        
        return None
    
    def detect_shocks(self, text: str) -> List[Tuple[str, float]]:
        """
        Detect shocks in text using keyword matching.
        
        Args:
            text: Text to analyze
        
        Returns:
            List of (shock_type, confidence) tuples, sorted by confidence
        """
        if not self.shock_ontology:
            return []
        
        text_lower = text.lower()
        detected = []
        
        for shock_type, shock_data in self.shock_ontology.items():
            keywords = shock_data.get('keywords', [])
            matches = sum(1 for kw in keywords if kw.lower() in text_lower)
            
            if matches > 0:
                # Simple confidence score based on keyword matches
                confidence = min(matches / len(keywords) * 2, 1.0)  # Normalize to 0-1
                detected.append((shock_type, confidence))
        
        # Sort by confidence (descending)
        detected.sort(key=lambda x: x[1], reverse=True)
        return detected
    
    def get_interventions_for_driver(self, driver: str) -> Optional[Dict]:
        """
        Get intervention recommendations for a specific driver.
        
        Args:
            driver: Driver name (e.g., "drought", "conflict")
        
        Returns:
            Dictionary with intervention categories, or None if not found
        """
        driver_lower = driver.lower()
        
        # Try exact match first
        if driver_lower in self.driver_interventions:
            return self.driver_interventions[driver_lower]
        
        # Try partial match (e.g., "drought" matches "drought_related")
        for key, value in self.driver_interventions.items():
            if driver_lower in key or key in driver_lower:
                return value
        
        return None
    
    def get_all_shock_keywords(self) -> Dict[str, List[str]]:
        """Get all shock keywords for use in prompts."""
        if not self.shock_ontology:
            return {}
        
        return {
            shock_type: shock_data.get('keywords', [])
            for shock_type, shock_data in self.shock_ontology.items()
        }

