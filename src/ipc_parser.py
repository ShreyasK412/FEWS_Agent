"""
IPC Classification Parser
Robustly identifies at-risk regions from IPC CSV data.
"""
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass


@dataclass
class RegionRiskAssessment:
    """Risk assessment for a region."""
    region: str
    current_phase: int
    projected_phase_ml1: Optional[int]  # Near-term (3-4 months)
    projected_phase_ml2: Optional[int]  # Medium-term (4-8 months)
    is_at_risk: bool
    trend: str  # "improving", "stable", "deteriorating", "unknown"
    population_affected: Optional[int]
    latest_current_date: str
    latest_projection_date: Optional[str]
    risk_level: str  # "HIGH", "MEDIUM", "LOW"
    key_indicators: List[str]
    geographic_full_name: str


class IPCParser:
    """Parse and analyze IPC classification data to identify at-risk regions."""
    
    def __init__(self, ipc_file: str = "ipc classification data/ipcFic_data.csv"):
        self.ipc_file = Path(ipc_file)
        self.df: Optional[pd.DataFrame] = None
    
    def load(self) -> bool:
        """Load IPC CSV file."""
        if not self.ipc_file.exists():
            print(f"❌ IPC file not found: {self.ipc_file}")
            return False
        
        try:
            self.df = pd.read_csv(self.ipc_file)
            print(f"✅ Loaded {len(self.df)} IPC records")
            return True
        except Exception as e:
            print(f"❌ Error loading IPC data: {e}")
            return False
    
    def identify_at_risk_regions(
        self, 
        min_phase: int = 3,
        include_deteriorating: bool = True,
        include_projected: bool = True
    ) -> List[RegionRiskAssessment]:
        """
        Identify regions at risk based on IPC classifications.
        
        Strategy:
        1. Get latest Current Situation (CS) for each region
        2. Get latest projections (ML1, ML2) if available
        3. Calculate trend from historical CS records
        4. Flag as at-risk if:
           - Current phase >= min_phase (default 3)
           - Projected phase >= min_phase
           - Deteriorating trend (phase increasing)
        
        Args:
            min_phase: Minimum IPC phase to flag as at-risk (default: 3)
            include_deteriorating: Flag regions with deteriorating trends
            include_projected: Consider projected phases in risk assessment
        
        Returns:
            List of risk assessments, sorted by risk level
        """
        if self.df is None:
            if not self.load():
                return []
        
        # Group by region
        region_data: Dict[str, pd.DataFrame] = {}
        for region in self.df['geographic_unit_name'].unique():
            region_df = self.df[self.df['geographic_unit_name'] == region].copy()
            region_data[region] = region_df
        
        assessments = []
        
        for region, region_df in region_data.items():
            # Get latest Current Situation (CS)
            cs_records = region_df[region_df['scenario'] == 'CS'].copy()
            if len(cs_records) == 0:
                continue
            
            # Sort by date, get latest
            cs_records['projection_start'] = pd.to_datetime(cs_records['projection_start'])
            cs_records = cs_records.sort_values('projection_start', ascending=False)
            latest_cs = cs_records.iloc[0]
            
            current_phase = int(float(latest_cs['value']))
            latest_current_date = str(latest_cs['projection_start'].date())
            geographic_full_name = str(latest_cs.get('geographic_unit_full_name', region))
            
            # Get latest projections
            ml1_records = region_df[region_df['scenario'] == 'ML1'].copy()
            ml2_records = region_df[region_df['scenario'] == 'ML2'].copy()
            
            projected_phase_ml1 = None
            projected_phase_ml2 = None
            latest_projection_date = None
            
            if len(ml1_records) > 0:
                ml1_records['projection_start'] = pd.to_datetime(ml1_records['projection_start'])
                ml1_records = ml1_records.sort_values('projection_start', ascending=False)
                latest_ml1 = ml1_records.iloc[0]
                projected_phase_ml1 = int(float(latest_ml1['value']))
                latest_projection_date = str(latest_ml1['projection_start'].date())
            
            if len(ml2_records) > 0:
                ml2_records['projection_start'] = pd.to_datetime(ml2_records['projection_start'])
                ml2_records = ml2_records.sort_values('projection_start', ascending=False)
                latest_ml2 = ml2_records.iloc[0]
                projected_phase_ml2 = int(float(latest_ml2['value']))
                if latest_projection_date is None:
                    latest_projection_date = str(latest_ml2['projection_start'].date())
            
            # Determine trend from historical CS records
            trend = "unknown"
            if len(cs_records) >= 2:
                # Compare latest with previous
                prev_phase = int(float(cs_records.iloc[1]['value']))
                if current_phase > prev_phase:
                    trend = "deteriorating"
                elif current_phase < prev_phase:
                    trend = "improving"
                else:
                    trend = "stable"
            
            # Check if at risk
            is_at_risk = False
            indicators = []
            
            # Current phase check
            if current_phase >= min_phase:
                is_at_risk = True
                indicators.append(f"Current IPC Phase {current_phase}")
            
            # Projected phase check
            if include_projected:
                if projected_phase_ml1 is not None and projected_phase_ml1 >= min_phase:
                    is_at_risk = True
                    indicators.append(f"Projected Phase {projected_phase_ml1} (Near-term)")
                if projected_phase_ml2 is not None and projected_phase_ml2 >= min_phase:
                    is_at_risk = True
                    indicators.append(f"Projected Phase {projected_phase_ml2} (Medium-term)")
            
            # Deteriorating trend check
            if include_deteriorating and trend == "deteriorating" and current_phase >= 2:
                is_at_risk = True
                indicators.append("Deteriorating trend")
            
            # Determine risk level
            max_phase = current_phase
            if projected_phase_ml1 is not None:
                max_phase = max(max_phase, projected_phase_ml1)
            if projected_phase_ml2 is not None:
                max_phase = max(max_phase, projected_phase_ml2)
            
            if max_phase >= 4:
                risk_level = "HIGH"
            elif max_phase >= 3:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"
            
            assessment = RegionRiskAssessment(
                region=region,
                current_phase=current_phase,
                projected_phase_ml1=projected_phase_ml1,
                projected_phase_ml2=projected_phase_ml2,
                is_at_risk=is_at_risk,
                trend=trend,
                population_affected=None,  # Not in current CSV
                latest_current_date=latest_current_date,
                latest_projection_date=latest_projection_date,
                risk_level=risk_level,
                key_indicators=indicators if indicators else ["No significant risk indicators"],
                geographic_full_name=geographic_full_name
            )
            
            assessments.append(assessment)
        
        # Sort by risk level (HIGH first), then by phase
        risk_order = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
        assessments.sort(
            key=lambda x: (
                risk_order.get(x.risk_level, 0),
                x.current_phase,
                x.projected_phase_ml1 or 0,
                x.projected_phase_ml2 or 0
            ),
            reverse=True
        )
        
        return assessments
    
    def get_region_assessment(self, region: str) -> Optional[RegionRiskAssessment]:
        """Get risk assessment for a specific region."""
        assessments = self.identify_at_risk_regions()
        for assessment in assessments:
            if assessment.region.lower() == region.lower():
                return assessment
        return None


if __name__ == "__main__":
    parser = IPCParser()
    assessments = parser.identify_at_risk_regions()
    
    at_risk = [a for a in assessments if a.is_at_risk]
    
    print(f"\n{'='*80}")
    print(f"AT-RISK REGIONS IDENTIFIED: {len(at_risk)} out of {len(assessments)} total")
    print(f"{'='*80}\n")
    
    print(f"High Risk (IPC 4+): {len([a for a in at_risk if a.risk_level == 'HIGH'])}")
    print(f"Medium Risk (IPC 3): {len([a for a in at_risk if a.risk_level == 'MEDIUM'])}")
    print(f"Low Risk (IPC 1-2): {len([a for a in at_risk if a.risk_level == 'LOW'])}")
    print()
    
    # Show top 20 at-risk regions
    for i, assessment in enumerate(at_risk[:20], 1):
        print(f"{i}. 🔴 {assessment.region} ({assessment.geographic_full_name})")
        print(f"   Current Phase: {assessment.current_phase}")
        if assessment.projected_phase_ml1:
            print(f"   Projected (Near-term): Phase {assessment.projected_phase_ml1}")
        if assessment.projected_phase_ml2:
            print(f"   Projected (Medium-term): Phase {assessment.projected_phase_ml2}")
        print(f"   Risk Level: {assessment.risk_level}")
        print(f"   Trend: {assessment.trend}")
        print(f"   Indicators: {', '.join(assessment.key_indicators)}")
        print()
