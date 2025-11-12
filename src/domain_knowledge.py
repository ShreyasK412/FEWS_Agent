"""
Domain Knowledge Layer for FEWS System
Provides structured lookups for livelihoods, rainfall seasons, shocks, and interventions.
"""
import json
import pandas as pd
import re
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
    clarification: Optional[str] = None


class DomainKnowledge:
    """Domain knowledge layer providing structured lookups."""
    
    def __init__(self, data_dir: str = "data/domain"):
        self.data_dir = Path(data_dir)
        self.livelihood_df: Optional[pd.DataFrame] = None
        self.rainfall_df: Optional[pd.DataFrame] = None
        self.shock_ontology: Dict = {}
        self.driver_interventions: Dict = {}
        self._fallback_livelihoods: Dict[str, LivelihoodInfo] = {}
        self._fallback_rainfall: Dict[str, RainfallSeasonInfo] = {}
        self._shock_to_driver: Dict[str, str] = {
            "drought": "Drought/Rainfall deficit",
            "conflict": "Conflict and insecurity",
            "displacement": "Displacement",
            "price_increase": "Price increases",
            "crop_pests": "Crop failure",
            "livestock_mortality": "Livestock losses",
            "market_disruption": "Market disruption",
            "humanitarian_access_constraints": "Humanitarian access constraints",
            "flooding": "Flooding",
            "macroeconomic_shocks": "Macroeconomic shocks"
        }
        
        self._load_all()
    
    def _load_all(self):
        """Load all domain knowledge files."""
        # Load livelihood systems
        livelihood_file = self.data_dir / "livelihood_systems.csv"
        if livelihood_file.exists():
            self.livelihood_df = pd.read_csv(livelihood_file)
            if not self.livelihood_df.empty:
                # Normalize names for matching
                self.livelihood_df['region_upper'] = self.livelihood_df['region'].str.upper()
                self.livelihood_df['zone_upper'] = self.livelihood_df['zone'].str.upper()
                self.livelihood_df['region_clean'] = self.livelihood_df['region_upper'].apply(self._normalize_location_name)
                self.livelihood_df['zone_clean'] = self.livelihood_df['zone_upper'].apply(self._normalize_location_name)
        else:
            print(f"⚠️  Warning: {livelihood_file} not found. Livelihood lookups will use fallback logic.")
        
        # Load rainfall seasons
        rainfall_file = self.data_dir / "rainfall_seasons.csv"
        if rainfall_file.exists():
            self.rainfall_df = pd.read_csv(rainfall_file)
            if not self.rainfall_df.empty:
                self.rainfall_df['region_upper'] = self.rainfall_df['region'].str.upper()
                self.rainfall_df['zone_upper'] = self.rainfall_df['zone'].str.upper()
                self.rainfall_df['region_clean'] = self.rainfall_df['region_upper'].apply(self._normalize_location_name)
                self.rainfall_df['zone_clean'] = self.rainfall_df['zone_upper'].apply(self._normalize_location_name)
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
        
        self._initialize_fallbacks()

    @staticmethod
    def _normalize_location_name(name: Optional[str]) -> str:
        """Normalize location names for matching."""
        if not name:
            return ""
        cleaned = name.upper().strip()
        replacements = [
            (" ADMINISTRATIVE", ""),
            (" ZONE", ""),
            (" REGION", ""),
            (" SPECIAL", ""),
            (" WOREDA", ""),
            (" DISTRICT", ""),
            (" SUBCITY", ""),
            (" SUB-CITY", ""),
        ]
        for old, new in replacements:
            cleaned = cleaned.replace(old, new)
        return " ".join(cleaned.split())

    def _initialize_fallbacks(self) -> None:
        """Initialize fallback maps for livelihoods and rainfall seasons."""
        self._fallback_livelihoods = {
            "TIGRAY": LivelihoodInfo(
                region="Tigray",
                zone="Highland areas",
                livelihood_system="rainfed cropping",
                elevation_category="highland",
                notes="Fallback: Highland cereal-based cropping system typical for Tigray."
            ),
            "AMHARA": LivelihoodInfo(
                region="Amhara",
                zone="Highland areas",
                livelihood_system="rainfed cropping",
                elevation_category="highland",
                notes="Fallback: Highland cereal-based cropping system typical for Amhara."
            ),
            "OROMIA": LivelihoodInfo(
                region="Oromia",
                zone="Mixed livelihood areas",
                livelihood_system="mixed farming",
                elevation_category="mid-altitude",
                notes="Fallback: Mixed crop-livestock livelihood system."
            ),
            "SOMALI": LivelihoodInfo(
                region="Somali",
                zone="Pastoral areas",
                livelihood_system="pastoral",
                elevation_category="lowland",
                notes="Fallback: Pastoral system typical for Somali region."
            ),
            "AFAR": LivelihoodInfo(
                region="Afar",
                zone="Pastoral areas",
                livelihood_system="pastoral",
                elevation_category="lowland",
                notes="Fallback: Arid pastoral system reliant on livestock."
            ),
            "SNNPR": LivelihoodInfo(
                region="SNNPR",
                zone="Mixed areas",
                livelihood_system="agropastoral",
                elevation_category="mid-altitude",
                notes="Fallback: Mixed enset-root crop and agropastoral system."
            ),
        }
        
        self._fallback_rainfall = {
            "TIGRAY": RainfallSeasonInfo(
                region="Tigray",
                zone="Highland areas",
                dominant_season="Kiremt (June-September)",
                secondary_season=None,
                season_months="Jun-Sep",
                notes="Fallback: Highland main cropping season."
            ),
            "AMHARA": RainfallSeasonInfo(
                region="Amhara",
                zone="Highland areas",
                dominant_season="Kiremt (June-September)",
                secondary_season=None,
                season_months="Jun-Sep",
                notes="Fallback: Highland main cropping season."
            ),
            "OROMIA": RainfallSeasonInfo(
                region="Oromia",
                zone="Mixed areas",
                dominant_season="Belg/Kiremt mixed rainfall pattern",
                secondary_season="Belg (Feb-May)",
                season_months="Feb-May, Jun-Sep",
                notes="Fallback: Mixed rainfall seasons across Oromia."
            ),
            "SOMALI": RainfallSeasonInfo(
                region="Somali",
                zone="Pastoral areas",
                dominant_season="Gu/Genna (Mar-May)",
                secondary_season="Deyr/Hageya (Oct-Dec)",
                season_months="Mar-May, Oct-Dec",
                notes="Fallback: Pastoral rainfall seasons."
            ),
            "AFAR": RainfallSeasonInfo(
                region="Afar",
                zone="Pastoral areas",
                dominant_season="Belg pulse rains",
                secondary_season=None,
                season_months="Feb-Apr",
                notes="Fallback: Erratic belg pulses and dry season conditions."
            ),
            "SNNPR": RainfallSeasonInfo(
                region="SNNPR",
                zone="Mixed areas",
                dominant_season="Belg/Kiremt mixed rainfall pattern",
                secondary_season=None,
                season_months="Mar-May, Jun-Sep",
                notes="Fallback: Mixed rainfall seasons for enset-root crop systems."
            ),
        }
    
    def get_livelihood_system(
        self,
        region: Optional[str],
        zone: Optional[str] = None,
        admin: Optional[str] = None
    ) -> Optional[LivelihoodInfo]:
        """
        Get livelihood system for a region/zone.
        
        Args:
            region: Region name (e.g., "Tigray", "SNNPR")
            zone: Optional zone name (e.g., "Central Zone", "Burji Special")
            admin: Optional woreda/district name
        
        Returns:
            LivelihoodInfo if found, None otherwise
        """
        if not region:
            region = ""
        region_upper = region.upper()
        region_clean = self._normalize_location_name(region)
        zone_clean = self._normalize_location_name(zone) if zone else ""
        admin_clean = self._normalize_location_name(admin) if admin else ""
        
        if self.livelihood_df is not None and not self.livelihood_df.empty:
            # 1. Zone-level exact/partial match within region
            if zone_clean:
                zone_matches = self.livelihood_df[
                    (self.livelihood_df['region_clean'] == region_clean) &
                    (
                        (self.livelihood_df['zone_clean'] == zone_clean) |
                        (self.livelihood_df['zone_clean'].str.contains(zone_clean, na=False, regex=False))
                    )
                ]
                if not zone_matches.empty:
                    row = zone_matches.iloc[0]
                    return LivelihoodInfo(
                        region=row['region'],
                        zone=row['zone'],
                        livelihood_system=row['livelihood_system'],
                        elevation_category=row.get('elevation_category', ''),
                        notes=row.get('notes', '')
                    )
            
            # 2. Region-only match
            if region_clean:
                region_matches = self.livelihood_df[
                    (self.livelihood_df['region_clean'] == region_clean) |
                    (self.livelihood_df['region_clean'].str.contains(region_clean, na=False, regex=False))
                ]
                if not region_matches.empty:
                    row = region_matches.iloc[0]
                    return LivelihoodInfo(
                        region=row['region'],
                        zone=row['zone'],
                        livelihood_system=row['livelihood_system'],
                        elevation_category=row.get('elevation_category', ''),
                        notes=row.get('notes', '')
                    )
            
            # 3. Admin name fallback (match admin to zone names)
            if admin_clean:
                admin_matches = self.livelihood_df[
                    self.livelihood_df['zone_clean'].str.contains(admin_clean, na=False, regex=False)
                ]
                if not admin_matches.empty:
                    row = admin_matches.iloc[0]
                    return LivelihoodInfo(
                        region=row['region'],
                        zone=row['zone'],
                        livelihood_system=row['livelihood_system'],
                        elevation_category=row.get('elevation_category', ''),
                        notes=row.get('notes', '')
                    )
        
        # 4. Regional fallback map
        if region_upper in self._fallback_livelihoods:
            return self._fallback_livelihoods[region_upper]
        
        # 5. Broad macro-region heuristics
        if "TIGRAY" in region_upper:
            return self._fallback_livelihoods["TIGRAY"]
        if "AMHARA" in region_upper:
            return self._fallback_livelihoods["AMHARA"]
        if "OROMIA" in region_upper:
            return self._fallback_livelihoods["OROMIA"]
        if "SOMALI" in region_upper:
            return self._fallback_livelihoods["SOMALI"]
        if "AFAR" in region_upper:
            return self._fallback_livelihoods["AFAR"]
        if "SNNP" in region_upper or "SNNPR" in region_upper:
            return self._fallback_livelihoods["SNNPR"]
        
        return None
    
    def get_rainfall_season(
        self,
        region: Optional[str],
        zone: Optional[str] = None,
        admin: Optional[str] = None
    ) -> Optional[RainfallSeasonInfo]:
        """
        Get rainfall season information for a region/zone.
        
        Args:
            region: Region name
            zone: Optional zone name
            admin: Optional woreda/district name
        
        Returns:
            RainfallSeasonInfo if found, None otherwise
        """
        if not region:
            region = ""
        region_upper = region.upper()
        region_clean = self._normalize_location_name(region)
        zone_clean = self._normalize_location_name(zone) if zone else ""
        admin_clean = self._normalize_location_name(admin) if admin else ""
        
        if self.rainfall_df is not None and not self.rainfall_df.empty:
            # 1. Zone-level exact/partial match within region
            if zone_clean:
                zone_matches = self.rainfall_df[
                    (self.rainfall_df['region_clean'] == region_clean) &
                    (
                        (self.rainfall_df['zone_clean'] == zone_clean) |
                        (self.rainfall_df['zone_clean'].str.contains(zone_clean, na=False, regex=False))
                    )
                ]
                if not zone_matches.empty:
                    row = zone_matches.iloc[0]
                    return RainfallSeasonInfo(
                        region=row['region'],
                        zone=row['zone'],
                        dominant_season=row['dominant_season'],
                        secondary_season=row.get('secondary_season'),
                        season_months=row['season_months'],
                        notes=row.get('notes', ''),
                        clarification=row.get('clarification')
                    )
            
            # 2. Region-only match
            if region_clean:
                region_matches = self.rainfall_df[
                    (self.rainfall_df['region_clean'] == region_clean) |
                    (self.rainfall_df['region_clean'].str.contains(region_clean, na=False, regex=False))
                ]
                if not region_matches.empty:
                    row = region_matches.iloc[0]
                    return RainfallSeasonInfo(
                        region=row['region'],
                        zone=row['zone'],
                        dominant_season=row['dominant_season'],
                        secondary_season=row.get('secondary_season'),
                        season_months=row['season_months'],
                        notes=row.get('notes', ''),
                        clarification=row.get('clarification')
                    )
            
            # 3. Admin fallback using zone matches
            if admin_clean:
                admin_matches = self.rainfall_df[
                    self.rainfall_df['zone_clean'].str.contains(admin_clean, na=False, regex=False)
                ]
                if not admin_matches.empty:
                    row = admin_matches.iloc[0]
                    return RainfallSeasonInfo(
                        region=row['region'],
                        zone=row['zone'],
                        dominant_season=row['dominant_season'],
                        secondary_season=row.get('secondary_season'),
                        season_months=row['season_months'],
                        notes=row.get('notes', ''),
                        clarification=row.get('clarification')
                    )
        
        # 4. Regional fallback map
        if region_upper in self._fallback_rainfall:
            return self._fallback_rainfall[region_upper]
        
        # 5. Macro-region heuristics
        if "TIGRAY" in region_upper:
            return self._fallback_rainfall["TIGRAY"]
        if "AMHARA" in region_upper:
            return self._fallback_rainfall["AMHARA"]
        if "OROMIA" in region_upper:
            return self._fallback_rainfall["OROMIA"]
        if "SOMALI" in region_upper:
            return self._fallback_rainfall["SOMALI"]
        if "AFAR" in region_upper:
            return self._fallback_rainfall["AFAR"]
        if "SNNP" in region_upper or "SNNPR" in region_upper:
            return self._fallback_rainfall["SNNPR"]
        
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
        
        detected.sort(key=lambda x: x[1], reverse=True)
        return detected

    def map_shocks_to_drivers(self, shock_types: List[str]) -> List[str]:
        """Map shock ontology keys to human-readable drivers."""
        drivers: List[str] = []
        for shock in shock_types:
            driver = self._shock_to_driver.get(shock)
            if driver and driver not in drivers:
                drivers.append(driver)
        return drivers
    
    def detect_shocks_by_zone(
        self, 
        text: str, 
        livelihood_zone: str, 
        region: str,
        ipc_phase: Optional[int] = None
    ) -> Tuple[List[str], List[Dict[str, any]]]:
        """
        Detect shocks using zone-appropriate keywords with confidence scoring.
        
        Args:
            text: Retrieved context text to analyze
            livelihood_zone: Livelihood system (e.g., 'rainfed cropping', 'pastoral', 'agropastoral')
            region: Region name for context
            ipc_phase: IPC phase for anomaly detection
            
        Returns:
            Tuple of (shock_types, detailed_results)
            - shock_types: List of detected shock type keys
            - detailed_results: List of dicts with shock type, keywords found, and confidence
        """
        detected_shocks = []
        shock_types = []
        text_lower = text.lower()
        
        # Determine zone category for keyword selection
        if 'rainfed' in livelihood_zone.lower() or 'cropping' in livelihood_zone.lower() or 'highland' in livelihood_zone.lower():
            zone_category = 'highland_cropping'
        elif 'pastoral' in livelihood_zone.lower() and 'agro' not in livelihood_zone.lower():
            zone_category = 'pastoral'
        elif 'agropastoral' in livelihood_zone.lower() or 'mixed' in livelihood_zone.lower():
            zone_category = 'agropastoral'
        else:
            zone_category = 'all_zones'
        
        # Analyze each shock type
        for shock_type, shock_data in self.shock_ontology.items():
            # Get zone-specific keywords
            zone_keywords = shock_data.get(f'keywords_{zone_category}', [])
            all_zone_keywords = shock_data.get('keywords_all_zones', [])
            
            # Combine keywords
            keywords = zone_keywords + all_zone_keywords
            
            # Check for keyword matches
            matches = [kw for kw in keywords if kw.lower() in text_lower]
            
            if matches:
                # Calculate confidence based on number of matches
                if len(matches) >= 3:
                    confidence = 'high'
                elif len(matches) >= 2:
                    confidence = 'medium'
                else:
                    confidence = 'low'
                
                shock_types.append(shock_type)
                detected_shocks.append({
                    'type': shock_type,
                    'keywords_found': matches[:5],  # Top 5 matches
                    'confidence': confidence,
                    'category': shock_data.get('category', 'unknown'),
                    'zone_category': zone_category
                })
        
        # If no shocks detected but IPC >= 3, flag as anomaly
        if not detected_shocks and ipc_phase and ipc_phase >= 3:
            detected_shocks.append({
                'type': 'unknown',
                'keywords_found': [],
                'confidence': 'low',
                'category': 'unknown',
                'note': f'IPC Phase {ipc_phase} indicates crisis but no specific shocks detected in retrieved text',
                'zone_category': zone_category
            })
            shock_types.append('unknown')
        
        return shock_types, detected_shocks
    
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

    def extract_detailed_shocks(
        self,
        text: str,
        livelihood_zone: str,
        admin_region: str
    ) -> List[Dict]:
        """
        Extract specific shocks with evidence using regex patterns.
        
        Args:
            text: Retrieved context from FEWS NET reports
            livelihood_zone: e.g., "highland_cropping", "pastoral", "rainfed_cropping"
            admin_region: e.g., "Tigray", "Somali"
        
        Returns:
            List of dicts with shock type, description, evidence, confidence
        """
        shocks = []
        text_lower = text.lower()
        region_lower = admin_region.lower() if admin_region else ""
        
        # Only analyze if text mentions this region or Ethiopia
        if region_lower and region_lower not in text_lower and "ethiopia" not in text_lower:
            return [{
                'type': 'data_gap',
                'description': f'Retrieved context does not specifically mention {admin_region} Region',
                'evidence': 'No regional match in retrieved text',
                'confidence': 'low'
            }]
        
        # Split text into sentences for evidence extraction
        sentences = re.split(r'[.!?]+', text)
        
        # Normalize livelihood zone
        livelihood_normalized = livelihood_zone.lower()
        is_highland_cropping = any(term in livelihood_normalized for term in ['highland', 'rainfed', 'cropping'])
        is_pastoral = 'pastoral' in livelihood_normalized
        
        # Define zone-specific patterns
        if is_highland_cropping:
            weather_patterns = [
                (r'delayed.*kiremt.*rain', 'Delayed kiremt rains'),
                (r'late onset.*kiremt', 'Late onset of kiremt season'),
                (r'dry spell.*tigray|dry spell.*amhara|dry spell.*northern', 'Dry spells during growing season'),
                (r'moisture deficit', 'Moisture deficits'),
                (r'below-average.*meher.*production|below.*average.*meher', 'Below-average meher production'),
                (r'sorghum.*delayed|delayed.*sorghum', 'Delayed sorghum development'),
                (r'late.*planting|planting.*delayed', 'Late planting due to delayed rains'),
            ]
            
            conflict_patterns = [
                (r'2020-2022.*conflict|conflict.*2020-2022|conflict.*2020.*2022', 'Conflict legacy (2020-2022)'),
                (r'labor migration.*constrain|labor migration.*decline|constrain.*labor migration', 'Constrained labor migration'),
                (r'road.*block|blocked.*road|road.*closure', 'Road blockages'),
                (r'fuel shortage.*tigray|fuel shortage.*amhara|fuel.*shortage.*northern', 'Fuel shortages'),
                (r'pretoria agreement|post-conflict', 'Post-conflict recovery challenges'),
            ]
            
        elif is_pastoral:
            weather_patterns = [
                (r'deyr.*below.*average|deyr.*fail|failed.*deyr', 'Below-average/failed deyr season'),
                (r'gu.*below.*average|gu.*fail|failed.*gu|genna.*fail', 'Below-average/failed gu/genna season'),
                (r'poor pasture|pasture.*degrad|pasture.*condition', 'Poor pasture conditions'),
                (r'drought.*pastoral|drought.*somali|drought.*borena', 'Drought in pastoral areas'),
            ]
            
            conflict_patterns = [
                (r'intercommunal.*conflict', 'Intercommunal conflict'),
                (r'displacement.*pastoral|pastoral.*displacement', 'Conflict-driven displacement'),
            ]
        else:
            # Default patterns for unknown zones
            weather_patterns = [
                (r'delayed.*rain|late.*rain', 'Delayed/late rainfall'),
                (r'dry spell|drought', 'Dry conditions'),
            ]
            conflict_patterns = [
                (r'conflict|insecurity', 'Conflict and insecurity'),
            ]
        
        # Economic patterns (common to all zones)
        economic_patterns = [
            (r'high.*food price|food price.*increas|price.*increas.*food', 'High food prices'),
            (r'inflation|currency.*depreciat', 'Inflation/currency depreciation'),
            (r'purchasing power.*below|purchasing power.*declin|below.*purchasing power', 'Below-average purchasing power'),
            (r'wage.*not.*kept.*pace|income.*below|wage.*below', 'Wages not keeping pace with inflation'),
        ]
        
        # Extract weather shocks
        for pattern, description in weather_patterns:
            matches = [s for s in sentences if re.search(pattern, s.lower())]
            if matches:
                # Get the most relevant sentence (prefer one mentioning the region)
                evidence_sentences = [s for s in matches if region_lower in s.lower()] if region_lower else matches
                evidence = (evidence_sentences[0] if evidence_sentences else matches[0]).strip()[:250]
                shocks.append({
                    'type': 'weather',
                    'description': description,
                    'evidence': evidence,
                    'confidence': 'high' if region_lower and region_lower in evidence.lower() else 'medium'
                })
        
        # Extract conflict shocks
        for pattern, description in conflict_patterns:
            matches = [s for s in sentences if re.search(pattern, s.lower())]
            if matches:
                evidence_sentences = [s for s in matches if region_lower in s.lower()] if region_lower else matches
                evidence = (evidence_sentences[0] if evidence_sentences else matches[0]).strip()[:250]
                shocks.append({
                    'type': 'conflict',
                    'description': description,
                    'evidence': evidence,
                    'confidence': 'high' if region_lower and region_lower in evidence.lower() else 'medium'
                })
        
        # Extract economic shocks
        for pattern, description in economic_patterns:
            matches = [s for s in sentences if re.search(pattern, s.lower())]
            if matches:
                evidence_sentences = [s for s in matches if region_lower in s.lower()] if region_lower else matches
                evidence = (evidence_sentences[0] if evidence_sentences else matches[0]).strip()[:250]
                shocks.append({
                    'type': 'economic',
                    'description': description,
                    'evidence': evidence,
                    'confidence': 'medium'
                })
        
        # Deduplicate shocks by description
        seen_descriptions = set()
        unique_shocks = []
        for shock in shocks:
            desc_key = shock['description'].lower()
            if desc_key not in seen_descriptions:
                seen_descriptions.add(desc_key)
                unique_shocks.append(shock)
        
        # If no shocks found, return data gap indicator
        if not unique_shocks:
            unique_shocks.append({
                'type': 'insufficient_data',
                'description': 'No specific shocks detected in retrieved context',
                'evidence': 'Pattern matching found no relevant shocks',
                'confidence': 'low',
                'note': f'Consider adding more {admin_region}-specific content to situation reports' if admin_region else 'Consider adding more region-specific content to situation reports'
            })
        
        return unique_shocks

