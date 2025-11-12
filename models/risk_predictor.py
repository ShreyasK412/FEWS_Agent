"""
Risk prediction models: Rule-based, ML, and Ensemble approaches.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class RiskAssessment:
    """Risk assessment output structure."""
    region_id: str
    risk_level: str  # HIGH, MEDIUM, LOW
    ipc_phase_prediction: int  # 1-5
    confidence_score: float  # 0-1
    key_drivers: List[str]
    time_horizon: str
    population_at_risk: Optional[int] = None


class RuleBasedPredictor:
    """
    Rule-based early warning system using IPC protocol thresholds.
    """
    
    def __init__(self):
        self.thresholds = {
            "price_increase_warning": 0.25,  # 25% increase in 3 months
            "rainfall_deficit_alert": -0.30,  # 30% deficit
            "conflict_escalation_threshold": 0.50,  # 50% increase
            "price_volatility_high": 0.20,  # 20% volatility
        }
    
    def predict(self, features: pd.DataFrame, region: str) -> RiskAssessment:
        """
        Predict risk using rule-based thresholds.
        
        Args:
            features: DataFrame with engineered features
            region: Region identifier
        
        Returns:
            RiskAssessment object
        """
        region_data = features[features["region"] == region].iloc[-1] if len(features) > 0 else None
        
        if region_data is None:
            return RiskAssessment(
                region_id=region,
                risk_level="UNKNOWN",
                ipc_phase_prediction=2,
                confidence_score=0.0,
                key_drivers=[],
                time_horizon="3 months"
            )
        
        risk_factors = []
        drivers = []
        
        # Check price increases
        if "yoy_price_change" in region_data and pd.notna(region_data["yoy_price_change"]):
            if region_data["yoy_price_change"] > self.thresholds["price_increase_warning"]:
                risk_factors.append(1.0)
                drivers.append(f"Price increase: {region_data['yoy_price_change']:.1%}")
        
        # Check rainfall deficit
        if "rainfall_deficit" in region_data and pd.notna(region_data["rainfall_deficit"]):
            if region_data["rainfall_deficit"] < self.thresholds["rainfall_deficit_alert"]:
                risk_factors.append(1.0)
                drivers.append(f"Rainfall deficit: {region_data['rainfall_deficit']:.1%}")
        
        # Check conflict escalation
        if "conflict_escalation" in region_data and pd.notna(region_data["conflict_escalation"]):
            if region_data["conflict_escalation"] > self.thresholds["conflict_escalation_threshold"]:
                risk_factors.append(1.0)
                drivers.append(f"Conflict escalation: {region_data['conflict_escalation']:.1%}")
        
        # Check current IPC phase
        if "ipc_phase" in region_data and pd.notna(region_data["ipc_phase"]):
            current_phase = region_data["ipc_phase"]
            if current_phase >= 3:
                risk_factors.append(1.0)
                drivers.append(f"Current IPC Phase {current_phase}")
        
        # Calculate risk level
        risk_score = np.mean(risk_factors) if risk_factors else 0.0
        
        if risk_score >= 0.7:
            risk_level = "HIGH"
            ipc_prediction = 4
        elif risk_score >= 0.4:
            risk_level = "MEDIUM"
            ipc_prediction = 3
        else:
            risk_level = "LOW"
            ipc_prediction = 2
        
        confidence = min(risk_score + 0.2, 1.0)  # Base confidence
        
        return RiskAssessment(
            region_id=region,
            risk_level=risk_level,
            ipc_phase_prediction=ipc_prediction,
            confidence_score=confidence,
            key_drivers=drivers if drivers else ["Insufficient data"],
            time_horizon="3 months",
            population_at_risk=None  # Would need population data
        )


class MLRiskPredictor:
    """
    Machine learning risk predictor (if sufficient historical data available).
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.model_path = model_path
        if model_path:
            self.load_model()
    
    def load_model(self):
        """Load trained model."""
        try:
            import pickle
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def predict(self, features: pd.DataFrame, region: str) -> RiskAssessment:
        """Predict using ML model."""
        if self.model is None:
            # Fallback to rule-based
            rule_predictor = RuleBasedPredictor()
            return rule_predictor.predict(features, region)
        
        # Extract features for region
        region_data = features[features["region"] == region].iloc[-1] if len(features) > 0 else None
        
        if region_data is None:
            return RiskAssessment(
                region_id=region,
                risk_level="UNKNOWN",
                ipc_phase_prediction=2,
                confidence_score=0.0,
                key_drivers=[],
                time_horizon="3 months"
            )
        
        # Use model to predict (implementation depends on model type)
        # This is a placeholder
        return RiskAssessment(
            region_id=region,
            risk_level="MEDIUM",
            ipc_phase_prediction=3,
            confidence_score=0.6,
            key_drivers=["ML prediction"],
            time_horizon="3 months"
        )


class EnsemblePredictor:
    """
    Ensemble predictor combining rule-based and ML approaches.
    """
    
    def __init__(self, ml_model_path: Optional[str] = None):
        self.rule_predictor = RuleBasedPredictor()
        self.ml_predictor = MLRiskPredictor(ml_model_path)
    
    def predict(self, features: pd.DataFrame, region: str) -> RiskAssessment:
        """
        Combine predictions from rule-based and ML models.
        """
        rule_assessment = self.rule_predictor.predict(features, region)
        ml_assessment = self.ml_predictor.predict(features, region)
        
        # Combine predictions (weighted average)
        # For now, prioritize rule-based if ML not available
        if ml_assessment.confidence_score > 0.5:
            # Use ML if available
            combined_phase = int((rule_assessment.ipc_phase_prediction + ml_assessment.ipc_phase_prediction) / 2)
            combined_confidence = (rule_assessment.confidence_score + ml_assessment.confidence_score) / 2
            combined_drivers = list(set(rule_assessment.key_drivers + ml_assessment.key_drivers))
        else:
            # Use rule-based
            combined_phase = rule_assessment.ipc_phase_prediction
            combined_confidence = rule_assessment.confidence_score
            combined_drivers = rule_assessment.key_drivers
        
        # Determine risk level
        if combined_phase >= 4:
            risk_level = "HIGH"
        elif combined_phase >= 3:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        return RiskAssessment(
            region_id=region,
            risk_level=risk_level,
            ipc_phase_prediction=combined_phase,
            confidence_score=combined_confidence,
            key_drivers=combined_drivers,
            time_horizon="3 months",
            population_at_risk=rule_assessment.population_at_risk
        )


if __name__ == "__main__":
    # Test
    sample_features = pd.DataFrame({
        "region": ["Tigray", "Tigray"],
        "yoy_price_change": [0.30, 0.15],
        "rainfall_deficit": [-0.35, -0.10],
        "ipc_phase": [4, 3]
    })
    
    predictor = EnsemblePredictor()
    assessment = predictor.predict(sample_features, "Tigray")
    print(assessment)

