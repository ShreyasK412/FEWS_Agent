"""
Intervention recommendation system.
Maps risk drivers to evidence-based interventions.
"""
import yaml
from pathlib import Path
from typing import List, Dict, Optional
from models.risk_predictor import RiskAssessment


class InterventionRecommender:
    """Recommend interventions based on risk assessment."""
    
    def __init__(self, playbook_path: Optional[str] = None):
        if playbook_path is None:
            playbook_path = Path(__file__).parent.parent / "kb" / "interventions" / "playbook.yaml"
        self.playbook_path = Path(playbook_path)
        self.playbook = self._load_playbook()
    
    def _load_playbook(self) -> List[Dict]:
        """Load intervention playbook."""
        if not self.playbook_path.exists():
            return []
        
        with open(self.playbook_path, "r") as f:
            data = yaml.safe_load(f)
            return data.get("interventions", [])
    
    def recommend(
        self,
        risk_assessment: RiskAssessment,
        document_context: List[Dict] = None
    ) -> List[Dict]:
        """
        Recommend interventions based on risk assessment.
        
        Args:
            risk_assessment: RiskAssessment object
            document_context: Optional document context from RAG
        
        Returns:
            List of recommended interventions with priorities
        """
        recommendations = []
        
        # Match drivers to playbook rules
        for intervention in self.playbook:
            driver = intervention.get("driver", "")
            condition = intervention.get("condition", "")
            
            # Check if risk assessment matches intervention conditions
            matches = self._check_condition(risk_assessment, driver, condition)
            
            if matches:
                recommendations.append({
                    "name": intervention.get("intervention", ""),
                    "driver": driver,
                    "requirements": intervention.get("ops_requirements", []),
                    "evidence_link": intervention.get("evidence_link", ""),
                    "priority": self._calculate_priority(risk_assessment, intervention)
                })
        
        # Sort by priority
        recommendations.sort(key=lambda x: x["priority"], reverse=True)
        
        # Enhance with document context if available
        if document_context:
            recommendations = self._enhance_with_context(recommendations, document_context)
        
        return recommendations[:5]  # Top 5 recommendations
    
    def _check_condition(
        self,
        assessment: RiskAssessment,
        driver: str,
        condition: str
    ) -> bool:
        """Check if risk assessment matches intervention condition."""
        # Simple matching - can be enhanced
        driver_lower = driver.lower()
        
        # Check if any key driver matches
        for key_driver in assessment.key_drivers:
            if driver_lower in key_driver.lower() or key_driver.lower() in driver_lower:
                return True
        
        # Check IPC phase
        if "multi_driver" in driver_lower and assessment.ipc_phase_prediction >= 3:
            return True
        
        return False
    
    def _calculate_priority(
        self,
        assessment: RiskAssessment,
        intervention: Dict
    ) -> float:
        """Calculate intervention priority score."""
        priority = 0.0
        
        # Higher risk = higher priority
        if assessment.risk_level == "HIGH":
            priority += 1.0
        elif assessment.risk_level == "MEDIUM":
            priority += 0.5
        
        # Higher IPC phase = higher priority
        priority += assessment.ipc_phase_prediction * 0.2
        
        # Confidence boost
        priority += assessment.confidence_score * 0.3
        
        return priority
    
    def _enhance_with_context(
        self,
        recommendations: List[Dict],
        context: List[Dict]
    ) -> List[Dict]:
        """Enhance recommendations with document context."""
        # Add evidence from documents
        for rec in recommendations:
            rec["evidence"] = []
            for doc in context[:3]:  # Top 3 context docs
                if rec["name"].lower() in doc.get("text", "").lower():
                    rec["evidence"].append({
                        "source": doc.get("source", "Unknown"),
                        "excerpt": doc.get("text", "")[:200]
                    })
        
        return recommendations


if __name__ == "__main__":
    from models.risk_predictor import RiskAssessment
    
    recommender = InterventionRecommender()
    
    test_assessment = RiskAssessment(
        region_id="Tigray",
        risk_level="HIGH",
        ipc_phase_prediction=4,
        confidence_score=0.85,
        key_drivers=["conflict_escalation", "price_increase"],
        time_horizon="3 months"
    )
    
    recommendations = recommender.recommend(test_assessment)
    print(f"Recommended {len(recommendations)} interventions:")
    for rec in recommendations:
        print(f"  - {rec['name']} (Priority: {rec['priority']:.2f})")

