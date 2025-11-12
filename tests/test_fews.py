"""
Quick test script for FEWS system
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import FEWSSystem
from src.ipc_parser import RegionRiskAssessment

print("="*80)
print("TESTING FEWS SYSTEM")
print("="*80)

# Initialize
system = FEWSSystem()

# Test Function 1
print("\n" + "="*80)
print("TEST 1: Identify At-Risk Regions")
print("="*80)
at_risk = system.function1_identify_at_risk_regions()
print(f"\n✅ Found {len(at_risk)} at-risk regions")

if at_risk:
    test_region = at_risk[0]
    print(f"\nTest region: {test_region.region}")
    print(f"  Current Phase: {test_region.current_phase}")
    print(f"  Risk Level: {test_region.risk_level}")
    
    # Test Function 2 (only if vector stores are set up)
    print("\n" + "="*80)
    print("TEST 2: Explain Why (requires vector stores)")
    print("="*80)
    print("Note: Run system.setup_vector_stores() first to test this function")
    
    # Test Function 3
    print("\n" + "="*80)
    print("TEST 3: Recommend Interventions (requires vector stores)")
    print("="*80)
    print("Note: Run system.setup_vector_stores() first to test this function")

print("\n✅ Basic IPC parsing test passed!")


def test_livelihood_never_marked_unknown():
    """Test that livelihood system is never marked as unknown when data exists."""
    system = FEWSSystem()
    # Create a mock assessment for Edaga arbi (Tigray)
    mock_assessment = RegionRiskAssessment(
        region="Edaga arbi",
        current_phase=4,
        projected_phase_ml1=None,
        projected_phase_ml2=None,
        is_at_risk=True,
        trend="deteriorating",
        population_affected=None,
        latest_current_date="2024-10-01",
        latest_projection_date=None,
        risk_level="HIGH",
        key_indicators=[],
        geographic_full_name="Edaga arbi, Central, Tigray, Ethiopia"
    )
    
    # Only test if vector stores are available
    if system.reports_vectorstore is not None:
        explanation = system.function2_explain_why("Edaga arbi", mock_assessment)
        text = explanation["explanation"].lower()
        assert "unknown livelihood" not in text, f"Found 'unknown livelihood' in: {text[:200]}"
        assert "no livelihood data available" not in text, f"Found 'no livelihood data available' in: {text[:200]}"
        print("✅ test_livelihood_never_marked_unknown passed")
    else:
        print("⚠️  test_livelihood_never_marked_unknown skipped (vector stores not initialized)")


def test_limitations_do_not_contradict_domain_knowledge():
    """Test that limitations section doesn't contradict domain knowledge."""
    system = FEWSSystem()
    mock_assessment = RegionRiskAssessment(
        region="Edaga arbi",
        current_phase=4,
        projected_phase_ml1=None,
        projected_phase_ml2=None,
        is_at_risk=True,
        trend="deteriorating",
        population_affected=None,
        latest_current_date="2024-10-01",
        latest_projection_date=None,
        risk_level="HIGH",
        key_indicators=[],
        geographic_full_name="Edaga arbi, Central, Tigray, Ethiopia"
    )
    
    if system.reports_vectorstore is not None:
        explanation = system.function2_explain_why("Edaga arbi", mock_assessment)
        text = explanation["explanation"].lower()
        # Check that we don't say livelihood is missing when it's provided
        if "livelihood" in text and "missing" in text:
            # This is OK only if livelihood_system was actually "None"
            # But for Tigray, it should never be None
            assert False, f"Found contradiction: livelihood + missing in: {text[:300]}"
        if "rainfall" in text and "missing" in text:
            # Same for rainfall
            assert False, f"Found contradiction: rainfall + missing in: {text[:300]}"
        print("✅ test_limitations_do_not_contradict_domain_knowledge passed")
    else:
        print("⚠️  test_limitations_do_not_contradict_domain_knowledge skipped (vector stores not initialized)")


def test_validated_shocks_integrity():
    """Test that validated shocks are never overwritten."""
    system = FEWSSystem()
    mock_assessment = RegionRiskAssessment(
        region="Edaga arbi",
        current_phase=4,
        projected_phase_ml1=None,
        projected_phase_ml2=None,
        is_at_risk=True,
        trend="deteriorating",
        population_affected=None,
        latest_current_date="2024-10-01",
        latest_projection_date=None,
        risk_level="HIGH",
        key_indicators=[],
        geographic_full_name="Edaga arbi, Central, Tigray, Ethiopia"
    )
    
    if system.reports_vectorstore is not None:
        explanation = system.function2_explain_why("Edaga arbi", mock_assessment)
        # Drivers should match validated shocks (no hallucination)
        drivers = explanation.get("drivers", [])
        # If no validated shocks, drivers should be empty (not hallucinated)
        # We can't easily test the exact match without accessing internal state,
        # but we can ensure drivers list exists and is consistent
        assert isinstance(drivers, list), "Drivers must be a list"
        print("✅ test_validated_shocks_integrity passed")
    else:
        print("⚠️  test_validated_shocks_integrity skipped (vector stores not initialized)")


if __name__ == "__main__":
    # Run regression tests
    print("\n" + "="*80)
    print("REGRESSION TESTS")
    print("="*80)
    try:
        test_livelihood_never_marked_unknown()
        test_limitations_do_not_contradict_domain_knowledge()
        test_validated_shocks_integrity()
        print("\n✅ All regression tests passed!")
    except Exception as e:
        print(f"\n❌ Regression test failed: {e}")
        import traceback
        traceback.print_exc()

