"""
Quick test script for FEWS system
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import FEWSSystem

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

