"""
Test script to verify shock detection improvements
"""
from src import FEWSSystem

def test_edaga_arbi():
    """Test shock detection for Edaga Arbi"""
    
    system = FEWSSystem()
    system.setup_vector_stores()
    
    # Get assessment
    assessment = system.ipc_parser.get_region_assessment("Edaga arbi")
    
    if not assessment:
        print("❌ Region 'Edaga arbi' not found")
        return
    
    print("\n" + "="*60)
    print("TESTING SHOCK DETECTION FOR EDAGA ARBI")
    print("="*60)
    
    # Run Function 2
    result = system.function2_explain_why("Edaga arbi", assessment)
    
    print("\n--- Detected Shocks ---")
    shocks = result.get('shocks_detailed', [])
    if shocks:
        for shock in shocks:
            print(f"\nType: {shock.get('type', 'unknown')}")
            print(f"Description: {shock.get('description', 'N/A')}")
            evidence = shock.get('evidence', 'None')
            print(f"Evidence: {evidence[:150]}..." if len(evidence) > 150 else f"Evidence: {evidence}")
            print(f"Confidence: {shock.get('confidence', 'unknown')}")
    else:
        print("No shocks detected")
    
    print("\n--- Drivers ---")
    print(result.get('drivers', []))
    
    print("\n--- Explanation Preview ---")
    explanation = result.get('explanation', '')
    if explanation:
        print(explanation[:500])
        print("...")
    else:
        print("No explanation generated")
    
    # Check quality
    print("\n" + "="*60)
    print("QUALITY CHECKS")
    print("="*60)
    
    # Check 1: Are shocks specific?
    generic_shocks = [s for s in shocks if s.get('description', '').lower() in ['weather', 'conflict', 'economic']]
    if generic_shocks:
        print("❌ FAIL: Found generic shock descriptions")
        for s in generic_shocks:
            print(f"   - {s.get('description')}")
    else:
        print("✅ PASS: All shocks have specific descriptions")
    
    # Check 2: Do shocks have evidence?
    shocks_with_evidence = [s for s in shocks if s.get('evidence') and len(s.get('evidence', '')) > 50]
    if shocks:
        if len(shocks_with_evidence) >= len(shocks) * 0.8:
            print(f"✅ PASS: {len(shocks_with_evidence)}/{len(shocks)} shocks have evidence")
        else:
            print(f"⚠️  WARN: Only {len(shocks_with_evidence)}/{len(shocks)} shocks have evidence")
    else:
        print("⚠️  WARN: No shocks detected")
    
    # Check 3: Are shocks region-appropriate?
    wrong_region_keywords = ['borena', 'somali region', 'pastoral', 'deyr', 'gu/genna', 'milk production', '288,000 displaced', 'borena-somali']
    explanation_lower = explanation.lower()
    wrong_keywords_found = [kw for kw in wrong_region_keywords if kw in explanation_lower]
    
    if wrong_keywords_found:
        print(f"❌ FAIL: Found wrong-region keywords: {wrong_keywords_found}")
    else:
        print("✅ PASS: No wrong-region keywords detected")
    
    # Check 4: Are shocks Tigray-specific?
    tigray_keywords = ['tigray', 'kiremt', 'meher', 'conflict 2020', 'fuel shortage', 'labor migration']
    tigray_keywords_found = [kw for kw in tigray_keywords if kw in explanation_lower]
    
    if tigray_keywords_found:
        print(f"✅ PASS: Found Tigray-specific keywords: {tigray_keywords_found[:3]}")
    else:
        print("⚠️  WARN: No Tigray-specific keywords found")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    test_edaga_arbi()

