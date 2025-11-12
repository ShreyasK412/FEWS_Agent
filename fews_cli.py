"""
FEWS CLI - Command-line interface for the Famine Early Warning System
"""
import sys
from src import FEWSSystem, RegionRiskAssessment


def print_assessment(assessment: RegionRiskAssessment):
    """Print a formatted risk assessment."""
    risk_emoji = "🔴" if assessment.risk_level == "HIGH" else "🟡" if assessment.risk_level == "MEDIUM" else "🟢"
    print(f"\n{risk_emoji} {assessment.region} ({assessment.geographic_full_name})")
    print(f"   Current IPC Phase: {assessment.current_phase}")
    if assessment.projected_phase_ml1:
        print(f"   Projected (Near-term): Phase {assessment.projected_phase_ml1}")
    if assessment.projected_phase_ml2:
        print(f"   Projected (Medium-term): Phase {assessment.projected_phase_ml2}")
    print(f"   Risk Level: {assessment.risk_level}")
    print(f"   Trend: {assessment.trend}")
    print(f"   Indicators: {', '.join(assessment.key_indicators)}")


def main():
    """Main CLI interface."""
    print("="*80)
    print("FAMINE EARLY WARNING SYSTEM (FEWS)")
    print("="*80)
    print("\nInitializing system...")
    
    # Initialize system
    system = FEWSSystem()
    
    # Setup vector stores (this may take a while first time)
    print("\nSetting up vector stores...")
    print("   Note: If vector stores already exist, this will be fast.")
    print("   If creating new ones, this may take 30-60 minutes for large PDFs.")
    system.setup_vector_stores()
    
    while True:
        print("\n" + "="*80)
        print("MAIN MENU")
        print("="*80)
        print("1. Identify at-risk regions")
        print("2. Explain why a region is at risk")
        print("3. Recommend interventions for a region")
        print("4. Full analysis (all 3 functions) for a region")
        print("5. Exit")
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == "1":
            # Function 1: Identify at-risk regions
            print("\n" + "="*80)
            print("IDENTIFYING AT-RISK REGIONS")
            print("="*80)
            
            at_risk = system.function1_identify_at_risk_regions()
            
            if not at_risk:
                print("\n⚠️  No at-risk regions identified.")
                continue
            
            # Show summary
            high_risk = [a for a in at_risk if a.risk_level == "HIGH"]
            medium_risk = [a for a in at_risk if a.risk_level == "MEDIUM"]
            
            print(f"\n📊 SUMMARY:")
            print(f"   Total at-risk regions: {len(at_risk)}")
            print(f"   High Risk (IPC 4+): {len(high_risk)}")
            print(f"   Medium Risk (IPC 3): {len(medium_risk)}")
            
            # Show top regions
            show_all = input("\nShow all regions? (y/n): ").strip().lower() == 'y'
            regions_to_show = at_risk if show_all else at_risk[:20]
            
            for i, assessment in enumerate(regions_to_show, 1):
                print_assessment(assessment)
            
            if not show_all and len(at_risk) > 20:
                print(f"\n... and {len(at_risk) - 20} more regions")
        
        elif choice == "2":
            # Function 2: Explain why
            region = input("\nEnter region name: ").strip()
            if not region:
                print("⚠️  Region name required")
                continue
            
            print(f"\nAnalyzing {region}...")
            assessment = system.ipc_parser.get_region_assessment(region)
            
            if assessment is None:
                print(f"❌ Region '{region}' not found in IPC data.")
                # Suggest similar regions
                all_regions = system.ipc_parser.identify_at_risk_regions(min_phase=1, include_projected=True, include_deteriorating=True)
                region_lower = region.lower()
                suggestions = []
                for r in all_regions[:100]:  # Check first 100
                    if region_lower in r.region.lower() or r.region.lower() in region_lower:
                        suggestions.append(r.region)
                    elif any(word in r.region.lower() for word in region_lower.split() if len(word) > 3):
                        suggestions.append(r.region)
                
                if suggestions:
                    print(f"\n💡 Did you mean one of these?")
                    for sug in suggestions[:5]:
                        print(f"   - {sug}")
                continue
            
            if not assessment.is_at_risk:
                print(f"ℹ️  {region} is not currently flagged as at-risk (IPC Phase {assessment.current_phase})")
                continue
            
            result = system.function2_explain_why(region, assessment)
            
            print("\n" + "="*80)
            print(f"EXPLANATION FOR {region.upper()}")
            print("="*80)
            print(f"\nIPC Phase: {result['ipc_phase']}")
            print(f"Data Quality: {result['data_quality']}")
            print(f"\n{result['explanation']}")
            
            if result['drivers']:
                print(f"\nIdentified Drivers: {', '.join(result['drivers'])}")
            
            if result['sources']:
                print(f"\nSources: {', '.join(result['sources'])}")
            
            if result['data_quality'] == 'insufficient':
                print("\n⚠️  Note: Insufficient data found. This has been logged to missing_info.log")
        
        elif choice == "3":
            # Function 3: Recommend interventions
            region = input("\nEnter region name: ").strip()
            if not region:
                print("⚠️  Region name required")
                continue
            
            print(f"\nAnalyzing {region}...")
            assessment = system.ipc_parser.get_region_assessment(region)
            
            if assessment is None:
                print(f"❌ Region '{region}' not found in IPC data.")
                # Suggest similar regions
                all_regions = system.ipc_parser.identify_at_risk_regions(min_phase=1, include_projected=True, include_deteriorating=True)
                region_lower = region.lower()
                suggestions = []
                for r in all_regions[:100]:  # Check first 100
                    if region_lower in r.region.lower() or r.region.lower() in region_lower:
                        suggestions.append(r.region)
                    elif any(word in r.region.lower() for word in region_lower.split() if len(word) > 3):
                        suggestions.append(r.region)
                
                if suggestions:
                    print(f"\n💡 Did you mean one of these?")
                    for sug in suggestions[:5]:
                        print(f"   - {sug}")
                continue
            
            # Optionally get drivers first
            get_drivers = input("Get drivers first? (y/n): ").strip().lower() == 'y'
            drivers = None
            
            if get_drivers:
                explanation = system.function2_explain_why(region, assessment)
                drivers = explanation.get('drivers', [])
                print(f"\nIdentified drivers: {', '.join(drivers) if drivers else 'None identified'}")
            
            result = system.function3_recommend_interventions(region, assessment, drivers)
            
            print("\n" + "="*80)
            print(f"INTERVENTION RECOMMENDATIONS FOR {region.upper()}")
            print("="*80)
            print(f"\n{result['recommendations']}")
            
            if result['sources']:
                print(f"\nSources: {', '.join(result['sources'])}")
            
            if result['limitations']:
                print(f"\n⚠️  Limitations: {result['limitations']}")
        
        elif choice == "4":
            # Full analysis
            region = input("\nEnter region name: ").strip()
            if not region:
                print("⚠️  Region name required")
                continue
            
            print(f"\n{'='*80}")
            print(f"FULL ANALYSIS FOR {region.upper()}")
            print("="*80)
            
            # Get assessment
            assessment = system.ipc_parser.get_region_assessment(region)
            if assessment is None:
                print(f"❌ Region '{region}' not found in IPC data.")
                continue
            
            # Print assessment
            print("\n1. RISK ASSESSMENT:")
            print_assessment(assessment)
            
            if not assessment.is_at_risk:
                print(f"\nℹ️  {region} is not currently flagged as at-risk.")
                continue
            
            # Explain why
            print("\n2. WHY IS THIS REGION AT RISK?")
            explanation = system.function2_explain_why(region, assessment)
            print(f"\n{explanation['explanation']}")
            if explanation['drivers']:
                print(f"\nDrivers: {', '.join(explanation['drivers'])}")
            
            # Recommend interventions
            print("\n3. INTERVENTION RECOMMENDATIONS:")
            interventions = system.function3_recommend_interventions(
                region, 
                assessment, 
                explanation.get('drivers', [])
            )
            print(f"\n{interventions['recommendations']}")
            
            if interventions['sources']:
                print(f"\nSources: {', '.join(interventions['sources'])}")
        
        elif choice == "5":
            print("\nExiting...")
            break
        
        else:
            print("⚠️  Invalid option. Please select 1-5.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

