"""
FEWS-aligned knowledge library for Ethiopia food security analysis.
Provides standardized seasonal calendars, livelihood impact templates,
and intervention guidance based on FEWS NET documents.
"""

from typing import Dict, List

# Seasonal calendar profiles per livelihood zone
# Based on FEWS NET Ethiopia Food Security Outlook
SEASONAL_CALENDARS = {
    "northern_cropping": {
        "zone_description": "Northern cropping areas (Amhara and Tigray highlands)",
        "livelihood_system": "Rainfed cereal cropping",
        "seasons": [
            {
                "name": "Belg",
                "months": "February–May",
                "description": "Secondary rainfall season; starts planting. Households often rely on remaining stocks from meher harvest."
            },
            {
                "name": "Kiremt",
                "months": "June–September",
                "description": "Main rainy season. Peak growing season for cereals. Also main lean season when food stocks deplete."
            },
            {
                "name": "Meher harvest",
                "months": "September–October",
                "description": "Begin of main harvest (sorghum, teff, maize). Replenishes food stocks for the year."
            },
            {
                "name": "Dry season",
                "months": "November–January",
                "description": "Post-harvest; relatively better food access if harvest was average or above. Land preparation for next season."
            }
        ],
        "lean_season_note": "The main lean season typically aligns with the kiremt (main rainy season, Jun–Sep) when food stocks are low and newly harvested green crops are not yet available.",
        "harvest_timing": "Meher harvest occurs September–October; delayed rains push this later, stretching the lean season."
    },
    "pastoral_south": {
        "zone_description": "Pastoral areas (Somali, Borena, South Omo)",
        "livelihood_system": "Pastoral/agropastoral livestock herding",
        "seasons": [
            {
                "name": "Gu/Genna",
                "months": "March–May",
                "description": "Main pastoral rainfall season. Fodder availability improves; milk production increases."
            },
            {
                "name": "Deyr/Hageya",
                "months": "October–December",
                "description": "Secondary pastoral rainfall season. Additional water and pasture availability."
            },
            {
                "name": "Dry season 1",
                "months": "January–February",
                "description": "Dry period between seasons; water and pasture stress."
            },
            {
                "name": "Dry season 2",
                "months": "June–September",
                "description": "Extended dry period; pastoralists migrate for water/pasture. Risk period for livestock mortality."
            }
        ],
        "lean_season_note": "Dry seasons (especially June–September) are critical stress periods with water scarcity, poor pasture, and high livestock mortality risk.",
        "harvest_timing": "No crop harvest; pastoral wealth is in livestock condition and milk availability."
    }
}

# Livelihood impact phrases by (livelihood_zone, shock_type)
# Used to ground Function 2 section E. Livelihood Impacts in FEWS text
LIVELIHOOD_IMPACT_TEMPLATES = {
    ("northern_cropping", "economic"): [
        "Households face reduced purchasing power as wage income does not keep pace with high staple prices. They reduce meal size and frequency, increase reliance on market purchases, and engage in coping strategies such as selling livestock, sending children to relatives, or risky migration.",
        "Constrained household income combined with persistently high food prices limits food and income access, reducing purchasing power and increasing reliance on market-purchased foods rather than own production.",
        "Economic shocks force households to liquidate assets (selling livestock, tools) earlier than normal seasonal patterns, weakening their ability to absorb future shocks."
    ],
    ("northern_cropping", "weather"): [
        "Delayed or below-average rainfall constrains own-produced food stocks, making households more dependent on markets and wage income to purchase food. Reduced harvest compounds the usual lean-season food gap.",
        "Moisture deficits during the growing season lower crop yields and harvest quality, resulting in shorter food stock duration and an extended lean season. Households exhaust own stores earlier and depend more on purchases.",
        "Erratic rainfall (late onset, early cessation) disrupts planting schedules and reduces production potential, leaving households with insufficient own food to meet annual needs."
    ],
    ("northern_cropping", "conflict"): [
        "Conflict-related constraints on labor migration reduce households' ability to earn seasonal income in sesame-producing zones. Limited off-farm income reduces purchasing power and food access.",
        "Insecurity and road blockages disrupt market functioning and raise transport costs, increasing food prices locally and reducing market supply. Households face both higher prices and reduced income.",
        "Displacement and population movements strain local community food resources and reduce households' ability to access productive assets and livelihoods."
    ],
    ("pastoral_south", "weather"): [
        "Poor rainfall during pastoral seasons (gu/genna, deyr/hageya) reduces fodder availability and water sources. Livestock body condition declines, milk production drops, and herd mortality risk increases.",
        "Extended dry seasons (Jun–Sep, Jan–Feb) force pastoralists to migrate beyond normal ranges, increasing expenditure on transport and feed purchases while reducing milk and livestock sales income.",
        "Drought-related livestock mortality removes productive assets and future breeding stock, reducing pastoral wealth and food/income access for years."
    ],
    ("pastoral_south", "conflict"): [
        "Intercommunal conflict restricts pastoral mobility and access to traditional water/pasture zones. Households must graze in marginal areas, increasing herd stress and reducing productivity.",
        "Displacement due to conflict separates herds from owners or forces rapid herd depletion to avoid capture. Displaced households lose productive assets and income.",
        "Insecurity raises costs of accessing markets for feed purchases and livestock sales, reducing net income and food access."
    ]
}

# IPC Phase definitions (to prevent hallucination and famine language)
IPC_PHASE_DEFINITIONS = {
    1: {
        "name": "Minimal/Stressed",
        "description": "Food consumption is adequate; minimal people in 'stressed' food consumption."
    },
    2: {
        "name": "Stressed",
        "description": "Households have adequate food, but are unable to afford additional non-food expenses."
    },
    3: {
        "name": "Crisis",
        "description": "Food consumption gaps exist and/or productive asset losses. Insufficient coping capacity to offset the gaps."
    },
    4: {
        "name": "Emergency",
        "description": "Severe food consumption gaps and/or extreme productive/livelihood asset losses. Extreme coping strategies."
    },
    5: {
        "name": "Catastrophe/Famine",
        "description": "Near-complete destitution, starvation, or death from hunger is likely or occurring."
    }
}

# Driver to intervention mapping for consistent Function 3 output
# Keys must match driver_interventions.json exactly
DRIVER_SHOCK_MAPPING = {
    "economic": {
        "shock_keywords": ["high food prices", "inflation", "currency depreciation", "market disruption", "purchasing power"],
        "intervention_domains": ["food_assistance", "cash_assistance", "market_support", "livelihood_protection"],
        "sphere_reference": "CALP (Cash and Voucher Assistance)",
        "key_interventions": [
            "Multi-purpose cash assistance (MPC) where markets function",
            "Commodity vouchers for essential foods",
            "Trader support to restore supply chains",
            "Livelihood asset support to maintain income generation"
        ]
    },
    "weather": {
        "shock_keywords": ["drought", "delayed rains", "below-average rainfall", "moisture deficit", "crop failure"],
        "intervention_domains": ["food_assistance", "agricultural_support", "livelihood_protection", "water"],
        "sphere_reference": "FAO (Food and Agriculture Organization) emergency guidelines",
        "key_interventions": [
            "General food distribution (GFD) or food-for-work where production fails",
            "Seeds and tools for next season planting",
            "Soil and water conservation (SWC) activities",
            "Water supply and conservation measures"
        ]
    },
    "conflict": {
        "shock_keywords": ["conflict", "insecurity", "displacement", "violence", "road blockage", "clashes"],
        "intervention_domains": ["food_assistance", "protection", "cash_assistance", "logistics"],
        "sphere_reference": "Sphere Humanitarian Charter (Protection and Food Security)",
        "key_interventions": [
            "Unconditional food assistance or cash in conflict-affected areas",
            "Safe humanitarian access corridors",
            "Protection mainstreaming (safeguarding, GBV prevention)",
            "Support for displaced and host populations"
        ]
    },
    "displacement": {
        "shock_keywords": ["displacement", "IDP", "refugee", "population movement"],
        "intervention_domains": ["food_assistance", "wash", "protection", "shelter"],
        "sphere_reference": "Sphere WASH and Protection standards",
        "key_interventions": [
            "Targeted food assistance for displaced populations",
            "WASH support (water trucking, sanitation facilities)",
            "Shelter support and non-food items",
            "Psychosocial support and community cohesion activities"
        ]
    }
}

def get_seasonal_calendar(livelihood_zone: str) -> Dict:
    """Retrieve correct seasonal calendar for a livelihood zone."""
    return SEASONAL_CALENDARS.get(livelihood_zone, {})

def get_livelihood_impacts(livelihood_zone: str, shock_type: str) -> List[str]:
    """Retrieve FEWS-aligned impact templates for a shock in a livelihood zone."""
    key = (livelihood_zone, shock_type)
    return LIVELIHOOD_IMPACT_TEMPLATES.get(key, [])

def get_ipc_phase_definition(phase: int) -> Dict:
    """Get IPC Phase definition to prevent hallucination."""
    return IPC_PHASE_DEFINITIONS.get(phase, {})

def get_driver_metadata(driver: str) -> Dict:
    """Get driver-to-intervention metadata for Function 3."""
    return DRIVER_SHOCK_MAPPING.get(driver, {})

def validate_shock_against_library(shock_type: str) -> bool:
    """Check if shock type is in the valid library."""
    valid_types = set()
    for mapping in DRIVER_SHOCK_MAPPING.values():
        valid_types.update(mapping.get("shock_keywords", []))
    return any(keyword in shock_type.lower() for keyword in valid_types)

