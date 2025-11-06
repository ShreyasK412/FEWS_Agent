"""
Intent detection for determining query type and appropriate response format.
"""

from typing import Dict, Tuple


class IntentDetector:
    """Detects user intent to determine response format."""
    
    # Keywords for price/market queries
    PRICE_KEYWORDS = [
        'price', 'prices', 'cost', 'costs', 'market', 'markets',
        'commodity', 'commodities', 'etb', 'usd', 'currency',
        'wheat', 'maize', 'teff', 'sorghum', 'barley', 'rice',
        'how much', 'what is the price', 'pricing'
    ]
    
    # Keywords for situation/context/humanitarian queries
    CONTEXT_KEYWORDS = [
        'what should', 'what can', 'how can', 'what to do',
        'crisis', 'situation', 'condition', 'status',
        'help', 'assist', 'support', 'intervention', 'response',
        'action', 'recommend', 'suggest', 'priority', 'urgent',
        'funding', 'gap', 'ration', 'food security', 'insecurity',
        'humanitarian', 'aid', 'assistance', 'need', 'needs'
    ]
    
    # Keywords for scenario/hypothetical queries
    SCENARIO_KEYWORDS = [
        'if', 'what if', 'suppose', 'imagine', 'scenario',
        'would', 'could', 'might', 'projection', 'forecast'
    ]
    
    def detect_intent(self, query: str) -> Dict[str, any]:
        """
        Detect query intent and determine response format.
        
        Returns:
            Dict with:
            - intent: 'price', 'context', 'scenario', or 'mixed'
            - is_scenario: bool
            - needs_structured_response: bool (True for context queries)
            - audience: 'analyst' or 'general' (can be inferred)
        """
        query_lower = query.lower()
        
        # Check for scenario keywords
        is_scenario = any(keyword in query_lower for keyword in self.SCENARIO_KEYWORDS)
        
        # Count keyword matches
        price_matches = sum(1 for kw in self.PRICE_KEYWORDS if kw in query_lower)
        context_matches = sum(1 for kw in self.CONTEXT_KEYWORDS if kw in query_lower)
        
        # Determine primary intent
        if context_matches > price_matches and context_matches > 0:
            intent = 'context'
        elif price_matches > 0 and context_matches == 0:
            intent = 'price'
        elif context_matches > 0 and price_matches > 0:
            intent = 'mixed'  # Could be price in context of crisis
        else:
            # Default to context if no clear match
            intent = 'context'
        
        # Determine if structured response needed
        needs_structured = intent in ['context', 'mixed'] or is_scenario
        
        # Infer audience (can be enhanced later)
        # For now, assume analyst if query is technical/complex
        audience = 'analyst' if any(word in query_lower for word in 
                                   ['policy', 'recommend', 'implication', 'analysis']) else 'general'
        
        return {
            'intent': intent,
            'is_scenario': is_scenario,
            'needs_structured_response': needs_structured,
            'audience': audience,
            'price_keywords_found': price_matches,
            'context_keywords_found': context_matches
        }

