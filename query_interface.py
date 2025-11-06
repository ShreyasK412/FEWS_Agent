"""
Interactive query interface for the RAG system.
"""

from typing import Dict
from rag_system import RAGSystem
from intent_detector import IntentDetector


class QueryInterface:
    """Interactive interface for querying the RAG system."""
    
    def __init__(self, rag_system: RAGSystem):
        self.rag_system = rag_system
        self.intent_detector = IntentDetector()
    
    def print_answer(self, result: Dict):
        """Pretty print the answer and sources with intent-aware formatting."""
        intent_info = result.get("intent_info", {})
        intent = intent_info.get("intent", "unknown")
        is_scenario = intent_info.get("is_scenario", False)
        
        print("\n" + "="*80)
        
        # Add intent indicator
        if is_scenario:
            print("📊 SCENARIO ANALYSIS")
        elif intent == "price":
            print("💰 PRICE DATA")
        elif intent == "context":
            print("🎯 HUMANITARIAN RESPONSE ANALYSIS")
        else:
            print("📋 ANALYSIS")
        
        print("="*80)
        print(result["result"])
        print("\n" + "-"*80)
        print("SOURCES:")
        print("-"*80)
        
        sources = result.get("source_documents", [])
        for i, doc in enumerate(sources, 1):
            source = doc.metadata.get("source", "Unknown")
            doc_type = doc.metadata.get("type", "Unknown")
            print(f"\n[{i}] Source: {source} ({doc_type})")
            print(f"    Preview: {doc.page_content[:200]}...")
        
        print("="*80 + "\n")
    
    def run_interactive(self):
        """Run interactive query loop."""
        print("\n" + "="*80)
        print("FOOD INSECURITY ANALYSIS SYSTEM FOR ETHIOPIA")
        print("="*80)
        print("\nAsk questions about food insecurity in Ethiopia.")
        print("The system will consult your documents and price data.")
        print("\nCommands:")
        print("  - Type your question and press Enter")
        print("  - Type 'exit' or 'quit' to exit")
        print("  - Type 'help' for example questions")
        print("\n" + "-"*80 + "\n")
        
        example_questions = [
            "What are the current maize prices in Addis Ababa?",
            "What can be done about the crisis in Ethiopia?",
            "What should we do about food insecurity in Tigray?",
            "How has the price of Teff changed over time?",
            "What if maize prices rise 20% in Oromia?",
            "What are the main factors affecting food insecurity in Ethiopia?",
            "What regions have the highest food prices and what interventions are needed?",
            "How can we help with the food crisis in Ethiopia?"
        ]
        
        while True:
            try:
                question = input("Question: ").strip()
                
                if not question:
                    continue
                
                if question.lower() in ['exit', 'quit', 'q']:
                    print("\nGoodbye!")
                    break
                
                if question.lower() == 'help':
                    print("\nExample questions:")
                    for i, eq in enumerate(example_questions, 1):
                        print(f"  {i}. {eq}")
                    print()
                    continue
                
                print("\nProcessing your question...")
                # Detect intent first
                intent_info = self.intent_detector.detect_intent(question)
                result = self.rag_system.query(question, intent_info=intent_info)
                self.print_answer(result)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}\n")
                continue
    
    def query_once(self, question: str):
        """Execute a single query and return result."""
        result = self.rag_system.query(question)
        self.print_answer(result)
        return result

