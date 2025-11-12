"""
Check if vector store has Tigray-specific content
"""
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

def check_tigray_content():
    try:
        embeddings = OllamaEmbeddings(model="llama3.2")
        vectorstore = Chroma(
            persist_directory="chroma_db_reports",
            embedding_function=embeddings
        )
    except Exception as e:
        print(f"❌ Cannot connect to vector store: {str(e)}")
        print("Please run: python3 fews_cli.py (to create vector store)")
        return
    
    test_queries = [
        "Tigray food security 2025",
        "Edaga arbi Central Zone",
        "kiremt delayed rains northern Ethiopia",
        "Tigray conflict labor migration",
        "fuel shortage Amhara Tigray",
    ]
    
    print("\n" + "="*70)
    print("VECTOR STORE QUALITY CHECK")
    print("="*70)
    
    total_results = 0
    tigray_results = 0
    wrong_region_results = 0
    generic_results = 0
    
    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"Query: {query}")
        print('='*70)
        
        try:
            results = vectorstore.similarity_search(query, k=3)
        except Exception as e:
            print(f"Error retrieving results: {str(e)}")
            continue
        
        for i, doc in enumerate(results, 1):
            total_results += 1
            print(f"\nResult {i}:")
            preview = doc.page_content[:400].replace('\n', ' ')
            print(preview)
            
            # Check if relevant
            content_lower = doc.page_content.lower()
            
            if any(word in content_lower for word in ['tigray', 'amhara', 'northern', 'edaga', 'kiremt', 'meher']):
                print("✅ RELEVANT (mentions northern regions/highland cropping)")
                tigray_results += 1
            elif any(word in content_lower for word in ['borena', 'somali region', 'pastoral', 'deyr', 'gu/genna', 'milk']):
                print("❌ WRONG REGION (southern pastoral content)")
                wrong_region_results += 1
            else:
                print("⚠️  GENERIC (no clear regional indicator)")
                generic_results += 1
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total results checked: {total_results}")
    print(f"✅ Tigray-relevant: {tigray_results} ({tigray_results*100//total_results if total_results else 0}%)")
    print(f"❌ Wrong region: {wrong_region_results} ({wrong_region_results*100//total_results if total_results else 0}%)")
    print(f"⚠️  Generic: {generic_results} ({generic_results*100//total_results if total_results else 0}%)")
    
    if wrong_region_results > total_results * 0.5:
        print("\n🚨 CRITICAL: Vector store contains mostly wrong-region content!")
        print("   Recreate with: rm -rf chroma_db_reports/ && python3 fews_cli.py")
    elif tigray_results < total_results * 0.5:
        print("\n⚠️  WARNING: Vector store has insufficient Tigray content")
        print("   Consider recreating or adding manual regional sections")
    else:
        print("\n✅ Vector store appears healthy for Tigray queries")

if __name__ == "__main__":
    check_tigray_content()

