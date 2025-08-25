import asyncio
import sys
import os

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.services.qdrant_service import qdrant_hybrid_service
from app.db.qdrant_client import connect_to_qdrant
from app.core.config import settings

async def test_qdrant_hybrid_service():
    """Test the Qdrant hybrid service functionality"""
    
    print("🚀 Testing Qdrant Hybrid Service")
    print("=" * 50)
    
    try:
        # Initialize connection
        print("1. Connecting to Qdrant...")
        connect_to_qdrant()
        print("✅ Connected successfully")
        
        # Create hybrid collection
        print("\n2. Creating hybrid collection...")
        success = await qdrant_hybrid_service.create_hybrid_collection(force_recreate=True)
        if success:
            print("✅ Hybrid collection created successfully")
        else:
            print("❌ Failed to create hybrid collection")
            return
        
        # Test document insertion
        print("\n3. Testing document insertion...")
        test_docs = [
            {
                "id": "doc1",
                "title": "Machine Learning Basics",
                "content": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed.",
                "tags": ["ml", "ai", "education"],
                "metadata": {"category": "tutorial", "difficulty": "beginner"}
            },
            {
                "id": "doc2", 
                "title": "Deep Learning Neural Networks",
                "content": "Deep learning uses artificial neural networks with multiple layers to model and understand complex patterns in data.",
                "tags": ["deep-learning", "neural-networks", "ai"],
                "metadata": {"category": "advanced", "difficulty": "expert"}
            },
            {
                "id": "doc3",
                "title": "Python Programming Guide", 
                "content": "Python is a high-level programming language known for its simplicity and readability. It's widely used in data science and machine learning.",
                "tags": ["python", "programming", "guide"],
                "metadata": {"category": "programming", "difficulty": "intermediate"}
            }
        ]
        
        for doc in test_docs:
            success = await qdrant_hybrid_service.upsert_document(
                document_id=doc["id"],
                content=doc["content"],
                title=doc["title"],
                tags=doc["tags"],
                metadata=doc["metadata"]
            )
            if success:
                print(f"✅ Document {doc['id']} inserted successfully")
            else:
                print(f"❌ Failed to insert document {doc['id']}")
        
        # Test document count
        print("\n4. Testing document count...")
        count = await qdrant_hybrid_service.count_documents()
        print(f"📊 Total documents: {count}")
        
        # Test hybrid search
        print("\n5. Testing hybrid search...")
        search_queries = [
            "machine learning artificial intelligence",
            "python programming",
            "neural networks deep learning"
        ]
        
        for query in search_queries:
            print(f"\n🔍 Searching for: '{query}'")
            results = await qdrant_hybrid_service.hybrid_search(
                query=query,
                limit=5,
                score_threshold=0.1
            )
            
            if results:
                print(f"Found {len(results)} results:")
                for i, result in enumerate(results[:3], 1):
                    title = result["payload"].get("title", "Untitled")
                    score = result["score"]
                    dense_score = result["dense_score"]
                    sparse_score = result["sparse_score"]
                    print(f"  {i}. {title} (Score: {score:.3f}, Dense: {dense_score:.3f}, Sparse: {sparse_score:.3f})")
            else:
                print("  No results found")
        
        # Test semantic search
        print("\n6. Testing semantic search...")
        semantic_query = "artificial intelligence learning"
        print(f"🧠 Semantic search for: '{semantic_query}'")
        semantic_results = await qdrant_hybrid_service.semantic_search(
            query=semantic_query,
            limit=3,
            score_threshold=0.1
        )
        
        if semantic_results:
            print(f"Found {len(semantic_results)} semantic results:")
            for i, result in enumerate(semantic_results, 1):
                title = result["payload"].get("title", "Untitled")
                score = result["score"]
                print(f"  {i}. {title} (Score: {score:.3f})")
        
        # Test keyword search
        print("\n7. Testing keyword search...")
        keyword_query = "python programming"
        print(f"🔤 Keyword search for: '{keyword_query}'")
        keyword_results = await qdrant_hybrid_service.keyword_search(
            query=keyword_query,
            limit=3,
            score_threshold=0.1
        )
        
        if keyword_results:
            print(f"Found {len(keyword_results)} keyword results:")
            for i, result in enumerate(keyword_results, 1):
                title = result["payload"].get("title", "Untitled")
                score = result["score"]
                print(f"  {i}. {title} (Score: {score:.3f})")
        
        # Test filtered search
        print("\n8. Testing filtered search...")
        filtered_results = await qdrant_hybrid_service.hybrid_search(
            query="machine learning",
            tags=["ai"],
            metadata_filters={"difficulty": "beginner"},
            limit=5
        )
        
        print(f"🎯 Filtered search (tags=['ai'], difficulty='beginner'):")
        if filtered_results:
            print(f"Found {len(filtered_results)} filtered results:")
            for i, result in enumerate(filtered_results, 1):
                title = result["payload"].get("title", "Untitled")
                score = result["score"]
                tags = result["payload"].get("tags", [])
                difficulty = result["payload"].get("metadata", {}).get("difficulty", "unknown")
                print(f"  {i}. {title} (Score: {score:.3f}, Tags: {tags}, Difficulty: {difficulty})")
        else:
            print("  No filtered results found")
        
        # Test document retrieval
        print("\n9. Testing document retrieval...")
        doc = await qdrant_hybrid_service.get_document("doc1")
        if doc:
            print(f"✅ Retrieved document: {doc['payload']['title']}")
        else:
            print("❌ Failed to retrieve document")
        
        print("\n🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_qdrant_hybrid_service())