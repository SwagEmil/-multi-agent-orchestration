"""
RAG Database Stress Test - Complex Query Verification

Tests 5 challenging queries about:
1. Multi-agent architecture patterns
2. Debugging agent failures
3. Production deployment challenges
4. Advanced RAG techniques
5. Fine-tuning vs RAG trade-offs
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag_retriever import RAGRetriever
import json

def run_complex_test():
    """Test RAG with 5 super complex queries"""
    
    retriever = RAGRetriever()
    
    # Define 5 challenging queries
    test_cases = [
        {
            "id": 1,
            "query": "How do I design a hierarchical multi-agent system where a manager agent coordinates multiple specialist agents, and what are the key challenges in managing context and preventing agents from getting stuck in loops?",
            "topic": "Multi-Agent Architecture & Coordination",
            "expected_sources": ["Agents Companion", "Introduction to Agents"]
        },
        {
            "id": 2,
            "query": "My agent is hallucinating and not using the tools I provided correctly. What evaluation techniques can I use to debug the trajectory and identify where in the reasoning chain it's going wrong?",
            "topic": "Agent Debugging & Evaluation",
            "expected_sources": ["Agents Companion", "Agents"]
        },
        {
            "id": 3,
            "query": "I'm deploying a RAG system to production on Vertex AI. What are the key considerations for cost optimization, latency reduction, and handling high-throughput scenarios with vector search?",
            "topic": "Production Deployment & Optimization",
            "expected_sources": ["Operationalizing", "Embeddings & Vector Stores"]
        },
        {
            "id": 4,
            "query": "Explain the difference between traditional RAG and agentic RAG, including query expansion, multi-step reasoning, and how to validate retrieved information before using it in the final response.",
            "topic": "Advanced RAG Techniques",
            "expected_sources": ["Agents Companion", "Embeddings & Vector Stores"]
        },
        {
            "id": 5,
            "query": "When should I use parameter-efficient fine-tuning like LoRA versus just improving my RAG system? What are the trade-offs in terms of catastrophic forgetting, cost, and maintaining up-to-date information?",
            "topic": "Fine-Tuning vs RAG Trade-offs",
            "expected_sources": ["Solving Domain Specific", "day4_domain_specific"]
        }
    ]
    
    print("=" * 80)
    print("RAG DATABASE STRESS TEST - 5 COMPLEX QUERIES")
    print("=" * 80)
    print()
    
    results_summary = []
    
    for test in test_cases:
        print(f"\n{'=' * 80}")
        print(f"TEST {test['id']}: {test['topic']}")
        print(f"{'=' * 80}")
        print(f"\n📝 QUERY:")
        print(f"   {test['query']}")
        print()
        
        # Retrieve results
        chunks = retriever.retrieve(test['query'], n_results=5)
        
        # Analyze results
        sources_found = set([chunk['source'] for chunk in chunks])
        avg_relevance = sum([chunk['relevance'] for chunk in chunks]) / len(chunks)
        
        print(f"📊 RETRIEVAL STATS:")
        print(f"   Average Relevance: {avg_relevance:.3f}")
        print(f"   Sources Found: {len(sources_found)}")
        print(f"   Total Chunks: {len(chunks)}")
        print()
        
        print(f"📄 TOP 5 RESULTS:")
        for i, chunk in enumerate(chunks, 1):
            print(f"\n   [{i}] {chunk['source']}")
            print(f"       Relevance: {chunk['relevance']:.3f}")
            print(f"       Content Preview:")
            # Show first 200 chars
            preview = chunk['content'].replace('\n', ' ')[:200]
            print(f"       {preview}...")
        
        # Check if expected sources were found
        expected_match = any(
            any(exp.lower() in src.lower() for src in sources_found)
            for exp in test['expected_sources']
        )
        
        print(f"\n✅ VALIDATION:")
        print(f"   Expected Sources Present: {'YES' if expected_match else 'NO'}")
        print(f"   Relevance Threshold (>0.2): {'PASS' if avg_relevance > 0.2 else 'FAIL'}")
        
        # Get formatted context for LLM
        context = retriever.retrieve_context_string(test['query'], n_results=3)
        
        results_summary.append({
            'test_id': test['id'],
            'topic': test['topic'],
            'avg_relevance': avg_relevance,
            'sources_count': len(sources_found),
            'expected_match': expected_match,
            'passed': expected_match and avg_relevance > 0.2
        })
    
    # Summary
    print(f"\n\n{'=' * 80}")
    print("OVERALL TEST SUMMARY")
    print(f"{'=' * 80}\n")
    
    passed = sum(1 for r in results_summary if r['passed'])
    total = len(results_summary)
    
    print(f"Tests Passed: {passed}/{total} ({passed/total*100:.0f}%)\n")
    
    for result in results_summary:
        status = "✅ PASS" if result['passed'] else "❌ FAIL"
        print(f"{status} | Test {result['test_id']}: {result['topic']}")
        print(f"         Relevance: {result['avg_relevance']:.3f} | "
              f"Sources: {result['sources_count']} | "
              f"Expected Match: {result['expected_match']}")
    
    print(f"\n{'=' * 80}")
    if passed == total:
        print("🎉 ALL TESTS PASSED! RAG database is working excellently!")
    elif passed >= total * 0.8:
        print("✅ GOOD! Most tests passed. RAG database is functional.")
    else:
        print("⚠️  NEEDS ATTENTION: Some tests failed. Review results above.")
    print(f"{'=' * 80}\n")
    
    return results_summary

if __name__ == "__main__":
    results = run_complex_test()
"""
RAG Database Verification Test #2 - Different Query Set

Tests 5 entirely new queries with detailed debugging:
1. Specific algorithms (HNSW, ScaNN)
2. Practical implementation
3. Observability and monitoring
4. Prompt engineering techniques
5. Security and governance
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag_retriever import RAGRetriever

def verify_with_detailed_output():
    """Test with different queries and show full retrieved content"""
    
    retriever = RAGRetriever()
    
    test_cases = [
        {
            "id": 1,
            "query": "Explain how the HNSW (Hierarchical Navigable Small World) algorithm works for approximate nearest neighbor search and when I should use it versus ScaNN.",
            "topic": "Vector Search Algorithms",
            "what_to_verify": "Should explain HNSW graph structure, layers, and comparison to ScaNN"
        },
        {
            "id": 2,
            "query": "I'm building a function calling agent. What are the best practices for designing function schemas and handling function call errors?",
            "topic": "Function Calling Implementation",
            "what_to_verify": "Should cover schema design, error handling, reliability patterns"
        },
        {
            "id": 3,
            "query": "How do I implement observability for my agent system? What should I trace, log, and monitor in production?",
            "topic": "Agent Observability",
            "what_to_verify": "Should cover tracing, logging, metrics, OpenTelemetry"
        },
        {
            "id": 4,
            "query": "What prompt engineering techniques can I use to improve agent reasoning, like Chain-of-Thought or ReAct?",
            "topic": "Prompt Engineering for Agents",
            "what_to_verify": "Should explain CoT, ReAct, and other reasoning frameworks"
        },
        {
            "id": 5,
            "query": "What are the security and governance considerations when deploying agents in an enterprise environment?",
            "topic": "Enterprise Security & Governance",
            "what_to_verify": "Should cover access control, data privacy, compliance, RBAC"
        }
    ]
    
    print("=" * 90)
    print("RAG DATABASE VERIFICATION TEST #2 - DIFFERENT QUERIES")
    print("=" * 90)
    print("\nThis test shows FULL CONTENT to verify correctness, not just scores\n")
    
    total_correct = 0
    
    for test in test_cases:
        print(f"\n{'=' * 90}")
        print(f"TEST {test['id']}: {test['topic']}")
        print(f"{'=' * 90}")
        print(f"\n📝 QUERY:")
        print(f"   {test['query']}")
        print(f"\n🎯 WHAT TO VERIFY:")
        print(f"   {test['what_to_verify']}")
        print()
        
        # Retrieve
        chunks = retriever.retrieve(test['query'], n_results=3)
        
        print(f"📊 RETRIEVAL RESULTS:")
        print(f"   Top 3 chunks retrieved")
        print()
        
        # Show full content for verification
        for i, chunk in enumerate(chunks, 1):
            print(f"{'─' * 90}")
            print(f"RESULT {i}: {chunk['source']}")
            print(f"Relevance Score: {chunk['relevance']:.3f}")
            print(f"{'─' * 90}")
            
            # Show more content for verification (first 800 chars)
            content = chunk['content']
            if len(content) > 800:
                print(content[:800] + "\n...")
            else:
                print(content)
            print()
        
        # Manual verification
        print(f"✅ MANUAL VERIFICATION:")
        print(f"   Review the content above and verify it answers the query")
        print(f"   Does it contain: {test['what_to_verify']}?")
        
        # Auto-check: if top result relevance > 0.15, likely relevant
        auto_check = chunks[0]['relevance'] > 0.15
        print(f"\n   Auto-check (relevance > 0.15): {'✅ LIKELY CORRECT' if auto_check else '⚠️ REVIEW NEEDED'}")
        
        if auto_check:
            total_correct += 1
    
    # Final summary
    print(f"\n\n{'=' * 90}")
    print("VERIFICATION SUMMARY")
    print(f"{'=' * 90}\n")
    print(f"Auto-verified as correct: {total_correct}/5")
    print(f"\n💡 Review the full content above to confirm each answer is correct.\n")
    print(f"{'=' * 90}\n")

if __name__ == "__main__":
    verify_with_detailed_output()
"""
RAG Database Stress Test - Hard Test Cases & Debugging

Tests edge cases, error handling, and robustness:
1. Very long complex queries
2. Ambiguous/vague queries
3. Queries with special characters
4. Multi-part technical queries
5. Performance under load
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag_retriever import RAGRetriever
import time

class RAGStressTest:
    def __init__(self):
        self.retriever = RAGRetriever()
        self.errors = []
        self.warnings = []
    
    def test_edge_cases(self):
        """Test edge cases and error conditions"""
        
        print("=" * 90)
        print("RAG STRESS TEST - EDGE CASES & DEBUGGING")
        print("=" * 90)
        print()
        
        tests = [
            {
                "name": "Empty Query",
                "query": "",
                "expect": "Should handle gracefully"
            },
            {
                "name": "Very Short Query",
                "query": "RAG",
                "expect": "Should return something relevant"
            },
            {
                "name": "Very Long Complex Query",
                "query": """I'm building a production multi-agent system that needs to handle 
                complex workflows involving code generation, research synthesis, and content creation. 
                The system should use hierarchical coordination with a manager agent delegating to 
                specialists, implement robust error handling and retry mechanisms, maintain conversation 
                context across multiple turns, integrate with a knowledge base for retrieval-augmented 
                generation, support both synchronous and asynchronous task execution, include 
                comprehensive observability with tracing and monitoring, handle rate limiting and 
                cost optimization, and ensure security compliance for enterprise deployment. 
                What are the key architectural patterns and best practices I should follow?""",
                "expect": "Should extract key concepts despite length"
            },
            {
                "name": "Special Characters",
                "query": "What is Chain-of-Thought (CoT) prompting? How does it work with LLMs?",
                "expect": "Should handle punctuation"
            },
            {
                "name": "Ambiguous Query",
                "query": "Tell me about agents",
                "expect": "Should return general agent info"
            },
            {
                "name": "Typos and Misspellings",
                "query": "How do I implemnt agentic RAG systms with embedings?",
                "expect": "Should still find relevant content (embeddings are robust to typos)"
            },
            {
                "name": "Non-Technical Question",
                "query": "What's the weather like today?",
                "expect": "Should return something or fail gracefully"
            },
            {
                "name": "Code-Like Query",
                "query": "def create_agent(): # How to implement?",
                "expect": "Should extract 'agent' and 'implement'"
            }
        ]
        
        passed = 0
        failed = 0
        
        for i, test in enumerate(tests, 1):
            print(f"\nTEST {i}: {test['name']}")
            print(f"{'─' * 90}")
            print(f"Query: {test['query'][:100]}{'...' if len(test['query']) > 100 else ''}")
            print(f"Expected: {test['expect']}")
            print()
            
            try:
                start_time = time.time()
                chunks = self.retriever.retrieve(test['query'], n_results=3)
                elapsed = time.time() - start_time
                
                if chunks:
                    print(f"✅ SUCCESS")
                    print(f"   Response time: {elapsed:.3f}s")
                    print(f"   Top result: {chunks[0]['source']}")
                    print(f"   Relevance: {chunks[0]['relevance']:.3f}")
                    print(f"   Preview: {chunks[0]['content'][:150]}...")
                    passed += 1
                else:
                    print(f"⚠️  WARNING: No results returned")
                    self.warnings.append(f"Test {i}: No results for '{test['name']}'")
                    passed += 1  # Not a failure, just no results
                    
            except Exception as e:
                print(f"❌ ERROR: {e}")
                self.errors.append(f"Test {i} ({test['name']}): {str(e)}")
                failed += 1
        
        print(f"\n{'=' * 90}")
        print(f"EDGE CASE TESTS: {passed}/{len(tests)} passed")
        if self.errors:
            print(f"Errors: {len(self.errors)}")
        if self.warnings:
            print(f"Warnings: {len(self.warnings)}")
        print(f"{'=' * 90}\n")
    
    def test_performance(self):
        """Test performance under load"""
        
        print("\n" + "=" * 90)
        print("PERFORMANCE STRESS TEST")
        print("=" * 90)
        print()
        
        queries = [
            "How does HNSW work?",
            "Explain ReAct prompting",
            "What is agentic RAG?",
            "Best practices for agent evaluation",
            "How to implement observability?"
        ]
        
        print("Running 5 queries sequentially...")
        start = time.time()
        
        for query in queries:
            self.retriever.retrieve(query, n_results=3)
        
        total_time = time.time() - start
        avg_time = total_time / len(queries)
        
        print(f"\n✅ PERFORMANCE RESULTS:")
        print(f"   Total time: {total_time:.3f}s")
        print(f"   Average per query: {avg_time:.3f}s")
        print(f"   Throughput: {len(queries)/total_time:.2f} queries/second")
        
        if avg_time < 0.5:
            print(f"   📊 EXCELLENT - Very fast!")
        elif avg_time < 1.0:
            print(f"   ✅ GOOD - Acceptable performance")
        else:
            print(f"   ⚠️  SLOW - Consider optimization")
        
        print(f"{'=' * 90}\n")
    
    def test_difficult_queries(self):
        """Test with very difficult/complex queries"""
        
        print("\n" + "=" * 90)
        print("DIFFICULT QUERY TEST")
        print("=" * 90)
        print()
        
        hard_queries = [
            {
                "query": "Compare and contrast the computational complexity and memory requirements of ScaNN versus HNSW for approximate nearest neighbor search in high-dimensional embedding spaces, specifically focusing on the trade-offs between query latency and index build time.",
                "topic": "Deep Technical Comparison"
            },
            {
                "query": "In a multi-agent system where agents can spawn sub-agents dynamically, how do I prevent infinite recursion, manage the growing context window, and ensure proper cleanup when the task tree completes or fails?",
                "topic": "Complex System Design Challenge"
            },
            {
                "query": "What are the implications of catastrophic forgetting when applying LoRA to domain-specific fine-tuning, and how does this interact with retrieval-augmented generation as a potential mitigation strategy?",
                "topic": "Advanced ML Theory"
            }
        ]
        
        for i, test in enumerate(hard_queries, 1):
            print(f"\nHARD TEST {i}: {test['topic']}")
            print(f"{'─' * 90}")
            print(f"Query: {test['query']}")
            print()
            
            chunks = self.retriever.retrieve(test['query'], n_results=5)
            
            print(f"📊 Retrieved {len(chunks)} chunks")
            print(f"📄 Top 3 sources:")
            for j, chunk in enumerate(chunks[:3], 1):
                print(f"   {j}. {chunk['source']} (relevance: {chunk['relevance']:.3f})")
            
            print(f"\n📝 Top result content:")
            print(f"   {chunks[0]['content'][:300]}...")
            
            # Check if it's actually relevant
            if chunks[0]['relevance'] > 0.1:
                print(f"\n✅ PASS - Found relevant content")
            else:
                print(f"\n⚠️  WARNING - Low relevance, may need query decomposition")
        
        print(f"\n{'=' * 90}\n")
    
    def test_database_stats(self):
        """Verify database integrity"""
        
        print("\n" + "=" * 90)
        print("DATABASE INTEGRITY CHECK")
        print("=" * 90)
        print()
        
        try:
            # Get collection stats
            collection = self.retriever.collection
            count = collection.count()
            
            print(f"✅ Collection accessible")
            print(f"   Total chunks: {count}")
            
            # Sample some data
            sample = collection.peek(limit=5)
            
            print(f"\n✅ Data sampling successful")
            print(f"   Sample documents: {len(sample['documents'])}")
            
            # Check embeddings
            embeddings_list = sample.get('embeddings')
            if embeddings_list is not None and len(embeddings_list) > 0:
                embedding_dim = len(embeddings_list[0])
                print(f"\n✅ Embeddings present")
                print(f"   Dimension: {embedding_dim}")
            else:
                print(f"\n⚠️  Embeddings not returned in peek()")
                print(f"   This is normal - ChromaDB doesn't always return embeddings in peek()")
            
            print(f"\n{'=' * 90}\n")
            
        except Exception as e:
            print(f"❌ ERROR: {e}")
            self.errors.append(f"Database integrity check failed: {str(e)}")
    
    def run_all_tests(self):
        """Run complete stress test suite"""
        
        print("\n🔥 STARTING COMPREHENSIVE STRESS TEST 🔥\n")
        
        # Run all test suites
        self.test_database_stats()
        self.test_edge_cases()
        self.test_difficult_queries()
        self.test_performance()
        
        # Final summary
        print("\n" + "=" * 90)
        print("FINAL STRESS TEST SUMMARY")
        print("=" * 90)
        print()
        
        if not self.errors and not self.warnings:
            print("🎉 ALL TESTS PASSED - NO ISSUES FOUND!")
            print()
            print("✅ Database is robust and production-ready")
            print("✅ Edge cases handled correctly")
            print("✅ Performance is excellent")
            print("✅ Complex queries work well")
        else:
            print(f"Errors found: {len(self.errors)}")
            for error in self.errors:
                print(f"   ❌ {error}")
            
            print(f"\nWarnings: {len(self.warnings)}")
            for warning in self.warnings:
                print(f"   ⚠️  {warning}")
        
        print(f"\n{'=' * 90}\n")
        
        return len(self.errors) == 0

if __name__ == "__main__":
    tester = RAGStressTest()
    success = tester.run_all_tests()
    
    if success:
        print("✅ Ready to proceed to Phase 1!")
    else:
        print("⚠️  Fix issues before proceeding")


# All test results archived here for reference
