"""
Comprehensive Integration Test Suite
Tests critical system behaviors not covered by unit tests:
- Multi-agent parallel execution
- Database transaction safety
- RAG retrieval accuracy
- Error recovery mechanisms
"""

import sys
import os
import unittest
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from database.db_manager import DatabaseManager
from agents.orchestrator import OrchestratorAgent
from agents.code_agent import CodeAgent
from agents.research_agent import ResearchAgent
from rag_retriever import RAGRetriever


class TestSystemIntegration(unittest.TestCase):
    """Tests that verify system-level behaviors and edge cases"""
    
    @classmethod
    def setUpClass(cls):
        print("\n🧪 Integration Test Suite")
        cls.db_path = "database/test_integration.db"
        cls.db = DatabaseManager(cls.db_path)
        cls.orchestrator = OrchestratorAgent(cls.db)
        cls.rag = RAGRetriever()

    def test_concurrent_database_writes(self):
        """Verify database handles concurrent writes safely (WAL mode)"""
        print("\n⚡ Testing Concurrent DB Writes...")
        
        conv_id = "test_concurrent"
        self.db.create_conversation(conv_id, "Test concurrent writes")
        
        errors = []
        success_count = [0]
        
        def write_bugs(thread_id):
            try:
                # Each thread needs its own connection in SQLite
                thread_db = DatabaseManager(self.db_path)
                for i in range(5):
                    thread_db.record_bug(
                        conversation_id=conv_id,
                        description=f"Bug from thread {thread_id}, iteration {i}",
                        severity="low"
                    )
                success_count[0] += 5
            except Exception as e:
                errors.append(f"Thread {thread_id}: {str(e)}")
        
        # Spawn 3 threads writing simultaneously
        threads = [threading.Thread(target=write_bugs, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Verify no errors
        self.assertEqual(len(errors), 0, f"Concurrent write errors: {errors}")
        
        # Verify all 15 bugs were recorded
        bugs = self.db.conn.execute(
            "SELECT COUNT(*) as count FROM bugs WHERE conversation_id=?", 
            (conv_id,)
        ).fetchone()['count']
        self.assertEqual(bugs, 15, f"Expected 15 bugs, got {bugs}")
        print(f"   ✅ All {bugs} concurrent writes successful")

    def test_rag_retrieval_quality(self):
        """Verify RAG returns relevant context for various query types"""
        print("\n📚 Testing RAG Retrieval Quality...")
        
        test_queries = [
            ("What is ReAct reasoning?", ["react", "reasoning", "thought"]),
            ("How do agents handle errors?", ["error", "handling", "exception"]),
            ("Explain multi-agent systems", ["multi", "agent", "coordination"])
        ]
        
        for query, expected_keywords in test_queries:
            context = self.rag.retrieve_context_string(query, n_results=3)
            self.assertTrue(len(context) > 100, f"Context too short for: {query}")
            
            # Check at least one expected keyword appears
            found = any(kw.lower() in context.lower() for kw in expected_keywords)
            self.assertTrue(found, f"No expected keywords in context for: {query}")
        
        print("   ✅ RAG retrieval quality verified")

    def test_agent_error_recovery(self):
        """Verify agents handle malformed inputs gracefully"""
        print("\n🛡️ Testing Agent Error Recovery...")
        
        code_agent = CodeAgent(self.db)
        
        # Test with empty task
        result = code_agent.execute({"description": ""})
        self.assertIsInstance(result, dict)
        self.assertIn("implementation", result)
        
        # Test with extremely long task
        long_task = "A" * 10000
        result = code_agent.execute({"description": long_task})
        self.assertIsInstance(result, dict)
        
        print("   ✅ Agents handle edge cases gracefully")

    def test_orchestrator_routing_consistency(self):
        """Verify Orchestrator routes same request consistently"""
        print("\n🎯 Testing Orchestrator Routing Consistency...")
        
        request = "Fix the authentication bug in the login module"
        
        # Run routing 3 times
        results = []
        for i in range(3):
            conv_id, task_id, routing = self.orchestrator.create_conversation(request)
            results.append(routing['agent'])
        
        # All should route to code_agent
        code_agent_count = sum(1 for r in results if "code" in r.lower())
        self.assertGreaterEqual(code_agent_count, 2, f"Routing inconsistent: {results}")
        
        print(f"   ✅ Routing consistency: {code_agent_count}/3 routed to code_agent")

    def test_database_schema_integrity(self):
        """Verify all expected tables and columns exist"""
        print("\n🗄️ Testing Database Schema...")
        
        expected_tables = ["conversations", "tasks", "bugs", "agent_interactions"]
        
        cursor = self.db.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row['name'] for row in cursor.fetchall()]
        
        for table in expected_tables:
            self.assertIn(table, tables, f"Missing table: {table}")
        
        # Verify bugs table has retry_count
        cursor.execute("PRAGMA table_info(bugs)")
        columns = [row['name'] for row in cursor.fetchall()]
        self.assertIn("retry_count", columns, "Missing retry_count column")
        
        print("   ✅ Database schema integrity verified")

    @classmethod
    def tearDownClass(cls):
        try:
            os.remove(cls.db_path)
            os.remove(cls.db_path + "-shm")
            os.remove(cls.db_path + "-wal")
        except OSError:
            pass


if __name__ == "__main__":
    unittest.main()
