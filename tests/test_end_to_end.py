"""
End-to-End Integration Test
Simulates a full user workflow:
1. User Request -> Orchestrator
2. Orchestrator -> Task Creation
3. Task -> Specialist Agent Execution
4. Verification of Final Output
"""

import sys
import os
import unittest
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from database.db_manager import DatabaseManager
from agents.orchestrator import OrchestratorAgent
from agents.code_agent import CodeAgent
from agents.research_agent import ResearchAgent

class TestEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n🚀 Starting End-to-End System Test...")
        cls.db_path = "database/test_e2e.db"
        cls.db = DatabaseManager(cls.db_path)
        cls.orchestrator = OrchestratorAgent(cls.db)
        
    def setUp(self):
        # Clean DB state if needed
        pass

    def test_full_flow_code_analysis(self):
        print("\n📝 Scenario: Code Analysis Request")
        
        # 1. User Request
        user_request = "Analyze this Python code for bugs: def add(a,b): return a-b"
        print(f"   User Request: {user_request}")
        
        # 2. Orchestrator Analysis
        print("   🤖 Orchestrator Analyzing...")
        conv_id, task_id, routing = self.orchestrator.create_conversation(user_request)
        
        print(f"   Conversation ID: {conv_id}")
        print(f"   Task ID: {task_id}")
        print(f"   Routing: {routing}")
        
        self.assertIsNotNone(conv_id)
        self.assertIsNotNone(task_id)
        self.assertIn("code", routing['agent'].lower()) # Should route to code agent
        
        # 3. Execute Assigned Task
        print(f"   🚀 Executing Task with {routing['agent']}...")
        
        # Fetch task from DB to get full details
        tasks = self.db.get_conversation_tasks(conv_id)
        task = tasks[0]
        
        # Initialize appropriate agent
        if "code" in task['assigned_agent'].lower():
            agent = CodeAgent(self.db)
        elif "research" in task['assigned_agent'].lower():
            agent = ResearchAgent()
        else:
            self.fail(f"Unexpected agent assigned: {task['assigned_agent']}")
            
        # Execute
        task_input = {
            "description": task['description'],
            "transcript_id": conv_id
        }
        result = agent.execute(task_input)
        
        print(f"   ✅ Agent Result: {result}")
        
        # 4. Verify Output
        self.assertIn("implementation", result)
        # It should find the bug (subtraction instead of addition)
        # Note: We can't guarantee exact wording, but we check for structure
        
        # Update Task in DB
        self.db.update_task_status(task['id'], 'completed', str(result))
        
        # Verify DB state
        updated_task = self.db.get_conversation_tasks(conv_id)[0]
        self.assertEqual(updated_task['status'], 'completed')
        print("   ✅ DB Updated Successfully")

    def test_full_flow_research(self):
        print("\n📝 Scenario: Research Request")
        
        # 1. User Request
        user_request = "Find the latest best practices for securing Python APIs in 2024."
        print(f"   User Request: {user_request}")
        
        # 2. Orchestrator Analysis
        print("   🤖 Orchestrator Analyzing...")
        conv_id, task_id, routing = self.orchestrator.create_conversation(user_request)
        
        print(f"   Routing: {routing}")
        self.assertIn("research", routing['agent'].lower())
        
        # 3. Execute Assigned Task
        print(f"   🚀 Executing Task with {routing['agent']}...")
        agent = ResearchAgent()
        
        # Fetch task details
        tasks = self.db.get_conversation_tasks(conv_id)
        task = tasks[0]
        
        task_input = {
            "description": task['description'],
            "transcript_id": conv_id
        }
        result = agent.execute(task_input)
        
        print(f"   ✅ Agent Result keys: {result.keys()}")
        self.assertIn("findings", result)
        self.assertTrue(len(result['findings']) > 0)

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
