"""
Test Specialist Agents
Tests each specialist agent individually to ensure they can:
1. Retrieve RAG context
2. Process a request
3. Return valid JSON output
"""

import sys
import os
import unittest
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from database.db_manager import DatabaseManager
from agents.code_agent import CodeAgent
from agents.research_agent import ResearchAgent
from agents.content_agent import ContentAgent
from agents.analysis_agent import AnalysisAgent

class TestSpecialistAgents(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n🚀 Setting up Agent Tests...")
        cls.db = DatabaseManager()
        
    def test_code_agent(self):
        print("\n🧪 Testing Code Agent...")
        agent = CodeAgent(self.db)
        task = {
            "description": "Write a Python function to validate email addresses using regex. Also check for common security pitfalls.",
            "transcript_id": "test_transcript_001"
        }
        result = agent.execute(task)
        
        print(f"   Result keys: {result.keys()}")
        self.assertIn("implementation", result)
        self.assertIn("bugs_found", result)
        print("   ✅ Code Agent passed")

    def test_research_agent(self):
        print("\n🧪 Testing Research Agent...")
        agent = ResearchAgent()
        task = {
            "description": "What are the latest best practices for prompt engineering with Gemini models?",
            "context": {"source": "user_request"}
        }
        result = agent.execute(task)
        
        print(f"   Result keys: {result.keys()}")
        self.assertIn("findings", result)
        print("   ✅ Research Agent passed")

    def test_content_agent(self):
        print("\n🧪 Testing Content Agent...")
        agent = ContentAgent()
        task = {
            "description": "Write a short summary of how RAG works for a non-technical audience."
        }
        result = agent.execute(task)
        
        print(f"   Result keys: {result.keys()}")
        self.assertIn("content", result)
        print("   ✅ Content Agent passed")

    def test_analysis_agent(self):
        print("\n🧪 Testing Analysis Agent...")
        agent = AnalysisAgent()
        task = {
            "description": "Analyze this metric: User retention dropped by 15% after the new UI update.",
            "context": {"metric": "retention", "change": "-15%"}
        }
        result = agent.execute(task)
        
        print(f"   Result keys: {result.keys()}")
        self.assertIn("analysis", result)
        print("   ✅ Analysis Agent passed")

if __name__ == "__main__":
    unittest.main()
