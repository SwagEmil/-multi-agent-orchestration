
import sys
import os
from pathlib import Path
import logging
from typing import List, Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from agents.orchestrator import OrchestratorAgent
from database.db_manager import DatabaseManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class KaggleTestRunner:
    def __init__(self):
        self.db = DatabaseManager("database/agent_system.db")
        self.orchestrator = OrchestratorAgent(self.db)
        self.results = []

    def run_test(self, test_case: Dict[str, Any]):
        query = test_case["query"]
        expected_agent = test_case["expected_agent"]
        difficulty = test_case["difficulty"]
        
        print(f"\n🔹 [{difficulty.upper()}] Testing: '{query}'")
        
        try:
            # We only test routing logic here to save time/tokens
            # For full execution, we would call create_conversation and then execute the agent
            conv_id, task_id, routing = self.orchestrator.create_conversation(query)
            
            actual_agent = routing['agent']
            reasoning = routing['reasoning']
            
            # Check if actual agent matches expected (or is in expected list)
            if isinstance(expected_agent, list):
                passed = actual_agent in expected_agent
            else:
                passed = actual_agent == expected_agent
            
            status = "✅ PASS" if passed else f"❌ FAIL (Got: {actual_agent}, Expected: {expected_agent})"
            print(f"   Result: {status}")
            print(f"   Reasoning: {reasoning}")
            
            self.results.append({
                "query": query,
                "difficulty": difficulty,
                "passed": passed,
                "actual": actual_agent,
                "expected": expected_agent
            })
            
        except Exception as e:
            print(f"   ❌ ERROR: {str(e)}")
            self.results.append({
                "query": query,
                "difficulty": difficulty,
                "passed": False,
                "error": str(e)
            })

    def print_summary(self):
        print("\n" + "="*50)
        print("🏆 KAGGLE TEST SUITE RESULTS")
        print("="*50)
        
        total = len(self.results)
        passed = sum(1 for r in self.results if r.get("passed"))
        
        print(f"\nTotal Tests: {total}")
        print(f"Passed:      {passed}")
        print(f"Failed:      {total - passed}")
        print(f"Success Rate: {passed/total*100:.1f}%")
        
        if passed < total:
            print("\n❌ FAILED TESTS:")
            for r in self.results:
                if not r.get("passed"):
                    print(f"- [{r['difficulty']}] '{r['query']}' -> Got {r.get('actual', 'Error')}, Expected {r.get('expected')}")

def main():
    runner = KaggleTestRunner()
    
    test_cases = [
        # --- EASY TESTS ---
        {
            "query": "What is Retrieval-Augmented Generation (RAG)?",
            "expected_agent": "research_agent",
            "difficulty": "easy"
        },
        {
            "query": "Write a Python function to calculate the factorial of a number.",
            "expected_agent": "code_agent",
            "difficulty": "easy"
        },
        {
            "query": "def greet(name) print(f'Hello, {name}')",
            "expected_agent": "code_agent",
            "difficulty": "easy"
        },
        
        # --- MEDIUM TESTS ---
        {
            "query": "Explain binary search and show me a Python implementation.",
            "expected_agent": ["code_agent", "research_agent"], # Orchestrator might pick either depending on weight
            "difficulty": "medium"
        },
        {
            "query": "Analyze the benefits of Chain of Thought prompting.",
            "expected_agent": ["analysis_agent", "research_agent"],
            "difficulty": "medium"
        },
        
        # --- HARD TESTS ---
        {
            "query": "Create a Streamlit app with a sidebar, a file uploader, and a button to process data.",
            "expected_agent": "code_agent",
            "difficulty": "hard"
        },
        {
            "query": "Design a microservices architecture for a scalable chat application.",
            "expected_agent": ["research_agent", "code_agent"],
            "difficulty": "hard"
        }
    ]
    
    print("🚀 Starting Kaggle Test Suite...")
    for test in test_cases:
        runner.run_test(test)
        
    runner.print_summary()

if __name__ == "__main__":
    main()
