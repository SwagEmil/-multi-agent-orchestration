
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from agents.code_agent import CodeAgent
from database.db_manager import DatabaseManager

def test_syntax_error():
    print("\n--- Testing Code Agent with Syntax Error ---")
    db = DatabaseManager("data/orchestration.db")
    agent = CodeAgent(db)
    
    # Test case from user
    task = {"description": "def greet(name) print(f'Hello, {name}')"}
    
    print(f"Input: {task['description']}")
    result = agent.execute(task)
    
    print(f"\nResult keys: {result.keys()}")
    print(f"\nThought process: {result.get('thought_process')}")
    print(f"\nImplementation: {result.get('implementation')}")
    print(f"\nBugs found: {result.get('bugs_found')}")

if __name__ == "__main__":
    test_syntax_error()
