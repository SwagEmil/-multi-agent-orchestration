
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from agents.research_agent import ResearchAgent
from agents.content_agent import ContentAgent
from agents.orchestrator import OrchestratorAgent
from database.db_manager import DatabaseManager

def test_research_agent():
    print("\n--- Testing Research Agent ---")
    agent = ResearchAgent()
    task = {"description": "What is chain of thought prompting?"}
    result = agent.execute(task)
    print(f"Result keys: {result.keys()}")
    print(f"Findings type: {type(result.get('findings'))}")
    print(f"Findings content:\n{result.get('findings')}")

def test_analysis_agent():
    print("\n--- Testing Analysis Agent ---")
    from agents.analysis_agent import AnalysisAgent
    agent = AnalysisAgent()
    task = {"description": "Analyze the benefits of Chain of Thought prompting", "context": {"topic": "LLM reasoning"}}
    result = agent.execute(task)
    print(f"Result keys: {result.keys()}")
    print(f"Analysis format: {result.get('format')}")
    print(f"Analysis content:\n{result.get('analysis')[:200]}...")

def test_orchestrator():
    print("\n--- Testing Orchestrator Routing ---")
    db = DatabaseManager("data/orchestration.db")
    orchestrator = OrchestratorAgent(db)
    prompt = "What is chain of thought prompting?"
    conv_id, task_id, routing = orchestrator.create_conversation(prompt)
    print(f"Routing result: {routing}")

if __name__ == "__main__":
    # test_orchestrator()
    # test_research_agent()
    test_analysis_agent()
