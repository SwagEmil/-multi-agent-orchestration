"""
Test ADK agents + observability
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from adk_agents import ResearchAgent
from database.db_manager import DatabaseManager

def test_research_agent():
    print("=" * 70)
    print("Testing ADK Research Agent with Observability")
    print("=" * 70)
    
    agent = ResearchAgent()
    
    # Get agent config
    config = agent.get_config()
    print(f"\n✅ Agent Config:")
    print(f"   Name: {config['name']}")
    print(f"   Model: {config['model']}")
    print(f"   Description: {config['description']}")
    
    # Test query
    print(f"\n🔍 Testing Query: 'What is RAG?'")
    result = agent.execute({"description": "What is RAG?"})
    
    print(f"\n📊 Result:")
    print(f"   Response: {result['response'][:150]}...")
    print(f"   Agent: {result['agent']}")
    
    print(f"\n✅ ADK Research Agent working!")
    
if __name__ == "__main__":
    test_research_agent()
