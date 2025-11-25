"""
LangGraph Workflow Definition
Orchestrates the multi-agent system using a state graph.
"""

import sys
from pathlib import Path
from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import operator

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.db_manager import DatabaseManager
from agents.orchestrator import OrchestratorAgent
from agents.code_agent import CodeAgent
from agents.research_agent import ResearchAgent
from agents.content_agent import ContentAgent
from agents.analysis_agent import AnalysisAgent

# Initialize System Components
# In a real app, these might be injected or initialized in a startup block
db = DatabaseManager()
orchestrator = OrchestratorAgent(db)
code_agent = CodeAgent(db)
research_agent = ResearchAgent()
content_agent = ContentAgent()
analysis_agent = AnalysisAgent()

# Define the Agent State
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    next_agent: str
    task_info: Dict[str, Any]
    rag_context: str
    reasoning: str

# Node Functions

def orchestrator_node(state: AgentState):
    """
    The Orchestrator node analyses the request and routes it.
    """
    print("--- ORCHESTRATOR NODE ---")
    messages = state['messages']
    last_message = messages[-1].content if messages else ""
    
    # Use the real OrchestratorAgent to analyze and route
    routing_result = orchestrator.analyze_and_route(last_message)
    
    next_agent = routing_result['agent']
    instructions = routing_result['instructions']
    reasoning = routing_result['reasoning']
    
    print(f"Routing to: {next_agent}")
    print(f"Instructions: {instructions[:50]}...")
    
    return {
        "next_agent": next_agent,
        "reasoning": reasoning,
        "task_info": {"description": instructions},
        "messages": [AIMessage(content=f"Orchestrator: Routing to {next_agent}. Task: {instructions}")]
    }

def code_agent_node(state: AgentState):
    print("--- CODE AGENT NODE ---")
    task_info = state.get('task_info', {})
    
    # Execute Code Agent
    result = code_agent.execute(task_info)
    
    return {
        "messages": [AIMessage(content=f"Code Agent Result: {result.get('implementation', 'No code generated')[:100]}...")]
    }

def research_agent_node(state: AgentState):
    print("--- RESEARCH AGENT NODE ---")
    task_info = state.get('task_info', {})
    
    # Execute Research Agent
    result = research_agent.execute(task_info)
    
    return {
        "messages": [AIMessage(content=f"Research Agent Findings: {str(result.get('findings', 'No findings'))[:100]}...")]
    }

def content_agent_node(state: AgentState):
    print("--- CONTENT AGENT NODE ---")
    task_info = state.get('task_info', {})
    
    # Execute Content Agent
    result = content_agent.execute(task_info)
    
    return {
        "messages": [AIMessage(content=f"Content Agent Output: {str(result.get('summary', 'No content'))}")]
    }

def analysis_agent_node(state: AgentState):
    print("--- ANALYSIS AGENT NODE ---")
    task_info = state.get('task_info', {})
    
    # Execute Analysis Agent
    result = analysis_agent.execute(task_info)
    
    return {
        "messages": [AIMessage(content=f"Analysis Agent Insights: {str(result.get('key_insights', 'No insights'))}")]
    }

# Conditional Edge Logic
def router(state: AgentState):
    return state['next_agent']

# Build the Graph
def create_graph():
    workflow = StateGraph(AgentState)

    # Add Nodes
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("code_agent", code_agent_node)
    workflow.add_node("research_agent", research_agent_node)
    workflow.add_node("content_agent", content_agent_node)
    workflow.add_node("analysis_agent", analysis_agent_node)

    # Set Entry Point
    workflow.set_entry_point("orchestrator")

    # Add Edges
    workflow.add_conditional_edges(
        "orchestrator",
        router,
        {
            "code_agent": "code_agent",
            "research_agent": "research_agent",
            "content_agent": "content_agent",
            "analysis_agent": "analysis_agent",
            "unknown": END, # Handle unknown routing
            "end": END
        }
    )

    # All agents return to END for this MVP
    workflow.add_edge("code_agent", END)
    workflow.add_edge("research_agent", END)
    workflow.add_edge("content_agent", END)
    workflow.add_edge("analysis_agent", END)

    return workflow.compile()

if __name__ == "__main__":
    # Test the full workflow
    print("🚀 Initializing Multi-Agent Workflow...")
    app = create_graph()
    
    test_query = "Check the authentication module for security bugs and fix them."
    print(f"\n🧪 Testing Query: {test_query}\n")
    
    inputs = {"messages": [HumanMessage(content=test_query)]}
    
    try:
        for output in app.stream(inputs):
            for key, value in output.items():
                print(f"\nOutput from {key}:")
                # print(f"  {value}") # Verbose
    except Exception as e:
        print(f"❌ Workflow Error: {e}")

