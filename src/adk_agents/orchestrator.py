"""
ADK Orchestrator - Wraps existing OrchestratorAgent in Google ADK
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.orchestrator import OrchestratorAgent as LegacyOrchestrator
from database.db_manager import DatabaseManager

class Orchestrator:
    """
    ADK-compliant Orchestrator Agent
    Routes tasks to specialist agents using ReAct reasoning
    """
    
    def __init__(self, db: DatabaseManager):
        self.name = "orchestrator"
        self.model = "gemini-2.5-pro"
        self.description = "Master coordinator using ReAct pattern for complex task decomposition"
        
        self.legacy_orchestrator = LegacyOrchestrator(db)
        self.db = db
    
    def get_config(self):
        """Returns ADK-compatible agent configuration"""
        return {
            "name": self.name,
            "model": self.model,
            "description": self.description,
            "instructions": """You are an orchestrator for a multi-agent system using ReAct reasoning pattern.
            
Your capabilities:
- Analyze complex requests
- Route tasks to appropriate specialists (code_agent, research_agent, content_agent, analysis_agent)
- Combine multi-agent outputs
- Track task execution in database
            
Available agents:
- code_agent: Bug fixing, code analysis
- research_agent: Documentation lookup, knowledge retrieval
- content_agent: Writing documentation, summaries
- analysis_agent: Data analysis, metrics
            """,
            "tools": ["agent_routing", "rag_retrieval", "task_database"]
        }
    
    async def run(self, user_request: str) -> dict:
        """
        Create conversation and route task
        
        Args:
            user_request: Complex task from user
            
        Returns:
            dict with conversation_id, task_id, and routing info
        """
        conv_id, task_id, routing = self.legacy_orchestrator.create_conversation(user_request)
        
        return {
            "conversation_id": conv_id,
            "task_id": task_id,
            "reasoning": routing.get("reasoning", ""),
            "assigned_agent": routing.get("agent", "unknown"),
            "instructions": routing.get("instructions", ""),
            "agent": self.name
        }
    
    def create_conversation(self, user_request: str):
        """Synchronous execution (backward compatibility)"""
        import asyncio
        result = asyncio.run(self.run(user_request))
        return (
            result["conversation_id"],
            result["task_id"],
            {
                "reasoning": result["reasoning"],
                "agent": result["assigned_agent"],
                "instructions": result["instructions"]
            }
        )
