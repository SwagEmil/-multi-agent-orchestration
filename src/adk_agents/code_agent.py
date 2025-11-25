"""
ADK Code Agent - Wraps existing CodeAgent in Google ADK
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.code_agent import CodeAgent as LegacyCodeAgent
from database.db_manager import DatabaseManager

class CodeAgent:
    """
    ADK-compliant Code Agent
    Analyzes code, finds bugs, suggests fixes
    """
    
    def __init__(self, db: DatabaseManager):
        self.name = "code_agent"
        self.model = "gemini-2.0-flash-exp"
        self.description = "Expert code analyst specializing in bug detection and fixes"
        
        self.legacy_agent = LegacyCodeAgent(db)
        self.db = db
    
    def get_config(self):
        """Returns ADK-compatible agent configuration"""
        return {
            "name": self.name,
            "model": self.model,
            "description": self.description,
            "instructions": """You are an expert code analyst with access to a knowledge base of best practices.
            
Your capabilities:
- Bug detection and analysis
- Code quality assessment
- Fix recommendations with examples
- Integration with bug tracking database
            """,
            "tools": ["code_analysis", "bug_database"]
        }
    
    async def run(self, code_or_description: str) -> dict:
        """
        Analyze code for bugs
        
        Args:
            code_or_description: Code snippet or description of problem
            
        Returns:
            dict with bugs found and fixes
        """
        task = {"description": f"Analyze this code/error and find bugs: {code_or_description}"}
        result = self.legacy_agent.execute(task)
        
        return {
            "bugs_found": result.get("bugs_found", []),
            "implementation": result.get("implementation", ""),
            "agent": self.name
        }
    
    def execute(self, task: dict) -> dict:
        """Synchronous execution (backward compatibility)"""
        import asyncio
        description = task.get("description", "")
        return asyncio.run(self.run(description))
