"""
ADK Research Agent - Wraps existing ResearchAgent in Google ADK
"""
from google.genai import types
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.research_agent import ResearchAgent as LegacyResearchAgent
from rag_retriever import RAGRetriever

class ResearchAgent:
    """
    ADK-compliant Research Agent
    Wraps existing ResearchAgent for Kaggle submission compliance
    """
    
    def __init__(self):
        self.name = "research_agent"
        self.model = "gemini-2.5-pro"
        self.description = "Expert research specialist with advanced RAG capabilities"
        
        # Use existing implementation
        self.legacy_agent = LegacyResearchAgent()
        self.rag = RAGRetriever()
    
    def get_config(self):
        """Returns ADK-compatible agent configuration"""
        return {
            "name": self.name,
            "model": self.model,
            "description": self.description,
            "instructions": """You are an expert research specialist with access to a comprehensive knowledge base about AI agents, RAG, embeddings, and multi-agent systems.
            
Your capabilities:
- Advanced RAG retrieval with query expansion and re-ranking
- ELI5 explanations with analogies
- Comprehensive, well-structured answers
- Adaptive response length based on question complexity
            """,
            "tools": ["rag_retrieval"]
        }
    
    async def run(self, query: str) -> dict:
        """
        Execute research query using ADK pattern
        
        Args:
            query: User's research question
            
        Returns:
            dict with findings and sources
        """
        # Delegate to existing implementation
        task = {"description": f"Answer this question using the knowledge base: {query}"}
        result = self.legacy_agent.execute(task)
        
        return {
            "response": result.get("findings", "No results found."),
            "sources": result.get("sources_used", ""),
            "agent": self.name
        }
    
    def execute(self, task: dict) -> dict:
        """Synchronous execution (backward compatibility)"""
        import asyncio
        query = task.get("description", "")
        return asyncio.run(self.run(query))
