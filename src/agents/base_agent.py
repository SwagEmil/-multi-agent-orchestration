"""
Base Agent Class
Provides common functionality for all specialist agents, including:
- RAG Context Retrieval
- LLM Initialization
- Response Parsing
"""

import os
import logging
from typing import Dict, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from abc import ABC, abstractmethod

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_retriever import RAGRetriever

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    def __init__(self, agent_name: str, role: str, model_name: str = None, temperature: float = 0.2):
        """
        Initialize the base agent with RAG capabilities
        
        Args:
            agent_name: Name of the agent (e.g., "Code Agent")
            role: Short description of the agent's role
            model_name: Override model (defaults to MODEL_FAST)
            temperature: Temperature for generation (default 0.2)
        """
        self.agent_name = agent_name
        self.role = role
        
        # Initialize LLM via Factory
        from utils.llm_factory import get_llm, MODEL_FAST
        self.llm = get_llm(
            model_name=model_name or MODEL_FAST, 
            temperature=temperature
        )
        
        # Initialize RAG Retriever
        self.rag = RAGRetriever()
        
        logger.info(f"🤖 {agent_name} initialized")

    def retrieve_context(self, query: str, n_results: int = 3) -> str:
        """
        Retrieve relevant context from the knowledge base
        """
        context = self.rag.retrieve_context_string(
            f"{self.role}: {query}",
            n_results=n_results
        )
        return context

    @abstractmethod
    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the assigned task. Must be implemented by subclasses.
        
        Args:
            task: Dictionary containing task details (description, context, etc.)
            
        Returns:
            Dictionary with execution results
        """
        pass
