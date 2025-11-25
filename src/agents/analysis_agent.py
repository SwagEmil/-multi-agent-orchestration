"""
Analysis Agent
Specialized in data analysis, metrics, and insights.
"""

import json
import logging
from typing import Dict, Any
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

class AnalysisAgent(BaseAgent):
    def __init__(self):
        from utils.llm_factory import MODEL_PRO
        super().__init__(
            agent_name="Analysis Agent",
            role="Data Analyst & Strategist",
            model_name=MODEL_PRO,  # Use Pro for precise, factual analysis
            temperature=0.1  # Lower temp for deterministic insights
        )

    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an analysis task
        """
        task_desc = task.get('description', '')
        data_context = task.get('context', {})
        
        logger.info(f"Analysis Agent processing: {task_desc[:50]}...")
        
        # 1. Retrieve RAG Context
        rag_context = self.retrieve_context(task_desc)
        
        # 2. Construct Prompt
        prompt = f"""You are an expert Analysis Agent.

KNOWLEDGE BASE CONTEXT:
{rag_context}

TASK:
{task_desc}

DATA CONTEXT:
{json.dumps(data_context, indent=2)}

YOUR RESPONSIBILITIES:
1. Analyze the provided data or request.
2. Identify trends, patterns, or key insights.
3. Provide data-driven recommendations.
4. Use the RAG context to benchmark against best practices.

OUTPUT:
Provide the analysis directly in Markdown format. Use headers, bullet points, and code blocks where appropriate.
"""
        
        # 3. Execute LLM
        response = self.llm.invoke(prompt)
        content = response.content
        
        # 4. Return result
        return {
            "analysis": content,
            "format": "markdown"
        }
