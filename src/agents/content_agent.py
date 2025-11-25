"""
Content Agent
Specialized in writing documentation, summaries, and reports.
"""

import json
import logging
from typing import Dict, Any
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

class ContentAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            agent_name="Content Agent",
            role="Technical Writer & Communicator",
            temperature=0.4  # Higher temp for natural, creative writing
        )

    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a content generation task
        """
        task_desc = task.get('description', '')
        
        logger.info(f"Content Agent processing: {task_desc[:50]}...")
        
        # 1. Retrieve RAG Context
        rag_context = self.retrieve_context(task_desc)
        
        # 2. Construct Prompt
        prompt = f"""You are an expert Content Agent / Technical Writer.

KNOWLEDGE BASE CONTEXT:
{rag_context}

TASK:
{task_desc}

YOUR RESPONSIBILITIES:
1. Write clear, concise, and accurate content.
2. Adapt tone to the target audience (usually technical).
3. Structure the output with proper Markdown formatting.
4. Use the RAG context to ensure factual accuracy.

OUTPUT:
Provide the content directly in Markdown format.
"""
        
        # 3. Execute LLM
        response = self.llm.invoke(prompt)
        content = response.content
        
        # 4. Return result as a simple dict
        return {"content": content, "format": "markdown"}
