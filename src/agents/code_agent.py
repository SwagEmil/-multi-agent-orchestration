"""
Code Agent
Specialized in code analysis, bug detection, and technical implementation.
"""

import json
import logging
import re
from typing import Dict, Any
from .base_agent import BaseAgent
from database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)

class CodeAgent(BaseAgent):
    def __init__(self, db: DatabaseManager):
        super().__init__(
            agent_name="Code Agent",
            role="Software Engineer & Security Analyst"
        )
        self.db = db

    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a coding or analysis task
        
        Args:
            task: {
                'description': str,
                'transcript_id': str (optional),
                'context': dict (optional)
            }
        """
        task_desc = task.get('description', '')
        transcript_id = task.get('transcript_id')
        
        logger.info(f"Code Agent processing: {task_desc[:50]}...")
        
        # 1. Retrieve RAG Context
        rag_context = self.retrieve_context(task_desc)
        
        # 2. Construct Prompt
        prompt = f"""You are an expert Code Agent specialized in software engineering, security, and bug fixing.

KNOWLEDGE BASE CONTEXT:
{rag_context}

TASK:
{task_desc}

YOUR RESPONSIBILITIES:
1. Analyze the request and context.
2. If this is a bug fix, identify the root cause.
3. If this is a new feature, provide a clean implementation.
4. CRITICAL: Identify any potential bugs or security issues in the code/request.

OUTPUT FORMAT (JSON):
{{
  "thought_process": "Your analysis...",
  "implementation": "The code solution or analysis...",
  "bugs_found": [
    {{
      "description": "Brief description of bug",
      "severity": "high|medium|low",
      "code_context": "Relevant code snippet"
    }}
  ]
}}

Ensure valid JSON output. If no bugs found, return empty list for "bugs_found".
"""
        
        # 3. Execute LLM
        response = self.llm.invoke(prompt)
        content = response.content
        
        # 4. Parse Output
        try:
            # Clean markdown code blocks if present
            clean_content = content.replace('```json', '').replace('```', '').strip()
            result = json.loads(clean_content)
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON, returning raw content")
            result = {
                "thought_process": "Failed to parse JSON response",
                "implementation": content,
                "bugs_found": []
            }

        # 5. Record Bugs in Database
        bugs = result.get('bugs_found', [])
        if bugs and transcript_id:
            logger.info(f"Found {len(bugs)} bugs, recording to database...")
            for bug in bugs:
                self.db.record_bug(
                    conversation_id=transcript_id,  # Map transcript_id to conversation_id
                    description=bug.get('description', 'Unknown bug'),
                    severity=bug.get('severity', 'medium'),
                    code_context=bug.get('code_context', '')
                )
                
        return result
