"""
Orchestrator Agent - Master coordinator with ReAct reasoning

Uses Gemini Pro for complex reasoning about task decomposition and routing.
"""

import os
import re
import logging
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Dict, List
import uuid
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_retriever import RAGRetriever
from database.db_manager import DatabaseManager

load_dotenv()
logger = logging.getLogger(__name__)

class OrchestratorAgent:
    def __init__(self, db: DatabaseManager):
        """
        Initialize Orchestrator with Gemini Pro for complex reasoning
        
        Args:
            db: Database manager instance
        """
        # Use Gemini 2.5 Pro for best reasoning capability
        from utils.llm_factory import get_llm, MODEL_PRO
        self.llm = get_llm(
            model_name=MODEL_PRO,
            temperature=0.1
        )
        
        self.rag = RAGRetriever()
        self.db = db
        
        logger.info("Orchestrator initialized with Gemini 2.0 Flash Thinking (reasoning model)")
    
    def parse_response(self, content: str) -> dict:
        """
        Parse LLM response in expected format
        
        Expected format:
        THOUGHT: <reasoning>
        AGENT: <agent_name>
        INSTRUCTIONS: <task details>
        """
        try:
            # Normalize content to handle potential markdown formatting
            content_clean = content.replace('**', '')
            
            # More robust regex patterns
            thought_match = re.search(r'(?:THOUGHT|REASONING):\s*(.+?)(?=(?:AGENT|INSTRUCTIONS):|$)', content_clean, re.DOTALL | re.IGNORECASE)
            agent_match = re.search(r'AGENT:\s*(.+?)(?=(?:INSTRUCTIONS):|$)', content_clean, re.DOTALL | re.IGNORECASE)
            instructions_match = re.search(r'INSTRUCTIONS:\s*(.+)', content_clean, re.DOTALL | re.IGNORECASE)
            
            result = {
                'reasoning': thought_match.group(1).strip() if thought_match else '',
                'agent': agent_match.group(1).strip().lower() if agent_match else 'unknown',
                'instructions': instructions_match.group(1).strip() if instructions_match else content
            }
            
            # Fallback: If agent is unknown, try to find it in the raw content
            if result['agent'] == 'unknown':
                valid_agents = ['research_agent', 'code_agent', 'content_agent', 'analysis_agent']
                for agent in valid_agents:
                    if agent in content.lower():
                        result['agent'] = agent
                        break
            
            logger.info(f"Parsed response: agent={result['agent']}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to parse response: {e}")
            return {
                'reasoning': '',
                'agent': 'unknown',
                'instructions': content
            }
    
    def analyze_and_route(self, user_request: str) -> Dict:
        """
        Analyze user request and determine routing
        
        Uses ReAct pattern:
        1. Retrieve relevant context from knowledge base
        2. Reason about the task
        3. Decide which agent should handle it
        
        Args:
            user_request: User's input/task request
            
        Returns:
            dict with {reasoning, agent, instructions, rag_context}
        """
        # Step 1: Get relevant context from RAG knowledge base
        rag_context = self.rag.retrieve_context_string(
            f"How to handle this type of task: {user_request}",
            n_results=3
        )
        
        logger.info(f"Retrieved RAG context ({len(rag_context)} chars)")
        
        # Step 2: Use ReAct reasoning to decide
        prompt = f"""You are an intelligent task router. Your PRIMARY job is to detect KNOWLEDGE QUESTIONS and route them to research_agent.

KNOWLEDGE BASE CONTEXT:
{rag_context}

USER REQUEST: {user_request}

═══════════════════════════════════════════════════════════════════
CRITICAL ROUTING RULES (FOLLOW EXACTLY):
═══════════════════════════════════════════════════════════════════

**IF the query is asking "What/How/Why/Explain/Define"** → ALWAYS `research_agent`

Available Agents:
1. **research_agent**: ANY knowledge/conceptual question
   - "What is X?"
   - "How does Y work?"  
   - "Explain Z"
   - "Define ABC"
   - "Tell me about..."
   
2. **code_agent**: Code with bugs OR explicit code generation request
   - Buggy code snippet provided
   - "Debug this code"
   - "Generate code for..."
   
3. **content_agent**: ONLY if explicitly asked to write docs/summary
   - "Write documentation for..."
   - "Create a summary of..."
   
4. **analysis_agent**: ONLY for data/metrics tasks
   - "Analyze this data..."
   - "Calculate metrics for..."

═══════════════════════════════════════════════════════════════════
EXAMPLES:
═══════════════════════════════════════════════════════════════════

❌ WRONG:
User: "What is Chain-of-Thought prompting?"
Agent: content_agent

✅ CORRECT:
User: "What is Chain-of-Thought prompting?"  
Agent: research_agent (This is a knowledge question!)

✅ CORRECT:
User: "def add(a,b): return a-b"
Agent: code_agent (This is buggy code)

✅ CORRECT:
User: "Write a summary of this project"
Agent: content_agent (Explicitly asked to write)

═══════════════════════════════════════════════════════════════════
FORMAT YOUR RESPONSE:
═══════════════════════════════════════════════════════════════════

**Thought**: [Is this a knowledge question? Code? Writing task?]
**Agent**: [agent_name]
**Instructions**: [Clear task for the agent]

NOW ROUTE THIS REQUEST:
"""
        
        response = self.llm.invoke(prompt)
        parsed = self.parse_response(response.content)
        parsed['rag_context'] = rag_context
        
        return parsed
    
    def create_conversation(self, user_request: str) -> str:
        """
        Create a new conversation and analyze request
        
        Returns:
            conversation_id
        """
        conversation_id = f"conv_{uuid.uuid4().hex[:12]}"
        
        logger.info(f"Creating conversation: {conversation_id}")
        
        # Store in database
        self.db.create_conversation(
            conversation_id=conversation_id,
            user_request=user_request
        )
        
        # Analyze and route
        routing = self.analyze_and_route(user_request)
        
        # Create task
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        self.db.create_task(
            task_id=task_id,
            conversation_id=conversation_id,
            description=routing['instructions'],
            assigned_agent=routing['agent'],
            priority='medium'
        )
        
        logger.info(f"Created task {task_id} for {routing['agent']}")
        
        return conversation_id, task_id, routing
    
    def monitor_bugs(self, conversation_id: str):
        """
        Check for new bugs and trigger Research Agent
        
        This is the bug-triggered research workflow
        """
        pending_bugs = self.db.get_pending_bugs(conversation_id)
        
        if not pending_bugs:
            logger.info("No pending bugs to research")
            return
        
        logger.info(f"Found {len(pending_bugs)} pending bugs, triggering research")
        
        for bug in pending_bugs:
            # Create research task for this bug
            research_query = f"""Research solutions for this bug:
            
Description: {bug['description']}
Severity: {bug['severity']}
Code Context: {bug['code_context']}

Find known solutions, documentation, or similar issues."""
            
            # This would trigger Research Agent (to be implemented in Phase 2)
            logger.info(f"Would dispatch to Research Agent for bug {bug['id']}")
            
            # For now, just mark as noted
            # TODO: Actually execute Research Agent in Phase 2
