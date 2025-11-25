"""
Chat Interface Component
Handles message rendering, user input, and agent dispatching.
"""

import streamlit as st
from agents.orchestrator import OrchestratorAgent
from agents.research_agent import ResearchAgent
from agents.code_agent import CodeAgent
from agents.content_agent import ContentAgent
from agents.analysis_agent import AnalysisAgent
from database.db_manager import DatabaseManager
import uuid
from .session_state import SessionState

class ChatInterface:
    def __init__(self, db: DatabaseManager):
        self.db = db
        self.orchestrator = OrchestratorAgent(db)
        self.research_agent = ResearchAgent()
        self.code_agent = CodeAgent(db)
        self.content_agent = ContentAgent()
        self.analysis_agent = AnalysisAgent()

    def render(self):
        """Render the main chat interface"""
        
        # Display Chat History
        messages = SessionState.get_messages()
        for msg in messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("mode"):
                    st.caption(f"via {msg['mode']}")

        # User Input
        if prompt := st.chat_input("How can I help you today?"):
            # Add User Message
            SessionState.add_message("user", prompt)
            with st.chat_message("user"):
                st.markdown(prompt)

            # Process with Agent
            mode = st.session_state.current_mode
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                
                with st.spinner(f"{mode} Active..."):
                    response = self._dispatch_agent(prompt, mode)
                
                response_placeholder.markdown(response)
                SessionState.add_message("assistant", response, mode)

    def _dispatch_agent(self, prompt: str, mode: str) -> str:
        """Route the prompt to the correct agent based on mode"""
        
        try:
            if mode == "Auto Mode":
                # Intelligent multi-agent orchestration
                conv_id, task_id, routing = self.orchestrator.create_conversation(prompt)
                
                agent_name = routing['agent']
                instructions = routing['instructions']
                
                # Don't show routing analysis to user - just execute
                
                # Check if this is a complex query requiring multiple agents
                is_complex = any(keyword in prompt.lower() for keyword in [
                    'and explain', 'and show', 'build and', 'create and', 
                    'implement and', 'both', 'also'
                ])
                
                if is_complex:
                    # Multi-agent workflow for complex queries
                    response = ""
                    
                    # Determine which agents to use based on query
                    agents_to_use = []
                    
                    # Check for knowledge/explanation needs
                    if any(word in prompt.lower() for word in ['what', 'how', 'why', 'explain']):
                        agents_to_use.append(('research_agent', 'Explain the concept'))
                    
                    # Check for code needs
                    if any(word in prompt.lower() for word in ['code', 'build', 'create', 'implement', 'function', 'def ']):
                        agents_to_use.append(('code_agent', 'Implement the solution'))
                    
                    # Execute each agent
                    for i, (agent_name, task_desc) in enumerate(agents_to_use):
                        if i > 0:
                            response += "\n\n---\n\n"
                        
                        task = {"description": f"{task_desc}: {prompt}"}
                        
                        if agent_name == "research_agent":
                            result = self.research_agent.execute(task)
                            response += result.get("findings", "No results.")
                        elif agent_name == "code_agent":
                            result = self.code_agent.execute(task)
                            if result.get("implementation"):
                                response += f"```python\n{result['implementation']}\n```"
                    
                else:
                    # Single-agent workflow - just show the answer
                    response = ""
                    
                    task = {"description": instructions}
                    
                    if agent_name == "research_agent":
                        result = self.research_agent.execute(task)
                        response += result.get("findings", "No results.")
                        
                    elif agent_name == "code_agent":
                        result = self.code_agent.execute(task)
                        
                        # Ensure result is a dict, not a string
                        if isinstance(result, str):
                            response += result
                        else:
                            # Format bugs if found
                            bugs = result.get("bugs_found", [])
                            if bugs:
                                response += "### 🐛 Bugs Found\n\n"
                                for bug in bugs:
                                    response += f"**{bug.get('description', 'Unknown bug')}** ({bug.get('severity', 'unknown')})\n"
                                    response += f"```python\n{bug.get('code_context', '')}\n```\n\n"
                            
                            # Format implementation if provided
                            if result.get("implementation"):
                                response += "### 🛠️ Solution\n"
                                impl = result['implementation']
                                # Clean up implementation string (remove extra backticks if present)
                                impl = impl.replace('```python', '').replace('```', '').strip()
                                response += f"```python\n{impl}\n```"
                            
                            # If no bugs and no implementation, show thought process
                            if not bugs and not result.get("implementation"):
                                response += result.get("thought_process", "No analysis generated.")
                            
                    elif agent_name == "content_agent":
                        result = self.content_agent.execute(task)
                        # Content agent returns dict with 'content' key
                        content = result.get("content", "")
                        # If content is still a dict (JSON), extract the text
                        if isinstance(content, dict):
                            response += content.get("content", str(content))
                        else:
                            response += content
                        
                    elif agent_name == "analysis_agent":
                        result = self.analysis_agent.execute(task)
                        response += result.get("analysis", result.get("insights", "No analysis generated."))
                        
                    else:
                        # Fallback to research agent
                        response += f"⚠️ Unknown agent `{agent_name}`. Using Research Agent.\n\n"
                        result = self.research_agent.execute({"description": prompt})
                        response += result.get("findings", "No results.")
                
                return response
            
            elif mode == "Learning Mode":
                # Use Research Agent for Q&A
                task = {"description": f"Answer this question using the knowledge base: {prompt}"}
                result = self.research_agent.execute(task)
                return result.get("findings", "I couldn't find an answer in the knowledge base.")

            elif mode == "Debugging Mode":
                # Use Code Agent for debugging
                task = {"description": f"Analyze this code/error and find bugs: {prompt}"}
                result = self.code_agent.execute(task)
                
                response = "### 🐛 Bug Analysis\n\n"
                if result.get("bugs_found"):
                    for bug in result["bugs_found"]:
                        response += f"**{bug['description']}** ({bug['severity']})\n"
                        response += f"```python\n{bug['code_context']}\n```\n\n"
                
                if result.get("implementation"):
                    response += "### 🛠️ Fix\n"
                    response += f"```python\n{result['implementation']}\n```"
                    
                return response if response != "### 🐛 Bug Analysis\n\n" else "No bugs detected, but here's the analysis:\n" + str(result)

            elif mode == "Coding Mode":
                # Use Orchestrator for complex tasks
                # Note: Orchestrator usually creates a plan. We might want to execute it too.
                # For MVP chat, let's just get the plan/response.
                
                # We need to adapt Orchestrator to return a direct response or run the flow.
                # For now, let's run a simplified flow: Orchestrator -> Plan
                
                conv_id, task_id, routing = self.orchestrator.create_conversation(prompt)
                
                response = f"### 📋 Plan Created\n\n**Reasoning:** {routing['reasoning']}\n\n"
                response = f"### 📋 Plan Created\n\n**Reasoning:** {routing['reasoning']}\n\n"
                response += "**Assigned Task:**\n"
                response += f"- **Agent:** {routing['agent']}\n"
                response += f"- **Instructions:** {routing['instructions']}\n"
                
                return response

            else:
                return "Unknown mode selected."

        except Exception as e:
            return f"❌ Error: {str(e)}"
