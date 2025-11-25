"""
Checkpoint 2: Agent Output Review
Execute agents -> Display results -> User approves outputs.
"""

import streamlit as st
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.db_manager import DatabaseManager
from agents.code_agent import CodeAgent
from agents.research_agent import ResearchAgent
from agents.content_agent import ContentAgent
from agents.analysis_agent import AnalysisAgent

def render():
    st.header("Step 2: Agent Execution & Review")
    
    db = DatabaseManager()
    conv_id = st.session_state.get('conversation_id')
    
    if not conv_id:
        st.error("No conversation found. Please go back to Step 1.")
        return

    # Fetch pending tasks
    tasks = db.get_conversation_tasks(conv_id)
    
    if not tasks:
        st.warning("No tasks found for this conversation.")
        return

    # Execution Section
    if 'execution_results' not in st.session_state:
        st.markdown("### ⚙️ Executing Agents...")
        
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Initialize Agents
        agents = {
            "code_agent": CodeAgent(db),
            "research_agent": ResearchAgent(),
            "content_agent": ContentAgent(),
            "analysis_agent": AnalysisAgent()
        }
        
        for i, task in enumerate(tasks):
            agent_name = task['assigned_agent']
            status_text.text(f"Running {agent_name}...")
            
            agent = agents.get(agent_name)
            if agent:
                # Execute Agent
                try:
                    task_input = {"description": task['description'], "transcript_id": conv_id}
                    output = agent.execute(task_input)
                    
                    # Update DB
                    db.update_task_status(task['id'], 'completed', str(output))
                    
                    results.append({
                        "agent": agent_name,
                        "task": task['description'],
                        "output": output
                    })
                except Exception as e:
                    st.error(f"Error executing {agent_name}: {e}")
                    results.append({
                        "agent": agent_name,
                        "task": task['description'],
                        "output": {"error": str(e)}
                    })
            else:
                st.error(f"Unknown agent: {agent_name}")
            
            progress_bar.progress((i + 1) / len(tasks))
            time.sleep(0.5) # UX pause
            
        st.session_state.execution_results = results
        status_text.text("Execution Complete!")
        st.rerun()

    # Review Section
    else:
        st.markdown("### 📤 Agent Outputs")
        
        results = st.session_state.execution_results
        
        for res in results:
            with st.expander(f"{res['agent']} Output", expanded=True):
                st.markdown(f"**Task:** {res['task']}")
                st.markdown("---")
                
                output = res['output']
                
                # Format output based on agent type
                if res['agent'] == 'code_agent':
                    if 'implementation' in output:
                        st.code(output['implementation'], language='python')
                    if 'bugs_found' in output and output['bugs_found']:
                        st.error(f"🐞 Bugs Found: {len(output['bugs_found'])}")
                        st.json(output['bugs_found'])
                        
                elif res['agent'] == 'research_agent':
                    st.markdown(output.get('findings', str(output)))
                    
                else:
                    st.write(output)
                    
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Approve Outputs", type="primary", use_container_width=True):
                st.success("Outputs approved! Proceeding to final plan...")
                st.session_state.current_step = 3
                st.rerun()
                
        with col2:
            if st.button("🔄 Retry Execution", type="secondary", use_container_width=True):
                del st.session_state.execution_results
                st.rerun()
