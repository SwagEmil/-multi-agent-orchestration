"""
Checkpoint 1: Task Review
User inputs request -> Orchestrator analyzes -> User reviews task breakdown.
"""

import streamlit as st
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.db_manager import DatabaseManager
from agents.orchestrator import OrchestratorAgent

def render():
    st.header("Step 1: Task Review & Orchestration")
    
    # Initialize DB and Orchestrator
    db = DatabaseManager()
    orchestrator = OrchestratorAgent(db)
    
    # Input Section
    with st.container():
        user_request = st.text_area(
            "Enter your request:",
            height=100,
            placeholder="Example: Analyze the meeting transcript and fix the login bug mentioned."
        )
        
        if st.button("Analyze & Plan", type="primary"):
            if not user_request:
                st.warning("Please enter a request.")
                return
            
            with st.spinner("Orchestrator is thinking... (ReAct Pattern)"):
                # 1. Create Conversation
                conv_id = f"conv_{len(db.conn.execute('SELECT * FROM conversations').fetchall()) + 1}"
                db.create_conversation(conv_id, user_request)
                st.session_state.conversation_id = conv_id
                
                # 2. Orchestrator Analysis
                # Note: In a real app, we'd pass the transcript here. For now, using request as context.
                analysis = orchestrator.analyze_and_route(user_request)
                
                # Store in session state for review
                st.session_state.analysis_result = analysis
                st.session_state.user_request = user_request
                
    # Review Section
    if 'analysis_result' in st.session_state:
        result = st.session_state.analysis_result
        
        st.markdown("### 🧠 Orchestrator Reasoning")
        st.info(result.get('reasoning', 'No reasoning provided.'))
        
        st.markdown("### 📋 Proposed Tasks")
        
        # Display tasks in a nice format
        agent = result.get('agent', 'Unknown')
        instructions = result.get('instructions', 'No instructions')
        
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown(f"**Assigned Agent:**\n`{agent}`")
        with col2:
            st.markdown(f"**Instructions:**\n{instructions}")
            
        st.markdown("---")
        
        # Approval Buttons
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("✅ Approve Plan", type="primary", use_container_width=True):
                # Save tasks to DB (simulated for single task MVP)
                # In full version, we'd iterate over a list of tasks
                task_id = f"task_{st.session_state.conversation_id}_1"
                db.create_task(
                    task_id=task_id,
                    conversation_id=st.session_state.conversation_id,
                    description=instructions,
                    assigned_agent=agent,
                    priority="high"
                )
                
                st.success("Plan approved! Proceeding to execution...")
                st.session_state.current_step = 2
                st.rerun()
                
        with col_b:
            if st.button("❌ Reject / Edit", type="secondary", use_container_width=True):
                st.warning("Please edit your request above and try again.")
                del st.session_state.analysis_result
                st.rerun()
