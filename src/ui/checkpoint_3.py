"""
Checkpoint 3: Final Plan Review
Aggregate outputs -> Display final plan -> Export.
"""

import streamlit as st
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.db_manager import DatabaseManager

def render():
    st.header("Step 3: Final Plan Review")
    
    if 'execution_results' not in st.session_state:
        st.error("No execution results found. Please complete Step 2.")
        return
        
    results = st.session_state.execution_results
    
    st.markdown("### 📝 Final Consolidated Plan")
    
    # Generate Markdown Plan
    plan_content = "# Multi-Agent Execution Plan\n\n"
    plan_content += f"**Date:** {st.session_state.get('conversation_id', 'Unknown')}\n\n"
    
    for res in results:
        agent_name = res['agent'].replace('_', ' ').title()
        plan_content += f"## {agent_name} Report\n"
        plan_content += f"**Task:** {res['task']}\n\n"
        
        output = res['output']
        if isinstance(output, dict):
            if 'implementation' in output:
                plan_content += "### Implementation\n```python\n" + output['implementation'] + "\n```\n"
            if 'findings' in output:
                plan_content += "### Findings\n" + str(output['findings']) + "\n"
            if 'analysis' in output:
                plan_content += "### Analysis\n" + str(output['analysis']) + "\n"
        else:
            plan_content += str(output) + "\n"
            
        plan_content += "\n---\n\n"
        
    # Display Plan
    st.markdown(plan_content)
    
    st.markdown("---")
    
    # Export Options
    st.download_button(
        label="💾 Download Plan as Markdown",
        data=plan_content,
        file_name="final_plan.md",
        mime="text/markdown",
        type="primary"
    )
    
    if st.button("🏁 Start New Workflow", type="secondary"):
        st.session_state.current_step = 1
        st.session_state.conversation_id = None
        if 'analysis_result' in st.session_state:
            del st.session_state.analysis_result
        if 'execution_results' in st.session_state:
            del st.session_state.execution_results
        st.rerun()
