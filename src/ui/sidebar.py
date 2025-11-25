"""
Sidebar Component
Handles navigation, mode selection, and settings.
"""

import streamlit as st
from .session_state import SessionState

def render_sidebar():
    with st.sidebar:
        st.markdown("## 🤖 Agent Orchestrator")
        
        # Mode Selection
        st.markdown("### 🎯 Select Mode")
        
        modes = {
            "Auto Mode": "🤖",
            "Learning Mode": "📚",
            "Debugging Mode": "🐛",
            "Coding Mode": "💻"
        }
        
        selected_mode = st.radio(
            "Choose your workflow:",
            options=list(modes.keys()),
            format_func=lambda x: f"{modes[x]} {x}",
            label_visibility="collapsed",
            key="current_mode"  # Direct binding to session_state
        )
        
        # SessionState.set_mode is handled automatically by key binding
        
        st.markdown("---")
        
        # Mode Description
        if selected_mode == "Auto Mode":
            st.success("**Auto Mode** ✨\n\nOrchestrator intelligently routes your query to the best agent(s). Recommended for most use cases.")
        elif selected_mode == "Learning Mode":
            st.info("**Learning Mode**\n\nAsk questions about AI agents, RAG, and system architecture. Uses the Knowledge Base.")
        elif selected_mode == "Debugging Mode":
            st.warning("**Debugging Mode**\n\nPaste error logs or buggy code. The Code & Analysis agents will diagnose issues.")
        elif selected_mode == "Coding Mode":
            st.success("**Coding Mode**\n\nFull Orchestrator access. Request new features, refactoring, or complex tasks.")
            
        st.markdown("---")
        
        # Settings
        with st.expander("⚙️ Settings"):
            st.selectbox(
                "Model",
                ["gemini-2.0-flash-exp", "gemini-2.5-pro (Reasoning)"],
                index=0,
                key="model_selector"
            )
            st.checkbox("Show Reasoning Chains", value=True)
            
        # Actions
        if st.button("🗑️ Clear Chat", use_container_width=True):
            SessionState.clear_history()
            st.rerun()
            
        st.markdown("---")
        st.caption(f"Session ID: {st.session_state.conversation_id[:8]}...")
