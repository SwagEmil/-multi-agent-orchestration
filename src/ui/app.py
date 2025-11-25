"""
Main Streamlit Application
Premium Chat UI for AI Agent Orchestration System
"""

import streamlit as st
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from database.db_manager import DatabaseManager
from ui.session_state import SessionState
from ui.sidebar import render_sidebar
from ui.chat_interface import ChatInterface

# Page config
st.set_page_config(
    page_title="AI Agent Orchestrator",
    page_icon="🤖",
    layout="wide"
)

# Initialize database
db = DatabaseManager("data/orchestration.db")

# Initialize chat interface with database
chat = ChatInterface(db)

def load_css():
    """Load custom CSS"""
    css_path = Path(__file__).parent / "styles.css"
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def main():
    # Initialize State
    SessionState.initialize()
    
    # Load Styles
    load_css()
    
    # Render Sidebar
    render_sidebar()
    
    # Header
    mode = st.session_state.current_mode
    st.markdown(f"# {mode}")
    
    # Dynamic Subtitle explaining the architecture
    if mode == "Auto Mode":
        st.caption("🤖 **Smart Orchestration**: I will automatically route your query to the best agent(s) for the job.")
    elif mode == "Coding Mode":
        st.caption("🤖 **Orchestrator Active**: I will plan tasks and delegate them to the Code, Research, and Content agents automatically.")
    elif mode == "Debugging Mode":
        st.caption("🐛 **Direct Line**: You are talking directly to the **Code Agent** and **Analysis Agent** for instant fixes.")
    elif mode == "Learning Mode":
        st.caption("📚 **Direct Line**: You are talking directly to the **Research Agent** to query the Knowledge Base.")
        
    st.markdown("---")
    
    # Chat Loop
    chat.render()

if __name__ == "__main__":
    main()
