"""
Session State Management
Handles chat history, active mode, and user preferences.
"""

import streamlit as st
from typing import List, Dict, Any
import uuid
from datetime import datetime

class SessionState:
    @staticmethod
    def initialize():
        """Initialize all session state variables"""
        if 'initialized' not in st.session_state:
            st.session_state.initialized = True
            st.session_state.conversation_id = f"conv_{uuid.uuid4().hex[:12]}"
            st.session_state.messages = []
            st.session_state.current_mode = "Auto Mode"  # Default to Auto Mode
            st.session_state.show_reasoning = True # Assuming this was the intended value
            st.session_state.model_preference = "gemini-2.0-flash-exp"
            st.session_state.sidebar_expanded = True
        
        # Active Mode (Learning, Debugging, Coding)
        if "current_mode" not in st.session_state:
            st.session_state.current_mode = "Auto Mode"
            
        # Chat History (List of message dicts)
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
        # Current Conversation ID
        if "conversation_id" not in st.session_state:
            st.session_state.conversation_id = str(uuid.uuid4())
            
        # User Preferences
        if "model_preference" not in st.session_state:
            st.session_state.model_preference = "gemini-2.0-flash-exp"
            
        # Sidebar State
        if "sidebar_expanded" not in st.session_state:
            st.session_state.sidebar_expanded = True

    @staticmethod
    def add_message(role: str, content: str, mode: str = None):
        """Add a message to the history"""
        message = {
            "id": str(uuid.uuid4()),
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "mode": mode or st.session_state.current_mode
        }
        st.session_state.messages.append(message)
        
    @staticmethod
    def get_messages() -> List[Dict[str, Any]]:
        """Get all messages for current session"""
        return st.session_state.messages
    
    @staticmethod
    def clear_history():
        """Clear chat history"""
        st.session_state.messages = []
        st.session_state.conversation_id = str(uuid.uuid4())
        
    @staticmethod
    def set_mode(mode: str):
        """Switch active mode"""
        if mode != st.session_state.current_mode:
            st.session_state.current_mode = mode
            # Optional: Add a system message indicating mode switch
            # SessionState.add_message("system", f"Switched to {mode}")
