"""
LLM Factory
Centralizes LLM initialization to support both Google AI Studio and Vertex AI.
Compatible with Kaggle Secrets for secure API key management.
"""

import os
import logging
from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI

# Import Kaggle-compatible API handler
try:
    from utils.kaggle_api_handler import setup_google_api
    KAGGLE_HANDLER_AVAILABLE = True
except ImportError:
    KAGGLE_HANDLER_AVAILABLE = False

# We import Vertex AI conditionally to avoid hard dependency if not used
try:
    from langchain_google_vertexai import ChatVertexAI
    VERTEX_AVAILABLE = True
except ImportError:
    VERTEX_AVAILABLE = False

logger = logging.getLogger(__name__)

# Model Constants
MODEL_FAST = "gemini-2.0-flash-exp"  # For fast agents (Content, Research, Code)
MODEL_PRO = "gemini-2.5-pro"  # Newest reasoning model (Confirmed Working)
MODEL_REASONING = MODEL_PRO  # Alias for backward compatibility
# Future: Perplexity integration for Research Agent web search

def get_llm(model_name: str = MODEL_FAST, temperature: float = 0.2):
    """
    Get an LLM instance based on environment configuration.
    
    Args:
        model_name: Name of the model to use
        temperature: Temperature for generation
        
    Returns:
        ChatGoogleGenerativeAI or ChatVertexAI instance
    """
    # Initialize API key if Kaggle handler is available
    if KAGGLE_HANDLER_AVAILABLE:
        setup_google_api()
    
    use_vertex = os.getenv("USE_VERTEX_AI", "false").lower() == "true"
    
    if use_vertex:
        if not VERTEX_AVAILABLE:
            logger.warning("⚠️ Vertex AI requested but langchain-google-vertexai not installed. Falling back to AI Studio.")
            return _get_ai_studio_llm(model_name, temperature)
            
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        
        if not project_id:
            logger.warning("⚠️ GOOGLE_CLOUD_PROJECT not set. Vertex AI might fail if not using default credentials.")
            
        credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if credentials_path:
            logger.info(f"🔑 Using Service Account credentials from: {credentials_path}")
        else:
            logger.info("🔑 Using Application Default Credentials (ADC)")
            
        logger.info(f"🔌 Using Vertex AI: {model_name} (Project: {project_id}, Loc: {location})")
        
        return ChatVertexAI(
            model_name=model_name,
            temperature=temperature,
            project=project_id,
            location=location,
            credentials=None, # Let it use GOOGLE_APPLICATION_CREDENTIALS env var
        )
    else:
        return _get_ai_studio_llm(model_name, temperature)

def _get_ai_studio_llm(model_name: str, temperature: float):
    """Helper to get AI Studio LLM"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.warning("⚠️ GOOGLE_API_KEY not set. LLM calls will fail.")
        
    logger.info(f"✨ Initializing Google AI Studio LLM: {model_name}")
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        google_api_key=api_key
    )
