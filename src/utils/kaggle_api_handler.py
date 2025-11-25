"""
Kaggle Secrets-Compatible API Key Handler
For Google AI Agents Intensive Capstone Project

This module safely loads Google API keys from Kaggle Secrets or environment variables.
Judges can fork the notebook and add their own keys without modifying code.
"""

import os
import logging

logger = logging.getLogger(__name__)

def get_google_api_key():
    """
    Load Google API Key with fallback chain:
    1. Kaggle Secrets (production/submission)
    2. Environment variable (local development)
    3. None (with helpful error message)
    
    Returns:
        str: API key or None if not found
    """
    # Try Kaggle Secrets first (for notebook submissions)
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        api_key = user_secrets.get_secret("GOOGLE_API_KEY")
        logger.info("✅ Production Mode: API Key loaded from Kaggle Secrets")
        return api_key
    except ImportError:
        # Not running in Kaggle environment
        pass
    except Exception as e:
        # Kaggle environment but secret not set
        logger.debug(f"Kaggle Secrets not available: {e}")
    
    # Fallback to environment variable (local testing)
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        logger.info("✅ Dev Mode: API Key loaded from environment variable")
        return api_key
    
    # No key found - provide helpful instructions
    logger.error("⚠️ ALERT: Google API Key not found.")
    print("\n" + "="*70)
    print("⚠️  Google API Key Not Found")
    print("="*70)
    print("\nTo run this agent, you need a Google API key:")
    print("\n📋 FOR KAGGLE NOTEBOOK USERS:")
    print("   1. Click 'Add-ons' → 'Secrets' in the top menu")
    print("   2. Add a new secret:")
    print("      - Label: GOOGLE_API_KEY")
    print("      - Value: [Your API key from https://makersuite.google.com/]")
    print("   3. Click 'Save & Run All'")
    print("\n📋 FOR LOCAL DEVELOPMENT:")
    print("   1. Copy .env.example to .env")
    print("   2. Add your key: GOOGLE_API_KEY=your_key_here")
    print("   3. Run: python src/main.py")
    print("="*70 + "\n")
    
    return None


def setup_google_api():
    """
    Initialize Google API key in environment.
    Call this once at the start of your notebook/script.
    
    Returns:
        bool: True if key was successfully loaded
    """
    api_key = get_google_api_key()
    
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
        logger.info("🔑 Google API configured successfully")
        return True
    else:
        logger.warning("❌ Google API configuration failed")
        return False


# For Vertex AI (optional)
def get_vertex_ai_config():
    """
    Get Vertex AI configuration from environment.
    
    Returns:
        dict: Configuration with project_id, location, credentials_path
    """
    config = {
        "use_vertex": os.getenv("USE_VERTEX_AI", "false").lower() == "true",
        "project_id": os.getenv("GOOGLE_CLOUD_PROJECT"),
        "location": os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
        "credentials_path": os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    }
    
    if config["use_vertex"]:
        logger.info(f"🌐 Vertex AI Mode: Project={config['project_id']}, Location={config['location']}")
    
    return config


if __name__ == "__main__":
    # Test the API key loading
    print("Testing API Key Handler...\n")
    
    if setup_google_api():
        print("✅ SUCCESS: API key loaded and ready to use!")
        print(f"   Key starts with: {os.environ.get('GOOGLE_API_KEY', '')[:10]}...")
    else:
        print("❌ FAILED: Could not load API key")
        print("   Follow the instructions above to configure your key")
