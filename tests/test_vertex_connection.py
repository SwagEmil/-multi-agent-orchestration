"""
Test Vertex AI Connection
Verifies that the system can authenticate and generate text using Vertex AI.
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Load env vars explicitly to be sure
load_dotenv()

from utils.llm_factory import get_llm

def test_connection():
    print("\n☁️ Testing Vertex AI Connection...")
    
    # Check env vars
    use_vertex = os.getenv("USE_VERTEX_AI")
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    
    print(f"   USE_VERTEX_AI: {use_vertex}")
    print(f"   Project: {project}")
    print(f"   Credentials: {creds}")
    
    try:
        # Initialize LLM
        # Testing target model: Gemini 2.0 Flash Experimental
        llm = get_llm(model_name="gemini-2.0-flash-exp")
        
        # Test generation
        print("\n   Sending test prompt...")
        response = llm.invoke("Hello, are you running on Vertex AI?")
        
        print(f"\n   ✅ Success! Response:\n   {response.content}")
        return True
        
    except Exception as e:
        print(f"\n   ❌ Connection Failed: {e}")
        return False

if __name__ == "__main__":
    test_connection()
