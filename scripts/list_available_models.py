"""
List Available Vertex AI Models
Queries the Model Garden to find which Foundation Models are actually available
for the current project and location.
"""

import os
from google.cloud import aiplatform
from google.auth import default
from dotenv import load_dotenv
from pathlib import Path

# Load env vars
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

def list_models():
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    
    print(f"🔍 Checking available models for:")
    print(f"   Project: {project_id}")
    print(f"   Location: {location}")

    try:
        aiplatform.init(project=project_id, location=location)
        
        # There isn't a simple "list all foundation models" API that works consistently 
        # for all users without specific permissions, but we can try to test 
        # specific known models to see which ones respond.
        
        from langchain_google_vertexai import ChatVertexAI
        
        candidates = [
            "gemini-2.0-flash-exp",
            "gemini-1.5-pro-001",
            "gemini-1.5-flash-001",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-1.0-pro",
            "gemini-1.0-pro-001",
            "gemini-pro"
        ]
        
        print("\n🧪 Testing Model Availability:")
        available = []
        
        for model in candidates:
            print(f"   Checking {model}...", end="", flush=True)
            try:
                llm = ChatVertexAI(
                    model_name=model,
                    project=project_id,
                    location=location,
                    max_output_tokens=1
                )
                llm.invoke("Hi")
                print(" ✅ AVAILABLE")
                available.append(model)
            except Exception as e:
                if "404" in str(e) or "not found" in str(e).lower():
                    print(" ❌ Not Found")
                elif "429" in str(e):
                    print(" ⚠️  Rate Limited (Available)")
                    available.append(model)
                else:
                    print(f" ❌ Error: {str(e)[:50]}...")

        print("\n📋 Summary of Available Models:")
        for m in available:
            print(f"   - {m}")
            
    except Exception as e:
        print(f"\n❌ Critical Error: {e}")

if __name__ == "__main__":
    list_models()
