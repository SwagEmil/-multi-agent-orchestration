"""
Exhaustive Vertex AI Model Scanner
Tests access to ALL known Google Foundation Models.
"""

import os
import sys
from google.cloud import aiplatform
from langchain_google_vertexai import ChatVertexAI
from dotenv import load_dotenv
from pathlib import Path

# Load env vars
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

def scan_models():
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    
    print(f"🔍 Scanning ALL models for:")
    print(f"   Project: {project_id}")
    print(f"   Location: {location}")
    print("-" * 50)

    # List of all known model IDs to test
    candidates = [
        # Gemini 2.0 (Experimental)
        "gemini-2.0-flash-exp",
        
        # Gemini 1.5 Pro (Latest)
        "gemini-1.5-pro-002",
        "gemini-1.5-pro-001",
        "gemini-1.5-pro",
        
        # Gemini 1.5 Flash (Fast)
        "gemini-1.5-flash-002",
        "gemini-1.5-flash-001",
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
        
        # Gemini 1.0 (Legacy)
        "gemini-1.0-pro-002",
        "gemini-1.0-pro-001",
        "gemini-1.0-pro",
        "gemini-pro",
        
        # PaLM 2 (Legacy)
        "chat-bison@002",
        "chat-bison",
        "text-bison@002"
    ]
    
    working_models = []
    
    for model in candidates:
        print(f"Testing {model:<25} ... ", end="", flush=True)
        try:
            llm = ChatVertexAI(
                model_name=model,
                project=project_id,
                location=location,
                max_output_tokens=1,
                request_timeout=5
            )
            # Try a simple invoke
            llm.invoke("Hi")
            print("✅ WORKING")
            working_models.append(model)
        except Exception as e:
            err = str(e).lower()
            if "404" in err or "not found" in err:
                print("❌ Not Found (404)")
            elif "429" in err or "quota" in err:
                print("⚠️  Rate Limited (But Exists!)")
                working_models.append(model)
            elif "403" in err or "permission" in err:
                print("🚫 Permission Denied (403)")
            else:
                print(f"❌ Error: {str(e)[:50]}")

    print("-" * 50)
    print(f"🎉 FOUND {len(working_models)} WORKING MODELS:")
    for m in working_models:
        print(f"   - {m}")

if __name__ == "__main__":
    scan_models()
