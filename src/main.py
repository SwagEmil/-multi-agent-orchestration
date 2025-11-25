#!/usr/bin/env python3
"""
Multi-Agent Orchestration System - CLI Interface

Test the orchestrator agent with RAG integration.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from agents.orchestrator import OrchestratorAgent
from database.db_manager import DatabaseManager
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

def main():
    """Main CLI loop"""
    
    print("=" * 70)
    print("MULTI-AGENT ORCHESTRATION SYSTEM - PHASE 1 TEST")
    print("=" * 70)
    print()
    
    # Check API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ ERROR: GOOGLE_API_KEY not found in .env file")
        print()
        print("Please:")
        print("1. Copy .env.example to .env")
        print("2. Add your Google API key")
        print("3. Get key from: https://makersuite.google.com/app/apikey")
        return
    
    try:
        # Initialize
        print("🔧 Initializing system...")
        db = DatabaseManager()
        orchestrator = OrchestratorAgent(db)
        
        print("✅ System initialized")
        print("📊 RAG database: 512 chunks loaded")
        print("🤖 Orchestrator: Gemini 2.0 Flash Thinking (reasoning)")
        print("💾 Database: database/agent_system.db")
        print()
        print("=" * 70)
        print("Enter requests to test orchestrator routing")
        print("Type 'quit' to exit")
        print("=" * 70)
        print()
        
        # Test loop
        while True:
            user_input = input("\n📝 You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            try:
                print("\n⏳ Processing...\n")
                
                # Create conversation and analyze
                conv_id, task_id, routing = orchestrator.create_conversation(user_input)
                
                print("=" * 70)
                print("🧠 ORCHESTRATOR ANALYSIS")
                print("=" * 70)
                print()
                print(f"💭 Reasoning:")
                print(f"   {routing['reasoning']}")
                print()
                print(f"🎯 Selected Agent: {routing['agent']}")
                print()
                print(f"📋 Task Instructions:")
                print(f"   {routing['instructions']}")
                print()
                print(f"📚 RAG Context Used:")
                print(f"   {routing.get('rag_context', 'No context retrieved')[:200]}...")
                print()
                print(f"💾 Conversation ID: {conv_id}")
                print(f"📌 Task ID: {task_id}")
                print()
                print("=" * 70)
                print("⏸️  [Agent execution will be implemented in Phase 2]")
                print("=" * 70)
                
            except Exception as e:
                print(f"\n❌ Error: {e}")
                logger.exception("Error processing request")
        
        print("\n👋 Goodbye!")
        db.close()
        
    except Exception as e:
        print(f"\n❌ Initialization failed: {e}")
        logger.exception("Failed to initialize system")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
