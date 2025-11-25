#!/usr/bin/env python3
"""
Automated Routing Test for Multi-Agent Orchestration
Tests if the orchestrator correctly routes tasks to the 4 specialist agents.
"""

import sys
import os
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
from agents.orchestrator import OrchestratorAgent
from database.db_manager import DatabaseManager

load_dotenv()

def run_tests():
    print("=" * 70)
    print("🧪 PHASE 1 ROUTING TEST SUITE")
    print("=" * 70)

    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ GOOGLE_API_KEY missing")
        return

    try:
        db = DatabaseManager()
        orchestrator = OrchestratorAgent(db)
        print("✅ System initialized")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return

    # Test Cases
    test_cases = [
        {
            "name": "Code Agent Test",
            "input": "Review the authentication module for security vulnerabilities and fix any SQL injection bugs.",
            "expected_agent": "code_agent"
        },
        {
            "name": "Research Agent Test",
            "input": "Find the latest documentation on LangGraph state management and best practices for persistence.",
            "expected_agent": "research_agent"
        },
        {
            "name": "Content Agent Test",
            "input": "Write a summary of the meeting notes and draft a blog post about our new agent architecture.",
            "expected_agent": "content_agent"
        },
        {
            "name": "Analysis Agent Test",
            "input": "Analyze the user engagement metrics from the last month and identify drop-off points.",
            "expected_agent": "analysis_agent"
        }
    ]

    results = []
    
    print("\n🚀 Running 4 Test Cases...\n")

    for test in test_cases:
        print(f"📋 Testing: {test['name']}")
        print(f"   Input: {test['input'][:60]}...")
        
        try:
            # Run orchestrator
            conv_id, task_id, routing = orchestrator.create_conversation(test['input'])
            
            actual_agent = routing['agent']
            reasoning = routing['reasoning']
            
            # Check result
            success = actual_agent == test['expected_agent']
            status = "✅ PASS" if success else "❌ FAIL"
            
            print(f"   Result: {status}")
            print(f"   Routed to: {actual_agent}")
            if not success:
                print(f"   Expected: {test['expected_agent']}")
            
            results.append({
                "name": test['name'],
                "success": success,
                "actual": actual_agent,
                "reasoning": reasoning
            })
            
            # Rate limit pause
            time.sleep(2)
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results.append({
                "name": test['name'],
                "success": False,
                "error": str(e)
            })

    # Summary
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for r in results if r.get('success'))
    total = len(results)
    
    for r in results:
        status = "✅" if r.get('success') else "❌"
        print(f"{status} {r['name']}: {r.get('actual', 'Error')}")
    
    print(f"\nTotal: {passed}/{total} Passed")
    
    if passed == total:
        print("\n🎉 ALL ROUTING TESTS PASSED!")
    else:
        print("\n⚠️ SOME TESTS FAILED")

    db.close()

if __name__ == "__main__":
    run_tests()
