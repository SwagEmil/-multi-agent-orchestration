"""
Test Bug Workflow
Verifies that the bug-triggered research workflow functions correctly.
"""

import sys
import os
import unittest
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from database.db_manager import DatabaseManager
from agents.research_agent import ResearchAgent
from workflows.bug_research_flow import process_pending_bugs

class TestBugWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = DatabaseManager("database/test_workflow.db")
        cls.research_agent = ResearchAgent()
        cls.conv_id = "test_flow_123"
        cls.db.create_conversation(cls.conv_id, "Test flow")

    def test_workflow_execution(self):
        print("\n🔄 Testing Bug-Triggered Research Workflow...")
        
        # 1. Inject a bug
        bug_id = self.db.record_bug(
            conversation_id=self.conv_id,
            description="Memory leak in image processing loop",
            severity="medium",
            code_context="while True: img = load_image()"
        )
        print(f"   Injected Bug #{bug_id}")
        
        # 2. Run the processor (simulating the monitor)
        print("   Running processor...")
        count = process_pending_bugs(self.db, self.research_agent)
        
        # 3. Verify processing happened
        self.assertEqual(count, 1)
        print("   Processor handled 1 bug.")
        
        # 4. Verify DB update
        # We need to query directly since get_pending_bugs won't return it anymore
        cursor = self.db.conn.cursor()
        cursor.execute("SELECT research_completed, research_findings FROM bugs WHERE id = ?", (bug_id,))
        row = cursor.fetchone()
        
        self.assertEqual(row['research_completed'], 1)
        self.assertTrue(len(row['research_findings']) > 10)
        print("   ✅ Bug marked as researched with findings.")

    @classmethod
    def tearDownClass(cls):
        try:
            os.remove("database/test_workflow.db")
            os.remove("database/test_workflow.db-shm")
            os.remove("database/test_workflow.db-wal")
        except OSError:
            pass

if __name__ == "__main__":
    unittest.main()
