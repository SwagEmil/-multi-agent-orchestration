"""
Test Bug Database Integration
Verifies that bugs can be recorded, retrieved, and updated.
"""

import sys
import os
import unittest
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from database.db_manager import DatabaseManager

class TestBugDatabase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = DatabaseManager("database/test_bugs.db")
        cls.conv_id = "test_conv_123"
        cls.db.create_conversation(cls.conv_id, "Test request")

    def test_bug_lifecycle(self):
        print("\n🐞 Testing Bug Lifecycle...")
        
        # 1. Record a bug
        bug_id = self.db.record_bug(
            conversation_id=self.conv_id,
            description="SQL Injection vulnerability in login",
            severity="high",
            code_context="SELECT * FROM users WHERE name = " + "user_input"
        )
        print(f"   Recorded bug ID: {bug_id}")
        self.assertIsNotNone(bug_id)
        
        # 2. Retrieve pending bugs
        pending = self.db.get_pending_bugs(self.conv_id)
        print(f"   Pending bugs: {len(pending)}")
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]['description'], "SQL Injection vulnerability in login")
        
        # 3. Update research
        self.db.update_bug_research(bug_id, "Use parameterized queries.")
        print("   Updated research findings.")
        
        # 4. Verify no longer pending
        pending_after = self.db.get_pending_bugs(self.conv_id)
        print(f"   Pending bugs after update: {len(pending_after)}")
        self.assertEqual(len(pending_after), 0)
        
        print("   ✅ Bug lifecycle passed")

    @classmethod
    def tearDownClass(cls):
        # Cleanup
        try:
            os.remove("database/test_bugs.db")
            os.remove("database/test_bugs.db-shm")
            os.remove("database/test_bugs.db-wal")
        except OSError:
            pass

if __name__ == "__main__":
    unittest.main()
