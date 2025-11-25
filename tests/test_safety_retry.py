"""
Test Safety Retry Logic
Verifies that the system stops retrying after max attempts to prevent infinite loops/costs.
"""

import sys
import os
import unittest
from unittest.mock import MagicMock
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from database.db_manager import DatabaseManager
from workflows.bug_research_flow import process_pending_bugs

class TestSafetyRetry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db_path = "database/test_safety.db"
        cls.db = DatabaseManager(cls.db_path)
        cls.conv_id = "test_safety_123"
        cls.db.create_conversation(cls.conv_id, "Test safety")

    def setUp(self):
        # Clear bugs table before each test
        self.db.conn.execute("DELETE FROM bugs")
        self.db.conn.commit()

    def test_max_retries(self):
        print("\n🛡️ Testing Max Retry Safety...")
        
        # 1. Inject a bug
        bug_id = self.db.record_bug(
            conversation_id=self.conv_id,
            description="Tricky bug that crashes agent",
            severity="high"
        )
        
        # 2. Mock a failing agent
        mock_agent = MagicMock()
        mock_agent.execute.side_effect = Exception("API Error or Timeout")
        
        # 3. Run processor multiple times (simulating loop)
        # Attempt 1
        print("   Attempt 1 (Should fail)...")
        process_pending_bugs(self.db, mock_agent)
        
        # Verify retry count = 1
        row = self.db.conn.execute("SELECT retry_count FROM bugs WHERE id=?", (bug_id,)).fetchone()
        self.assertEqual(row['retry_count'], 1)
        
        # Attempt 2
        print("   Attempt 2 (Should fail)...")
        process_pending_bugs(self.db, mock_agent)
        self.assertEqual(self.db.conn.execute("SELECT retry_count FROM bugs WHERE id=?", (bug_id,)).fetchone()['retry_count'], 2)
        
        # Attempt 3
        print("   Attempt 3 (Should fail)...")
        process_pending_bugs(self.db, mock_agent)
        self.assertEqual(self.db.conn.execute("SELECT retry_count FROM bugs WHERE id=?", (bug_id,)).fetchone()['retry_count'], 3)
        
        # Attempt 4 (Should be ignored due to max retries)
        print("   Attempt 4 (Should be skipped)...")
        count = process_pending_bugs(self.db, mock_agent)
        
        # Verify it was NOT processed
        self.assertEqual(count, 0)
        
        # Verify retry count is still 3 (didn't try again)
        self.assertEqual(self.db.conn.execute("SELECT retry_count FROM bugs WHERE id=?", (bug_id,)).fetchone()['retry_count'], 3)
        
        print("   ✅ Safety check passed: Bug ignored after 3 failures.")

    @classmethod
    def tearDownClass(cls):
        try:
            os.remove(cls.db_path)
            os.remove(cls.db_path + "-shm")
            os.remove(cls.db_path + "-wal")
        except OSError:
            pass

if __name__ == "__main__":
    unittest.main()
