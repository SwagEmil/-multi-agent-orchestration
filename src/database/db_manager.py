"""
SQLite database schema for multi-agent orchestration system

Tables:
- conversations: User requests and transcripts
- tasks: Individual tasks assigned to agents  
- agent_interactions: Record of agent executions
- bugs: Bugs identified by Code Agent
"""

import sqlite3
from pathlib import Path
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_path="database/agent_system.db"):
        """Initialize database connection"""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Register adapters for datetime
        sqlite3.register_adapter(datetime, lambda d: d.isoformat())
        sqlite3.register_converter("TIMESTAMP", lambda v: datetime.fromisoformat(v.decode()))
        
        self.conn = sqlite3.connect(
            str(self.db_path), 
            check_same_thread=False,
            detect_types=sqlite3.PARSE_DECLTYPES
        )
        self.conn.row_factory = sqlite3.Row  # Return dicts instead of tuples
        
        # Enable WAL mode for concurrent access
        self.conn.execute("PRAGMA journal_mode=WAL")
        
        self._create_tables()
    
    def _create_tables(self):
        """Create all database tables"""
        cursor = self.conn.cursor()
        
        # Conversations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                user_request TEXT NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Tasks table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                description TEXT NOT NULL,
                assigned_agent TEXT NOT NULL,
                priority TEXT CHECK(priority IN ('high', 'medium', 'low')),
                status TEXT CHECK(status IN ('pending', 'in_progress', 'completed', 'failed')),
                output TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            )
        ''')
        
        # Agent interactions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS agent_interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                agent_name TEXT NOT NULL,
                input TEXT NOT NULL,
                output TEXT,
                rag_context TEXT,
                execution_time_ms INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (task_id) REFERENCES tasks(id)
            )
        ''')
        
        # Bugs table (for Code Agent discoveries)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS bugs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                description TEXT NOT NULL,
                severity TEXT CHECK(severity IN ('high', 'medium', 'low')),
                code_context TEXT,
                identified_by TEXT DEFAULT 'Code Agent',
                research_completed BOOLEAN DEFAULT 0,
                research_findings TEXT,
                retry_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_tasks_conversation ON tasks(conversation_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_bugs_research ON bugs(research_completed)')
        
        self.conn.commit()
        logger.info("Database tables created successfully")
    
    def create_conversation(self, conversation_id: str, user_request: str, metadata: dict = None):
        """Create new conversation"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO conversations (id, user_request, metadata)
            VALUES (?, ?, ?)
        ''', (conversation_id, user_request, json.dumps(metadata) if metadata else None))
        self.conn.commit()
        return conversation_id
    
    def create_task(self, task_id: str, conversation_id: str, description: str, 
                   assigned_agent: str, priority: str = 'medium'):
        """Create new task"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO tasks (id, conversation_id, description, assigned_agent, priority, status)
            VALUES (?, ?, ?, ?, ?, 'pending')
        ''', (task_id, conversation_id, description, assigned_agent, priority))
        self.conn.commit()
        return task_id
    
    def update_task_status(self, task_id: str, status: str, output: str = None):
        """Update task status"""
        cursor = self.conn.cursor()
        cursor.execute('''
            UPDATE tasks 
            SET status = ?, output = ?, completed_at = ?
            WHERE id = ?
        ''', (status, output, datetime.now() if status == 'completed' else None, task_id))
        self.conn.commit()
    
    def record_agent_interaction(self, task_id: str, agent_name: str, input_text: str,
                                output: str = None, rag_context: str = None, exec_time_ms: int = None):
        """Record agent execution"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO agent_interactions (task_id, agent_name, input, output, rag_context, execution_time_ms)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (task_id, agent_name, input_text, output, rag_context, exec_time_ms))
        self.conn.commit()
    
    def record_bug(self, conversation_id: str, description: str, severity: str,
                  code_context: str = None):
        """Record bug found by Code Agent"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO bugs (conversation_id, description, severity, code_context, retry_count)
            VALUES (?, ?, ?, ?, 0)
        ''', (conversation_id, description, severity, code_context))
        self.conn.commit()
        return cursor.lastrowid
    
    def increment_bug_retry(self, bug_id: int):
        """Increment retry count for a bug"""
        cursor = self.conn.cursor()
        cursor.execute('''
            UPDATE bugs 
            SET retry_count = retry_count + 1
            WHERE id = ?
        ''', (bug_id,))
        self.conn.commit()

    def get_pending_bugs(self, conversation_id: str = None, max_retries: int = 3):
        """Get bugs that haven't been researched and haven't exceeded retry limit"""
        cursor = self.conn.cursor()
        if conversation_id:
            cursor.execute('''
                SELECT * FROM bugs 
                WHERE research_completed = 0 
                AND conversation_id = ?
                AND retry_count < ?
                ORDER BY severity DESC, created_at ASC
            ''', (conversation_id, max_retries))
        else:
            cursor.execute('''
                SELECT * FROM bugs 
                WHERE research_completed = 0
                AND retry_count < ?
                ORDER BY severity DESC, created_at ASC
            ''', (max_retries,))
        return cursor.fetchall()
    
    def update_bug_research(self, bug_id: int, findings: str):
        """Mark bug research as complete"""
        cursor = self.conn.cursor()
        cursor.execute('''
            UPDATE bugs 
            SET research_completed = 1, research_findings = ?
            WHERE id = ?
        ''', (findings, bug_id))
        self.conn.commit()
    
    def get_conversation_tasks(self, conversation_id: str):
        """Get all tasks for a conversation"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT * FROM tasks WHERE conversation_id = ?
            ORDER BY created_at
        ''', (conversation_id,))
        return cursor.fetchall()
    
    def close(self):
        """Close database connection"""
        self.conn.close()
