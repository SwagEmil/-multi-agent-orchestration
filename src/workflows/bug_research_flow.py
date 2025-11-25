"""
Bug-Triggered Research Workflow
Monitors the database for new bugs and triggers the Research Agent.
"""

import sys
import time
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.db_manager import DatabaseManager
from agents.research_agent import ResearchAgent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_pending_bugs(db, research_agent):
    """
    Process all currently pending bugs once.
    Returns the number of bugs processed.
    """
    # Only get bugs that haven't exceeded max retries (default 3)
    pending_bugs = db.get_pending_bugs(max_retries=3)
    processed_count = 0
    
    if pending_bugs:
        logger.info(f"Found {len(pending_bugs)} pending bugs!")
        
        for bug in pending_bugs:
            bug_id = bug['id']
            description = bug['description']
            severity = bug['severity']
            code_context = bug['code_context']
            retry_count = bug['retry_count']
            
            logger.info(f"🔍 Researching Bug #{bug_id} (Attempt {retry_count + 1}): {description}")
            
            # Create Research Task
            task = {
                "description": f"Research solutions for this {severity} severity bug: {description}",
                "context": {
                    "bug_id": bug_id,
                    "severity": severity,
                    "code_context": code_context,
                    "source": "Bug Monitor"
                }
            }
            
            # Execute Research Agent
            try:
                result = research_agent.execute(task)
                findings = result.get('findings', 'No findings returned.')
                
                # Update Database
                db.update_bug_research(bug_id, findings)
                logger.info(f"✅ Research completed for Bug #{bug_id}")
                processed_count += 1
                
            except Exception as e:
                logger.error(f"❌ Research failed for Bug #{bug_id}: {e}")
                # Increment retry count so we don't loop forever
                db.increment_bug_retry(bug_id)
                
    return processed_count

def run_bug_monitor(interval_seconds=5):
    """
    Continuously monitor for new bugs and trigger research.
    """
    db = DatabaseManager()
    research_agent = ResearchAgent()
    
    logger.info("🐞 Bug Monitor started. Waiting for bugs...")
    
    try:
        while True:
            process_pending_bugs(db, research_agent)
            # Wait before next check
            time.sleep(interval_seconds)
            
    except KeyboardInterrupt:
        logger.info("🛑 Bug Monitor stopped.")
    finally:
        db.close()

if __name__ == "__main__":
    run_bug_monitor()
