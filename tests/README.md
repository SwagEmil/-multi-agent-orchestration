# Test Suite Documentation

## Current Active Tests (18 total)

### Core Integration Tests
- **test_end_to_end.py** (2 tests)
  - `test_full_flow_code_analysis`: Full user request → Orchestrator → CodeAgent → DB pipeline
  - `test_full_flow_research`: Full user request → Orchestrator → ResearchAgent → DB pipeline

- **test_system_integration.py** (5 tests) 🆕
  - `test_concurrent_database_writes`: Verifies SQLite WAL mode handles concurrent writes
  - `test_rag_retrieval_quality`: Validates RAG returns relevant context
  - `test_agent_error_recovery`: Tests graceful handling of malformed inputs
  - `test_orchestrator_routing_consistency`: Ensures routing determinism
  - `test_database_schema_integrity`: Validates all tables and columns exist

### Component Tests
- **test_specialist_agents.py** (4 tests)
  - Tests individual execution of Code, Research, Content, and Analysis agents

- **test_bug_db.py** (1 test)
  - Validates bug lifecycle: record → retrieve → update

- **test_bug_workflow.py** (1 test)
  - Tests bug-triggered research workflow

- **test_safety_retry.py** (1 test)
  - Verifies retry limit prevents infinite loops

### UI Tests
- **test_ui_checkpoints.py** (3 tests) 🆕
  - `test_checkpoint1_task_creation`: Validates Checkpoint 1 logic
  - `test_checkpoint2_task_execution_flow`: Validates Checkpoint 2 logic
  - `test_checkpoint3_final_plan_generation`: Validates Checkpoint 3 logic

### Connection Tests
- **test_vertex_connection.py** (1 test)
  - Validates Vertex AI connection and authentication

## Legacy Tests (moved to tests/legacy/)
- `ALL_RAG_TESTS.py`: Phase 0 RAG validation tests
- `test_phase1_routing.py`: Early orchestrator routing tests
- `RAG_TEST_ANALYSIS.md`, `STRESS_TEST_RESULTS.md`, `VERIFICATION_TEST2_RESULTS.md`: Historical test reports

## Test Coverage Summary

### ✅ Well Covered
- Agent execution (all 4 specialist agents)
- Database operations (CRUD operations)
- Bug workflow (creation → research → resolution)
- Safety mechanisms (retry limits)
- End-to-end integration (orchestrator → agents → DB)
- Concurrent operations (multi-threaded DB writes)
- RAG retrieval quality
- UI checkpoint logic

### ⚠️ Identified Gaps (Potential Future Tests)
1. **Performance Testing**: No load tests for sustained high traffic
2. **API Rate Limit Handling**: No tests for graceful degradation under rate limits
3. **Large Payload Testing**: No tests for extremely large user inputs (10k+ chars)
4. **LangGraph State Transitions**: No explicit tests for workflow state graph navigation
5. **Multi-Conversation Concurrency**: No tests for handling 100+ simultaneous conversations

## Running Tests

### Run All Active Tests
```bash
pytest tests/ -v --ignore=tests/legacy
```

### Run Specific Test Suite
```bash
pytest tests/test_system_integration.py -v
pytest tests/test_end_to_end.py -v
```

### Run with Coverage Report
```bash
pytest tests/ --cov=src --cov-report=html --ignore=tests/legacy
```

## Test Maintenance Notes
- All tests use isolated test databases (auto-cleaned in `tearDownClass`)
- Database adapters registered to prevent deprecation warnings
- Tests designed for parallel execution (no shared state)
- RAG retrieval tests depend on `data/vector_db/` being populated
