# Model Configuration Guide

## Current Model Usage

### All Specialist Agents
**Model**: `gemini-2.0-flash-exp`  
**Temperature**: `0.2`  
**Agents**: Code Agent, Research Agent, Content Agent, Analysis Agent

### Orchestrator Agent
**Model**: `gemini-2.0-flash-exp` (fallback)  
**Temperature**: `0.1`  
**Note**: Originally intended for `gemini-2.0-flash-thinking-exp-1219`, but not available in Vertex AI yet

---

## Recommended Model Configuration

### 1. Orchestrator Agent
**Recommended**: `gemini-2.0-flash-exp` (current) or `gemini-1.5-pro` (better reasoning)  
**Temperature**: `0.1`

**Rationale**:
- Needs strong reasoning for task decomposition and routing
- Benefits from chain-of-thought capabilities
- Lower temperature for consistent routing decisions

**Alternative for Better Reasoning**:
```python
model_name="gemini-1.5-pro"  # Stronger reasoning, higher cost
```

---

### 2. Code Agent
**Recommended**: `gemini-2.0-flash-exp` (current) ✅  
**Temperature**: `0.2`

**Rationale**:
- Fast response for code analysis
- Good balance of speed/quality for bug detection
- Handles structured JSON output well

**Alternative for Complex Code**:
```python
model_name="gemini-1.5-pro"  # For very large codebases
temperature=0.1  # More deterministic
```

---

### 3. Research Agent
**Recommended**: `gemini-2.0-flash-exp` (current) ✅  
**Temperature**: `0.3` (suggested increase)

**Rationale**:
- Fast retrieval and synthesis
- Slightly higher temperature for creative connections
- Good at summarizing RAG context

**Suggested Change**:
```python
temperature=0.3  # More diverse research insights
```

---

### 4. Content Agent
**Recommended**: `gemini-2.0-flash-exp` (current) ✅  
**Temperature**: `0.4` (suggested increase)

**Rationale**:
- Writing benefits from creativity
- Higher temperature = more natural language
- Still deterministic enough for documentation

**Suggested Change**:
```python
temperature=0.4  # Better for creative writing
```

---

### 5. Analysis Agent
**Recommended**: `gemini-2.0-flash-exp` (current) ✅  
**Temperature**: `0.1` (suggested decrease)

**Rationale**:
- Needs precise, factual analysis
- Lower temperature for consistent metrics interpretation
- Deterministic for business decisions

**Suggested Change**:
```python
temperature=0.1  # More deterministic analysis
```

---

## Cost vs. Performance Matrix

| Agent | Current Model | Cost/1M tokens | Speed | Recommended Alternative |
|-------|--------------|----------------|-------|------------------------|
| Orchestrator | gemini-2.0-flash-exp | Free (beta) | ⚡⚡⚡ | gemini-1.5-pro ($3.50) |
| Code Agent | gemini-2.0-flash-exp | Free (beta) | ⚡⚡⚡ | gemini-1.5-flash ($0.075) |
| Research Agent | gemini-2.0-flash-exp | Free (beta) | ⚡⚡⚡ | gemini-1.5-flash ($0.075) |
| Content Agent | gemini-2.0-flash-exp | Free (beta) | ⚡⚡⚡ | gemini-1.5-flash ($0.075) |
| Analysis Agent | gemini-2.0-flash-exp | Free (beta) | ⚡⚡⚡ | gemini-1.5-flash ($0.075) |

**Note**: Gemini 2.0 Flash is experimental and free during beta. Fallback to 1.5 Flash for production.

---

## Production Recommendations

### High-Volume Production
```python
# llm_factory.py
MODEL_FAST = "gemini-1.5-flash"  # Stable, cheap ($0.075/1M)
MODEL_REASONING = "gemini-1.5-pro"  # Better orchestration ($3.50/1M)
```

### Quality-First Production
```python
# llm_factory.py
MODEL_FAST = "gemini-1.5-pro"  # Best quality
MODEL_REASONING = "gemini-1.5-pro"  # Consistent quality
```

### Cost-Optimized Production
```python
# llm_factory.py
MODEL_FAST = "gemini-1.5-flash"  # Cheapest stable model
MODEL_REASONING = "gemini-1.5-flash"  # Same for consistency
```

---

## How to Change Models

### Global Change (Recommended)
Edit `src/utils/llm_factory.py`:
```python
# Model Constants
MODEL_FAST = "gemini-1.5-flash"  # Change this
MODEL_REASONING = "gemini-1.5-pro"  # And this
```

### Per-Agent Customization
Edit individual agent files:

**Example: Use Pro for Code Agent only**
```python
# src/agents/code_agent.py
from utils.llm_factory import get_llm
self.llm = get_llm(
    model_name="gemini-1.5-pro",  # Override for this agent
    temperature=0.1
)
```

---

## Temperature Recommendations by Agent

| Agent | Current Temp | Recommended Temp | Reason |
|-------|-------------|------------------|--------|
| Orchestrator | 0.1 | **0.1** ✅ | Need consistent routing |
| Code Agent | 0.2 | **0.2** ✅ | Good balance for code |
| Research Agent | 0.2 | **0.3** 🔄 | More creative connections |
| Content Agent | 0.2 | **0.4** 🔄 | Natural writing style |
| Analysis Agent | 0.2 | **0.1** 🔄 | Precise, factual analysis |

---

## Implementation Plan for Recommended Changes

### Step 1: Update Temperature for Research Agent
```python
# src/agents/research_agent.py
self.llm = get_llm(
    model_name=MODEL_FAST, 
    temperature=0.3  # Changed from 0.2
)
```

### Step 2: Update Temperature for Content Agent
```python
# src/agents/content_agent.py
self.llm = get_llm(
    model_name=MODEL_FAST, 
    temperature=0.4  # Changed from 0.2
)
```

### Step 3: Update Temperature for Analysis Agent
```python
# src/agents/analysis_agent.py
self.llm = get_llm(
    model_name=MODEL_FAST, 
    temperature=0.1  # Changed from 0.2
)
```

---

## Testing Recommendations

After changing models/temperatures:
1. Run `pytest tests/test_specialist_agents.py`
2. Run `pytest tests/test_end_to_end.py`
3. Check output quality manually via CLI: `python src/main.py`

---

## Fallback Strategy

If Gemini 2.0 Flash becomes unstable or rate-limited:
1. Uncomment line in `llm_factory.py`:
   ```python
   MODEL_FAST = "gemini-1.5-flash"
   ```
2. System will automatically use stable 1.5 Flash across all agents
