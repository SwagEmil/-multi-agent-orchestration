# How to Enable Different Gemini Models in Vertex AI

## Quick Access Links

**Direct Link to Enable APIs:**
https://console.cloud.google.com/apis/library?project=code-review-479122

**Vertex AI Model Garden:**
https://console.cloud.google.com/vertex-ai/model-garden?project=code-review-479122

---

## Step-by-Step Guide

### Step 1: Enable Vertex AI API (Already Done ✅)
You already have Vertex AI API enabled. Skip to Step 2.

---

### Step 2: Access Model Garden

1. Go to: https://console.cloud.google.com/vertex-ai/model-garden?project=code-review-479122
2. Or navigate: **Google Cloud Console** → **Vertex AI** → **Model Garden**

---

### Step 3: Enable Gemini 1.5 Pro

#### Option A: Via Model Garden UI
1. In Model Garden, search for **"Gemini 1.5 Pro"**
2. Click on the **Gemini 1.5 Pro** card
3. Click **"ENABLE"** button
4. Wait 1-2 minutes for activation

#### Option B: Via gcloud CLI
```bash
gcloud services enable aiplatform.googleapis.com --project=code-review-479122
```

---

### Step 4: Grant Permissions (Already Done ✅)

Your service account already has **Vertex AI Administrator** role, so no additional permissions needed.

---

### Step 5: Verify Model Availability

Run this test:
```bash
cd "/Users/emilwiecek/Documents/projects/Code Reviews/multi-agent-orchestration"
python tests/test_vertex_connection.py
```

---

## Available Gemini Models

Once enabled, you can use these models:

### Gemini 2.0 (Experimental - Free)
- `gemini-2.0-flash-exp` ✅ **Currently Active**
- Fast, free during beta
- Used by all agents currently

### Gemini 1.5 (Stable - Paid)
- `gemini-1.5-pro-001` - Best reasoning ($3.50/1M input tokens)
- `gemini-1.5-flash-001` - Fast and cheap ($0.075/1M input tokens)

### Gemini 1.0 (Legacy)
- `gemini-1.0-pro` - Older model, still available

---

## How to Switch Models

### Method 1: Global Change (All Agents)

Edit `src/utils/llm_factory.py`:
```python
# Model Constants
MODEL_FAST = "gemini-1.5-flash-001"  # Change this line
MODEL_PRO = "gemini-1.5-pro-001"      # And this line
```

Then restart your application.

---

### Method 2: Per-Agent Change

**Example: Use Pro for Orchestrator**

Edit `src/agents/orchestrator.py`:
```python
from utils.llm_factory import get_llm
self.llm = get_llm(
    model_name="gemini-1.5-pro-001",  # Specify model directly
    temperature=0.1
)
```

**Example: Use Pro for Analysis Agent**

Edit `src/agents/analysis_agent.py`:
```python
from utils.llm_factory import MODEL_PRO
super().__init__(
    agent_name="Analysis Agent",
    role="Data Analyst & Strategist",
    model_name=MODEL_PRO,  # Uncomment this line
    temperature=0.1
)
```

---

## Testing After Model Change

Always test after changing models:

```bash
# Test all agents
pytest tests/test_specialist_agents.py -v

# Test end-to-end
pytest tests/test_end_to_end.py -v

# Test Vertex connection
python tests/test_vertex_connection.py
```

---

## Troubleshooting

### Error: "404 Publisher Model not found"

**Cause**: Model not enabled or not available in your region

**Fix**:
1. Check Model Garden: https://console.cloud.google.com/vertex-ai/model-garden?project=code-review-479122
2. Ensure model is enabled
3. Try different region (currently `us-central1`)

**Change Region**:
Edit `.env`:
```bash
GOOGLE_CLOUD_LOCATION=us-west1  # Try different region
```

---

### Error: "403 Permission Denied"

**Cause**: Service account missing permissions

**Fix**:
1. Go to IAM: https://console.cloud.google.com/iam-admin/iam?project=code-review-479122
2. Find: `account-1@code-review-479122.iam.gserviceaccount.com`
3. Ensure it has: **Vertex AI Administrator** (already set ✅)

---

## Cost Comparison

| Model | Input Cost | Output Cost | Speed | Use Case |
|-------|-----------|-------------|-------|----------|
| gemini-2.0-flash-exp | **FREE** (beta) | **FREE** (beta) | ⚡⚡⚡ | Current (all agents) |
| gemini-1.5-flash-001 | $0.075/1M | $0.30/1M | ⚡⚡⚡ | Production (cost-efficient) |
| gemini-1.5-pro-001 | $3.50/1M | $10.50/1M | ⚡⚡ | High-quality reasoning |

**Recommendation**: 
- Development: Use `gemini-2.0-flash-exp` (free)
- Production: Mix of Flash (cheap) and Pro (quality)

---

## Recommended Production Configuration

Edit `src/utils/llm_factory.py`:
```python
# Hybrid approach: Fast for most, Pro for reasoning
MODEL_FAST = "gemini-1.5-flash-001"  # Cheap, stable
MODEL_PRO = "gemini-1.5-pro-001"      # Premium reasoning
```

Then enable Pro for Orchestrator and Analysis Agent (uncomment lines in their `__init__` methods).

This gives you:
- 80% cost savings (Code, Research, Content use Flash)
- 20% premium quality (Orchestrator, Analysis use Pro)
