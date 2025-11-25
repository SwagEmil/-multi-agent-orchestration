# Day 5: MLOps & Production Deployment for Gen AI - Technical Knowledge Extraction

**Source:** Kaggle 5-Day Gen AI Intensive Course  
**URL:** https://www.youtube.com/watch?v=eZ-8UQ_t4YM

---

## MLOps for Generative AI - New Paradigm

**Traditional MLOps** focused on:
- Training data management
- Model versioning
- Deployment pipelines

**GenAI MLOps adds:**
- Prompt management & versioning
- Chain/workflow orchestration
- Agent-specific lifecycle (AgentOps)
- Tool registry management

---

## Gen AI Lifecycle Phases

### 1. Discovery Phase
- Explore foundation models
- Prototyping with AI Studio
- Experimentation with different prompts/models

### 2. Development & Experimentation
- **Prompt engineering** as first-class artifact
- Prompt templates with placeholders
- RAG chain development
- Agent workflow design

### 3. Evaluation
- **Auto-evaluators** (LLM-as-judge)
- Custom evaluation datasets
- Business KPI tracking  
- Point-wise vs pairwise comparison

### 4. Deployment
- Chaining components (prompts, models, adapters)
- CI/CD for prompt updates
- Version control for all artifacts

### 5. Monitoring
- **Drift detection**: Input distribution, output quality
- Performance tracking (latency, cost)
- Model skew

### 6. Governance
- Artifact lineage (prompts → chains → adapters)
- Compliance tracking
- Access controls

---

## Development Paradigm Shift

### Traditional ML
1. Collect training data
2. Train model from scratch or fine-tune
3. Deploy model

### Generative AI
1. **Start with foundation model**
2. **Adapt via prompt engineering** (primary method)
3. **Optional**: Fine-tuning for specialization
4. **Chain with RAG** for grounding

**Key Insight**: Prompts are **code + data artifacts** (tied to model version).

---

## Data Practices for Gen AI

### Types of Data

1. **Prompts**  
   - Few-shot examples
   - System instructions
   - Templates

2. **Grounding Data**  
   - Vector stores for RAG
   - Knowledge graphs
   - Real-time APIs

3. **Task-Specific Datasets**  
   - Fine-tuning data
   - Evaluation sets

4. **Human Feedback**  
   - RLHF (Reinforcement Learning from Human Feedback)
   - Preference rankings

5. **Synthetic Data**  
   - LLM-generated examples
   - Edge case creation
   - Cost-effective scaling

---

## Evaluation & Monitoring

### Automated Evaluation

**Auto-Evaluators (LLM Judges):**
- Define rubrics for quality
- Use strong model to judge weaker model outputs
- **Point-wise**: Score single response
- **Pairwise**: Compare two responses side-by-side

**Benefit**: Pairwise reduces ties, gives clearer signals.

###  Custom Evaluation Datasets
- **Business-specific KPIs**
- Domain relevance
- Brand voice consistency

### Agent-Specific Metrics
- Tool selection accuracy
- Tool call sequence correctness
- Final output quality despite varied paths

### Multimodal Evaluation
**Emerging area:**
- Text-to-image quality
- Text-to-video coherence
- Use rubric-driven evaluation frameworks

---

## Deployment & Governance

### Artifact Management

**What to Version:**
1. **Prompts**: Templates and few-shot examples
2. **Chains**: Sequence of model calls + logic
3. **Adapters**: Fine-tuned LoRA weights
4. **Tools**: Function definitions for agents

**CI/CD Integration:**
- Prompts treated like code
- Automated tests on prompt updates
- Validation before deployment

### Version Control Best Practices
- **Strict versioning**: Tie prompt to model version
- **Lineage tracking**: Which prompt → which model → which deployment
- **Rollback capability**: Quick revert if quality degrades

### Governance
- **Compliance**: Audit trail for all artifacts
- **Access control**: Who can deploy prompts/models
- **Explainability**: Trace outputs to specific prompts + data

---

## AgentOps (Agent-Specific MLOps)

### Agent Lifecycle Differences

**Agent-specific concerns:**
1. **Tool orchestration** via tool registry
2. **Strategic tool selection** (agents choose from many tools)
3. **Agent evaluation** (trajectory, not just output)
4. **Observability** (memory, reasoning steps, tool traces)
5. **Deployment pipelines** for multi-step workflows

### Tool Registry
- Centralized catalog of available functions
- Descriptions, schemas, authentication
- Version control for tools
- Permissions and access control

### Agent Evaluation (Covered Day 3)
- Tool selection metrics
- Trajectory analysis
- Outcome-based scoring

---

## Vertex AI Platform

**End-to-End GenAI MLOps:**

### Discovery
-  AI Studio for prototyping
- Tuning service

### Development
- Prompt management UI and SDK
- Vertex Experiments (track iterations)
- Evaluation services

### Deployment
- Vertex AI Prediction (scalable endpoints)
- Agent Engine (managed runtime)
- Monitoring dashboards (cost, latency, token usage)

### Governance
- Model Registry (version all artifacts)
- Vertex Pipelines (automate workflows)
- Integrated observability (Cloud Trace, Logging)

---

## Managing Cost & Latency

### Cost Management

**Monitor:**
- Input/output token counts
- Model selection (expensive vs cheap models)
- API call frequency

**Optimize:**
- **Model selection**: Use smallest capable model
- **Caching**: Store common responses
- **Batching**: Group requests where possible

**Vertex AI Dashboards:**
- Integrated observability for cost tracking
- Token consumption metrics
- Estimate costs during development phase

### Latency Optimization

**Strategies:**
1. **Model selection**: Faster models for simple tasks
2. **Streaming**: Return tokens as generated
3. **Parallel calls**: When tasks independent
4. **Caching**: Prompt/response caching

**Monitoring:**
- Track P50, P95, P99 latencies
- Alert on degradation
- A/B test optimizations

---

## Continuous Evaluation in Production

**Post-Deployment Monitoring:**

### Collect Traces
- Every user interaction logged
- Store: user query, model response, tool calls, latency

### Build Evaluation Datasets from Logs
- Sample representative interactions
- Label (human or LLM judge)
- Create ongoing eval set

### Run Periodic Evaluations
- Daily/weekly quality checks
- Compare to baseline
- Detect drift early

### Feedback Loop
- Poor results → investigate
- Update prompts or retrain
- Redeploy and validate

---

## Open Telemetry Integration

**For Custom Observability:**
- Instrument your application
- Send traces to your preferred backend
- Flexibility for specific tools/dashboards

**Vertex AI also provides:**
- Out-of-box dashboards
- Cloud Trace integration
- Looker Studio templates

---

## Prompt Management

### Why It Matters
**Prompts are critical artifacts:**
- Define model behavior
- Tied to specific model versions
- Evolve over time like code

### Vertex AI Prompt Management

**Features:**
1. **UI for iteration**: Test prompts in AI Studio
2. **SDK for deployment**: Programmatic access
3. **Version control**: Track changes over time
4. **Testing integration**: Wire into CI/CD

**Workflow:**
1. Develop prompt in AI Studio
2. Save as versioned template
3. Reference in code via SDK
4. CI/CD validates before deploy

---

 ## Fine-Tuning Integration

**Vertex AI Tuning Service:**

**Capabilities:**
- Full fine-tuning
- LoRA (parameter-efficient)
- Track experiments (hyperparams, data, results)
- Version adapters in Model Registry

**Integration:**
- Deploy multiple adapters to same base model
- A/B test adapters
- Monitor adapter performance vs base

---

## Infrastructure Automation

### Vertex AI Custom Jobs
**For fine-tuning:**
- Spin up multi-GPU clusters programmatically
- Managed infrastructure
- Multiple regions
- Pay only for training time

### Vertex AI Pipelines
**For workflows:**
- Automate data ingestion → embedding → indexing
- Schedule periodic updates
- Orchestrate complex GenAI workflows

---

## Agent Deployment - Agent Engine

**Managed Runtime:**
- **Framework-agnostic**: LangChain, LangGraph, custom
- **Auto-scaling**: Handle load automatically
- **Observability**: Cloud Trace, Logging out-of-box
- **Monitoring**: Latency, costs, errors

**Use Case:**  
Deploy agents without managing servers or Kubernetes.

---

## Multimodal MLOps

**Emerging Challenges:**

### Text-to-Image/Video
- Evaluation harder (subjective quality)
- Rubric-driven frameworks needed
- Human feedback loop critical

### Multimodal Inputs
- Images, audio, video as inputs
- Evaluation requires modality-specific metrics

---

## Production Checklist

### Before Deployment
- [ ] Evaluation suite created
- [ ] Baseline metrics established
- [ ] Cost budget defined
- [ ] Latency requirements set
- [ ] Monitoring dashboards configured
- [ ] Rollback plan ready
- [ ] Governance/compliance reviewed

### After Deployment
- [ ] Monitor cost and latency
- [ ] Collect user feedback
- [ ] Run periodic evaluations
- [ ] Track drift metrics
- [ ] Iterate on prompts/models
- [ ] Document changes

---

## 5-Year Future Outlook

**Predictions:**

1. **Acceleration**: Faster model improvements, shorter iteration cycles
2. **Agent orchestrators**: Multiple semi-autonomous agents collaborating
3. **Better embeddings**: Multimodal, matryoshka, efficient mapping
4. **Improved tooling**: Even easier prototyping → production
5. **Standardization**: Industry standards for prompts, agents, evaluation

**Adaptation crucial**: Field evolving too fast for rigid specialization.

---

## Key Takeaways

1. **GenAI MLOps extends traditional MLOps** with prompts, chains, agents as first-class artifacts
2. **Prompt engineering primary adaptation** method (vs training from scratch)
3. **Evaluation paradigm shift**: Auto-evaluators, pairwise comparison, agent trajectories
4. **AgentOps** adds tool management, trajectory evaluation, observability complexity
5. **Vertex AI provides end-to-end platform** from prototyping to production
6. **Cost & latency monitoring critical** - track tokens, choose models carefully
7. **Continuous evaluation from logs** - build datasets from production traces
8. **Governance essential**: Version control prompts, adapters, tools like code
9. **Multimodal introduces new challenges** - evaluation, data types, monitoring
10. **Speed of change unprecedented** - stay adaptable, keep learning, iterate fast
