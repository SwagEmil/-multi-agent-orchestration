# Day 3: AI Agents & Function Calling - Technical Knowledge Extraction

**Source:** Kaggle 5-Day Gen AI Intensive Course  
**URL:** https://youtube.com/watch?v=g6MVIEzFTjY

---

## What are AI Agents?

**Definition:**  
AI agents are systems that can **observe**, **reason**, and **act** autonomously to accomplish tasks.

**Core Loop:**  
1. **Observe**: Perceive environment and context
2. **Reason**: Process information and plan actions  
3. **Act**: Execute actions via tools/functions

---

## Agent Architecture Components

### 1. Tools
Mechanisms that extend agent capabilities:
- **Extensions**: Built-in model capabilities
- **Functions**: Custom code the agent can call
- **Data Stores**: Access to external knowledge (databases, APIs, files

)

### 2. Orchestration Layer
Frameworks that manage agent behavior:
- **ReAct** (Reasoning + Acting):  
  - Agent reasons about what to do
  - Takes action
  - Observes result
  - Iterates until task complete

- **Chain-of-Thought**:  
  - Break complex tasks into steps
  - Reason through each step explicitly
  - Improves accuracy on complex problems

### 3. Function Calling
**Critical for Production Agents**

**Why It Matters:**  
- Agents need to interact with real-world systems (APIs, databases, services)
- Quality of function calling directly impacts agent reliability

**Best Practices:**  
- Clear function descriptions
- Well-defined schemas
- Error handling
- Authentication/authorization

---

## Model Context Protocol (MCP)

**Purpose:**  
Standardized way for LLMs to interact with external tools and data sources.

**Benefits:**  
- Tool interoperability across different agent frameworks
- Easier integration of new tools
- Security and access control standards

**Google Cloud Support:**  
MCP servers enable agents to securely access tools and services.

---

## Multi-Agent Systems

### Architectures

#### Hierarchical
- **Manager agent**: Coordinates and delegates
- **Worker agents**: Execute specific subtasks
- **Use case**: Complex workflows with clear task decomposition

#### Collaborative
- **Peer agents**: Work together on shared goal
- **Communication**: Agents share information and coordinate
- **Use case**: Tasks requiring diverse expertise

### Benefits
- Specialization: Each agent optimized for specific domain
- Scalability: Add agents as needed
- Fault tolerance: System continues if one agent fails

---

## Agentic RAG

**Traditional RAG:**  
Query → Retrieve → Generate

**Agentic RAG:**  
- **Agent decides** when to retrieve information
- **Multiple retrieval steps** based on intermediate results
- **Query reformulation**: Agent refines queries based on retrieved content
- **Tool selection**: Choose between multiple data sources

**Advantages:**  
- More accurate results through iterative refinement
- Handles complex, multi-step questions
- Adapts retrieval strategy dynamically

---

## Agent Evaluation

### Key Challenges
**Traditional metrics don't capture agent quality:**
- Agents have non-deterministic paths to solutions
- Same task may require different tool sequences
- Success depends on final outcome, not just single response

### Evaluation Approaches

#### 1. Tool Selection Metrics
- Did agent pick the right tools?
- In the correct order?
- With appropriate parameters?

#### 2. Trajectory Evaluation
- Assess entire sequence of actions
- Check for efficiency (avoid redundant steps)
- Validate reasoning at each step

#### 3. Outcome-Based Metrics
- Did agent accomplish the task?
- How long did it take?
- Resource usage (API calls, computation)

#### 4. LLM-as-Judge
- Use another LLM to evaluate agent behavior
- Define rubrics for acceptable agent actions
- Automate evaluation at scale

### Custom Metrics (Vertex AI)
Define domain-specific metrics:
- Correlation between tool usage and response quality
- Task-specific success criteria
- Business logic validation

---

## Agent Observability

### Critical for Production

**What to Monitor:**
1. **Memory/State Tracking**  
   - What information agent retains across turns
   - Context window usage
   - State transitions

2. **Tool Call Tracing**  
   - Which tools called, when, and why
   - Tool call latency
   - Success/failure rates

3. **Reasoning Steps**  
   - Agent's decision-making process
   - Why certain actions chosen
   - Debugging failed attempts

**Vertex AI Agent Engine:**  
- Cloud Trace integration out-of-the-box
- Logs for each agent step
- Monitoring dashboards for costs and latency

**Open Telemetry:**  
- DIY integration option
- Custom observability frameworks
- Flexibility for specific requirements

---

## Agent Security

**Critical Considerations:**

### Authentication & Authorization
- Control which APIs/data agents can access
- Token/credential management
- Principle of least privilege

### Adversarial Attack Mitigation
- Prompt injection defenses
- Input validation
- Output filtering

### Data Privacy
- Sensitive data handling
- PII protection
- Audit trails

**Like Mobile App Stores:**  
Agents need approval processes similar to iPhone/Android apps:
- Permissions for specific resources
- Security reviews
- Access controls (camera, location → APIs, databases)

---

## Production Deployment

### Agent Starter Pack (Google Cloud)
**Problem**: Months to production (3-9 months typical)

**Solution**: Production-ready templates

**Includes:**
- API server
- UI playground
- CI/CD pipelines (Cloud Build)
- Terraform samples for infrastructure
- Load testing against Agent Engine
- Automated observability (Cloud Trace, Looker dashboards)

**Workflow:**
1. Select agent template
2. Customize for business logic
3. Deploy to staging (automated)
4. Run load tests
5. One-button deploy to production

### Vertex AI Agent Engine
**Managed Runtime:**
- Framework-agnostic (LangChain, LangGraph, etc.)
- Automatic scaling
- Built-in observability
- Cloud Trace, Logging, Monitoring integration

**Use Case:**  
Deploy custom agents without managing infrastructure.

---

## Agent Ops Principles

### CI/CD for Agents
- **Version control**: Prompts, tools, agent definitions
- **Testing**: Automated evaluation on representative tasks
- **Rollback**: Quick revert if agent performance degrades
- **Staged rollout**: Canary deployments

### Synthetic Data Generation
**For Training & Evaluation:**
- Use LLMs to generate test scenarios
- Create edge cases programmatically
- Scalable compared to human labeling

**For Tool Development:**
- Generate diverse input examples
- Test tool robustness

---

## Framework Landscape

**Popular Options:**
- **LangChain**: Comprehensive toolkit, large ecosystem
- **LangGraph**: State-based agent orchestration
- **Vertex AI**: Managed platform integration

**Selection Criteria:**
- Team expertise
- Integration requirements
- Scale needs
- Observability requirements

---

## Real-World Applications

### NotebookLM
- **Functionality**: Document understanding and Q&A
- **Agent behavior**: Retrieves from uploaded documents
- **Innovation**: Generates podcast-style summaries

### Project Mariner
- **Functionality**: Browser automation
- **Agent behavior**: Navigates web pages, fills forms
- **Use case**: Complex multi-step web tasks

---

## Best Practices

### 1. Start Simple
- Single-agent before multi-agent
- Clear, focused tasks
- Limited tool set initially

### 2. Evaluation First
- Define success metrics before building
- Create eval dataset early
- Automated testing in CI/CD

### 3. Observability from Day 1
- Log all agent actions
- Track costs (API calls, tokens)
- Monitor latency

### 4. Iterate on Tool Descriptions
- Clear, unambiguous instructions
- Include examples in descriptions
- Test with edge cases

### 5. Handle Failures Gracefully
- Retry logic for transient errors
- Fallback behaviors
- Human escalation paths

---

## Function Calling Deep Dive

### Quality Factors

**Model Capability:**
- Reasoning ability crucial for function selection
- Iterative problem-solving (try → fail → retry)
- Understanding complex instructions

**Function Schema Design:**
- Clear parameter names and types
- Comprehensive descriptions
- Example values where helpful

**Context Management:**
- Function call history
- Intermediate results
- Error messages from previous attempts

### Gemini Function Calling
- High-quality function calling in Gemini models
- Supports complex, multi-step function sequences
- Handles ambiguous requests via clarification

---

## Challenges & Future Directions

### Current Challenges
1. **Consistency**: Agents can take unpredictable paths
2. **Evaluation**: No standard benchmarks
3. **Cost**: Multiple API calls expensive at scale
4. **Latency**: Agent loops add overhead

### Research Areas
1. **Auto-evaluation frameworks**: LLM-based judges for agent quality
2. **Efficient tool registries**: Manage thousands of tools
3. **Memory management**: Long-running agents with conversation history
4. **Multi-agent communication protocols**: Standardize agent-to-agent interaction

---

## Key Takeaways

1. **Agents = Observe + Reason + Act** in autonomous loop
2. **Function calling quality** directly impacts agent reliability in production
3. **MCP** provides standardization for tool access
4. **Multi-agent systems** enable specialization and scalability
5. **Agentic RAG** improves on traditional RAG through dynamic retrieval
6. **Evaluation is hard** - requires custom metrics, trajectory analysis, LLM judges
7. **Observability crucial** - trace, log, monitor all agent actions
8. **Security non-negotiable** - authentication, authorization, attack mitigation
9. **Agent Ops** extends MLOps with agent-specific needs (tool versioning, trajectory eval)
10. **Frameworks matter** - choose based on team needs, scale, observability requirements
