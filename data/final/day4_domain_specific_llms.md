# Day 4: Domain-Specific LLMs & Fine-Tuning - Technical Knowledge Extraction

**Source:** Kaggle 5-Day Gen AI Intensive Course  
**URL:** https://www.youtube.com/watch?v=AN2tpHi26OE

---

## Specializing LLMs for Domains

**Goal:**  
Adapt general-purpose LLMs for specific domains (healthcare, cybersecurity, finance) that require specialized knowledge and compliance.

**Key Domains:**
- **Healthcare**: MedLM, MedPaLM
- **Cybersecurity**: SecLM
- **Finance, Legal, Scientific Research**

---

## When to Use Fine-Tuning vs RAG

### RAG (Retrieval Augmented Generation)
**Best for:**
- Dynamic, frequently updating information
- Large knowledge bases (millions of documents)
- When information needs to change without retraining

**Limitations:**
- Retrieval quality depends on search accuracy
- Context window constraints
- May miss nuanced domain knowledge

### Fine-Tuning
**Best for:**
- Specialized behavior/tone/style
- Domain-specific reasoning patterns
- Compliance requirements (e.g., medical disclaimers)
- When knowledge is stable

**Limitations:**
- Expensive (compute, data labeling)
- Knowledge cutoff at training time
- Catastrophic forgetting risk

### Hybrid Approach (Recommended)
- **Fine-tune** for domain expertise, style, reasoning
- **RAG** for current facts, references, citations

---

## Fine-Tuning Strategies

### Full Fine-Tuning
- **Method**: Update all model parameters
- **Data needs**: Large datasets (thousands to millions of examples)
- **Cost**: Very high compute requirements
- **Use case**: Creating entirely new domain model

### Parameter-Efficient Fine-Tuning (PEFT)

#### LoRA (Low-Rank Adaptation)
- **Method**: Add small trainable matrices to frozen model
- **Parameters updated**: <1% of total model
- **Benefits**:  
  - Much faster training
  - Lower compute cost
  - Smaller adapter files (easy to version/deploy)
- **Trade-off**: Slightly lower performance than full fine-tuning

#### Adapter Tuning
- **Method**: Insert small neural network layers
- **Benefits**: Multiple adapters for different tasks
- **Use case**: Single base model → specialized adapters per domain

---

## MedLM & MedPaLM (Healthcare AI)

### MedLM
- **Purpose**: Medical question answering, clinical documentation
- **Training**: Fine-tuned on medical literature, clinical notes
- **Capabilities**:
  - Diagnosis suggestions
  - Treatment recommendations
  - Clinical documentation assistance

### MedPaLM
- **Benchmark**: Medical licensing exam questions (USMLE-style)
- **Performance**: Expert-level accuracy on medical reasoning
- **Safety**: Extensive validation, calibrated uncertainty

### Healthcare-Specific Challenges
1. **Regulatory Compliance**  
   - HIPAA, GDPR for patient data
   - Model explainability for clinical decisions

2. **High Stakes**  
   - Errors can harm patients
   - Need calibrated confidence scores

3. **Domain Knowledge Depth**  
   - Medical terminology
   - Drug interactions
   - Diagnostic reasoning

---

## SecLM (Cybersecurity AI)

### Purpose
Specialized LLM for cybersecurity tasks:
- Threat detection
- Vulnerability analysis
- Security code review
- Incident response assistance

### Training Data
- Security bulletins and CVE databases
- Malware analysis reports
- Security best practices
- Code vulnerability patterns

### Key Capabilities
- Identify security vulnerabilities in code
- Suggest remediation strategies
- Explain attack vectors
- Map to security frameworks (MITRE ATT&CK)

---

## Catastrophic Forgetting

### Problem
When fine-tuning on domain-specific data, model "forgets" general capabilities.

**Example:**  
- Fine-tune on medical data
- Model excels at medical questions
- But loses ability to answer general knowledge questions

### Mitigation Strategies

#### 1. Replay Mixing
- Include general-purpose data in fine-tuning
- Mix 10-30% general examples with domain data
- **Example**: For every 100 medical examples, include 20 general Q&A pairs

#### 2. Regularization
- Constrain how much parameters can change
- Keeps model close to original weights
- Techniques: L2 regularization, weight decay

#### 3. Progressive Fine-Tuning
- Start with general fine-tuning
- Gradually increase domain-specific data ratio
- Slower but preserves general capabilities better

#### 4. Adapter-Based Approaches
- Keep base model frozen (preserves general knowledge)
- Only train domain-specific adapters
- **Best of both**: General + specialized capabilities

---

## Domain-Specific Evaluation

### Why Different from General Evaluation

**Domain metrics matter more:**  
- Medical: Diagnostic accuracy, treatment safety
- Cyber: Vulnerability detection rate, false positives
- Legal: Citation accuracy, precedent relevance

### Evaluation Approaches

#### 1. Human Expert Review
- Gold standard but expensive
- Scale limited (can't eval every model update)

#### 2. Benchmark Datasets
- Medical: USMLE, PubMedQA, BioASQ
- Code Security: CodeXGLUE, Sec

Bench
- Finance: FinQA, ConvFinQA

#### 3. Auto-Eval with LLM Judges
- Use expert-validated rubrics
- LLM checks for domain-specific criteria
- **Example**: Medical response must include uncertainty, contraindications

#### 4. A/B Testing in Production
- Deploy to subset of users
- Monitor domain-specific metrics
- Collect expert feedback loop

---

## Safety & Alignment for Specialized Domains

### Medical Safety
- **Conservative responses**: When uncertain, suggest consulting doctor
- **No diagnosis claims**: Assistance only, not replacement for professionals
- **Clear disclaimers**: Every medical response includes limitations
- **Uncertainty calibration**: Model knows when it doesn't know

### Cybersecurity Ethics
- **No malicious use**: Refuse to generate exploit code
- **Defensive focus**: Emphasize protection, not attack
- **Responsible disclosure**: Encourage proper vulnerability reporting

---

## Production Considerations

### Vertex AI Fine-Tuning Service
**Features:**
- Managed infrastructure for fine-tuning jobs
- Multiple GPU support across regions
- Integrated with Vertex Experiments (track hyperparameters, results)
- Vertex Model Registry (version adapters and models)

**Workflow:**
1. Upload training data to Cloud Storage
2. Launch fine-tuning job (Vertex AI Custom Jobs)
3. Monitor in Vertex Experiments
4. Deploy adapter to Vertex AI Prediction
5. A/B test against base model

### Adapter Management
**Best Practices:**
- Version control adapters like code
- Tag with metadata (dataset, hyperparameters, metrics)
- Store in Artifact Registry
- Deploy multiple adapters to same base model

### Cost Optimization
**Strategies:**
- Use PEFT instead of full fine-tuning
- Distill large fine-tuned model → smaller model
- Cache common domain queries
- Batch inference requests

---

## RAG + Fine-Tuning Hybrid Example

**Medical Documentation Assistant:**

1. **Base Model**: Gemini 1.5 Pro

2. **Fine-Tuning (LoRA)**:  
   - Medical terminology
   - Clinical documentation style
   - SOAP note format (Subjective, Objective, Assessment, Plan)

3. **RAG Component**:  
   - Patient's previous medical records
   - Latest drug interaction databases
   - Current clinical guidelines

**Workflow:**
- User: "Generate progress note for patient X"
- RAG retrieves: Recent vitals, medication list, prior notes
- Fine-tuned model: Formats in clinical style with medical terminology
- Output: Properly formatted SOAP note with current patient data

---

## Fine-Tuning Data Requirements

### Data Quality > Data Quantity

**Key Principles:**
1. **Representative**: Cover full task distribution
2. **Diverse**: Include edge cases and variations
3. **Clean**: No errors, consistent formatting
4. **Balanced**: Equal representation of categories

### Minimum Data Sizes
- **Full fine-tuning**: 10K - 100K+ examples
- **LoRA**: 1K - 10K examples
- **Few-shot in-context**: 5-100 examples

**Synthetic Data:**
- Use base LLM to generate training examples
- Expert validation required
- Cost-effective scaling

---

## Regulatory & Compliance

### Healthcare (HIPAA, GDPR)
- **Data residency**: Where training data stored
- **Model access controls**: Who can query model
- **Audit trails**: Log all model interactions
- **De-identification**: Remove PII from training data

### Financial Services
- **Explainability**: Justify model decisions for regulators
- **Fairness**: No discriminatory outputs
- **Data retention**: Compliance with financial records laws

---

## Key Takeaways

1. **Fine-tuning adapts LLMs** for specialized domains beyond general knowledge
2. **PEFT (LoRA) preferred** for cost/efficiency vs full fine-tuning
3. **Hybrid RAG + fine-tuning** combines strengths: domain expertise + current facts
4. **Catastrophic forgetting prevented** via replay mixing, regularization, or adapters
5. **Domain-specific eval crucial** - use expert benchmarks, human review, domain metrics
6. **Med LM/MedPaLM** demonstrate healthcare AI feasibility with proper validation
7. **SecLM** shows cybersecurity domain adaptation
8. **Safety critical** in high-stakes domains: uncertainty, disclaimers, conservative outputs
9. **Adapter management** like code: version control, metadata, multiple deployments
10. **Regulatory compliance** required: data residency, audit trails, explainability
