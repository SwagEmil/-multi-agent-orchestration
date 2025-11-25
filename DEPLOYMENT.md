# Deployment Guide

## Option 1: Local Docker

### Build
```bash
docker build -t agent-orchestrator .
```

### Run
```bash
docker run -p 8501:8501 \
  -e GOOGLE_APPLICATION_CREDENTIALS=/app/service-account.json \
  -v $(pwd)/service-account.json:/app/service-account.json \
  agent-orchestrator
```

Access at `http://localhost:8501`

---

## Option 2: Google Cloud Run

### Prerequisites
- Google Cloud project with billing enabled
- `gcloud` CLI installed
- Service account with Vertex AI permissions

### Deploy
```bash
# Set project
gcloud config set project YOUR_PROJECT_ID

# Deploy
gcloud run deploy agent-orchestrator \
  --source . \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars USE_VERTEX_AI=true,GOOGLE_CLOUD_PROJECT=YOUR_PROJECT_ID
```

### Get URL
```bash
gcloud run services describe agent-orchestrator \
  --region us-central1 \
  --format="value(status.url)"
```

---

## Option 3: Vertex AI Agent Engine

### Setup
```bash
# Enable APIs
gcloud services enable aiplatform.googleapis.com

# Deploy agent
gcloud ai agents deploy \
  --region=us-central1 \
  --display-name="orchestrator" \
  --source=src/adk_agents/orchestrator.py
```

### Test
```bash
curl -X POST \
  https://YOUR_REGION-aiplatform.googleapis.com/v1/projects/YOUR_PROJECT/locations/YOUR_REGION/agents/YOUR_AGENT:predict \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -d '{"query": "What are types of agents?"}'
```

---

## Environment Variables

Required:
- `USE_VERTEX_AI=true`
- `GOOGLE_CLOUD_PROJECT=your-project-id`
- `GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json`

Optional:
- `GOOGLE_CLOUD_LOCATION=us-central1` (default)

---

## Troubleshooting

### 404 Model Not Found
- Ensure Vertex AI API is enabled
- Check service account has `Vertex AI User` role
- Verify `gemini-2.5-pro` is available in your region

### Permission Denied
```bash
gcloud projects add-iam-policy-binding YOUR_PROJECT \
  --member="serviceAccount:YOUR_SA@YOUR_PROJECT.iam.gserviceaccount.com" \
  --role="roles/aiplatform.user"
```

### ChromaDB Not Found
```bash
# Rebuild vector database
python scripts/reingest_rag.py
```
