# NCR Root Cause Classifier Dashboard
## BERTopic-Powered ETQ-style Root Cause Entry + Analytics

---

## What This App Does
- Mimics the ETQ Root Cause Analysis subform workflow
- Uses **BERTopic (SBERT cosine similarity)** to suggest top-3 root cause categories from free-text descriptions
- Lets users select the best category, save records with full ETQ field structure
- Export as **Excel**, **CSV**, or **PDF-ready HTML**
- Built-in **Analytics Dashboard**: frequency charts, trends, descriptive stats

---

## Quickstart (Local)

```bash
# 1. Clone or copy the app.py and requirements.txt to a folder
# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
streamlit run app.py
```

App opens at http://localhost:8501

---

## Deploy on Azure (App Service — Recommended for Presentation)

### Option A: Azure Web App (simplest, ~10 min)

**Prerequisites:** Azure CLI installed, logged in (`az login`)

```bash
# 1. Create resource group
az group create --name ncr-rca-rg --location eastus

# 2. Create App Service Plan (B2 = 2 core CPU, enough for SBERT)
az appservice plan create \
  --name ncr-rca-plan \
  --resource-group ncr-rca-rg \
  --sku B2 \
  --is-linux

# 3. Create Web App (Python 3.10)
az webapp create \
  --resource-group ncr-rca-rg \
  --plan ncr-rca-plan \
  --name ncr-root-cause-app \
  --runtime "PYTHON:3.10"

# 4. Set startup command
az webapp config set \
  --resource-group ncr-rca-rg \
  --name ncr-root-cause-app \
  --startup-file "streamlit run app.py --server.port 8000 --server.address 0.0.0.0"

# 5. Deploy (zip deploy)
zip -r deploy.zip app.py requirements.txt
az webapp deployment source config-zip \
  --resource-group ncr-rca-rg \
  --name ncr-root-cause-app \
  --src deploy.zip
```

App will be available at: `https://ncr-root-cause-app.azurewebsites.net`

> **Note:** First cold start takes ~2-3 min while SBERT model downloads. Subsequent loads use cache.

---

### Option B: Azure Container Instance (fastest cold start if you pre-bake model)

```dockerfile
# Dockerfile (create alongside app.py)
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Pre-download the SBERT model during build (avoids cold start delay)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')"
COPY app.py .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
# Build and push
az acr create --name ncrrcaacr --resource-group ncr-rca-rg --sku Basic --admin-enabled true
az acr build --registry ncrrcaacr --image ncr-rca:latest .

# Deploy container
az container create \
  --resource-group ncr-rca-rg \
  --name ncr-rca-container \
  --image ncrrcaacr.azurecr.io/ncr-rca:latest \
  --cpu 2 --memory 4 \
  --ports 8501 \
  --dns-name-label ncr-root-cause \
  --registry-login-server ncrrcaacr.azurecr.io \
  --registry-username $(az acr credential show --name ncrrcaacr --query username -o tsv) \
  --registry-password $(az acr credential show --name ncrrcaacr --query passwords[0].value -o tsv)
```

---

### Option C: Azure ML Compute (if you already have AML workspace)

Run directly in your existing Azure ML environment — no new resources needed:

```bash
# In your AML terminal / notebook terminal
pip install -r requirements.txt
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

Then expose port 8501 via the AML compute's public URL.

---

## Using with Your Training Data

1. Open the app sidebar → "Upload Root_Causes.xlsx"
2. Click "Retrain Centroids from Data"
3. The SBERT centroids will be rebuilt from your **14,341 real training rows** → much better suggestions

This replaces the default "category-name-only" embeddings with real example-based centroids.

---

## ETQ Field Mapping

| ETQ Field | App Field |
|---|---|
| NCR Number | NCR Number |
| Problem Description | Problem Description |
| Description for Root Cause #N | RC Description |
| Root Cause Category | Root Cause Category (BERTopic-suggested or manual) |
| Return or Eliminate Root Cause | Retain or Eliminate |
| Rationale | Rationale |
| Investigation Information | Investigation Info |
| Containment | Containment |
| Risk Assessment | Risk Assessment |
| CAPA Decision | Disposition / CAPA |

---

## Architecture

```
User types RC description
        ↓
   SBERT encodes text
   (paraphrase-multilingual-mpnet-base-v2)
        ↓
   Cosine similarity vs category centroids
        ↓
   Top-3 suggestions shown as clickable buttons
        ↓
   User selects / overrides category
        ↓
   Record saved to session state
        ↓
   Export: Excel | CSV | HTML→PDF
        ↓
   Analytics: charts + descriptive stats
```
