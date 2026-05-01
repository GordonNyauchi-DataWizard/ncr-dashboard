# NCR Root Cause Classifier Dashboard

ETQ-style Streamlit app powered by BERTopic (SBERT) for structured root cause classification.

---

## How It Works

1. User enters a free-text root cause description
2. BERTopic returns **3 category suggestions** with confidence scores + a **No Match** option
3. User clicks the best match — one click, no dropdown
4. Each root cause saved as its own structured record (ETQ field structure)
5. Export records as Excel, CSV, or PDF-ready HTML
6. Analytics dashboard: frequency charts, trends, descriptive stats

---

## Setup

### Files needed in GitHub repo
```
app.py
requirements.txt
```
> The training dataset is NOT in GitHub — it is loaded securely from Azure Blob Storage.

---

## Training Data — Azure Blob Storage (Secure)

The app loads `Root_Causes_Final_Dataset.xlsx` directly from a private Azure Blob Storage container via a SAS URL stored in Streamlit secrets.

### Steps to configure:

**1. Upload the Excel to Azure Blob Storage**
- Go to portal.azure.com → your storage account
- Open Storage browser → Blob containers → your container
- Upload `Root_Causes_Final_Dataset.xlsx`
- Click the file → Generate SAS → set expiry 1 year → copy the Blob SAS URL

**2. Add the URL to Streamlit Secrets**
- Go to share.streamlit.io → your app → Settings → Secrets
- Add:
```toml
TRAINING_DATA_URL = "https://your-account.blob.core.windows.net/container/Root_Causes_Final_Dataset.xlsx?sp=r&..."
```
- Click Save — app will redeploy automatically

**3. Sidebar will show:**
```
✅ Trained from Azure Blob Storage (secure) — 14,341 rows, 58 categories
```

---

## Deploy on Streamlit Community Cloud

1. Push `app.py` and `requirements.txt` to a public GitHub repo
2. Go to https://share.streamlit.io → Create app → Deploy a public app from GitHub
3. Paste the GitHub URL to app.py (e.g. `https://github.com/username/repo/blob/main/app.py`)
4. Click Deploy
5. Add the `TRAINING_DATA_URL` secret in app settings

---

## ETQ Field Mapping

| ETQ Field | App Field |
|---|---|
| NCR Number | NCR Number |
| Root Cause Description | RC Description |
| Root Cause Category | Root Cause Category (BERTopic suggested) |
| Return or Eliminate | Retain or Eliminate |
| Rationale | Rationale |
| Investigation Information | Investigation Info |
| Containment | Containment |
| Risk Assessment | Risk Assessment |
| CAPA Decision | Disposition / CAPA |

---

## Model Details

- **Model:** `paraphrase-multilingual-mpnet-base-v2` (supports English + Spanish)
- **Method:** SBERT cosine similarity against per-category centroids built from training data
- **Confidence threshold:** 0.30
- **Excluded categories:** `Other` and `Cannot be determined` (never shown as suggestions)
- **Output:** Top 3 matching categories + No Match option
- **Fallback:** If no training data available, uses category name embeddings directly
