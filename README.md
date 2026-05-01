# NCR Root Cause Classifier Dashboard

ETQ-style Streamlit app powered by BERTopic (SBERT) for structured root cause classification.

## How It Works

1. User enters a free-text root cause description
2. BERTopic returns **3 category suggestions** with confidence scores + a **No Match** option
3. User clicks the best match — one click, no dropdown
4. Each root cause saved as its own record with full ETQ field structure
5. Export records as Excel, CSV, or PDF-ready HTML
6. Analytics dashboard: frequency charts, trends, descriptive stats

## Setup

Place these files in the same folder:
```
app.py
requirements.txt
Root_Causes_Final_Dataset.xlsx   ← your training data
```

Then:
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push files to a public GitHub repo
2. Go to https://share.streamlit.io → Create app
3. Point to your repo and app.py
4. Deploy — live URL in ~2 minutes

## ETQ Field Mapping

| ETQ Field | App Field |
|---|---|
| NCR Number | NCR Number |
| Root Cause Description | RC Description |
| Root Cause Category | Root Cause Category (BERTopic) |
| Return or Eliminate | Retain or Eliminate |
| Rationale | Rationale |
| Investigation Information | Investigation Info |
| Containment | Containment |
| Risk Assessment | Risk Assessment |
| CAPA Decision | Disposition / CAPA |

## Notes

- `Other` and `Cannot be determined` are excluded from suggestions (matches original script logic)
- Confidence threshold: 0.30 (SBERT cosine similarity)
- Model: `paraphrase-multilingual-mpnet-base-v2` — supports English and Spanish
- Without training Excel, app runs in fallback mode using category name embeddings
