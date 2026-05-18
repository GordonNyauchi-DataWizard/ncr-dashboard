# NCR Classification Dashboard
## Root Causes & Corrective Actions

Beautiful, simple Streamlit app powered by BERTopic (SBERT) for structured classification of:
- **Root Causes** → Select from 58 pre-defined categories
- **Corrective Actions** → Select from configurable categories
- **Records Browser** → View, filter, and export all saved records

---

## 🎨 Design

- **Clean white background** with red accent colors (#dc2626)
- **Minimalist UI** — No clutter, only what matters
- **Three tabs**: Root Causes | Corrective Actions | Records Browser
- **One-click suggestions** — Enter description → BERTopic suggests 3 matches → Click one
- **Export options** — Excel, CSV, HTML (print-to-PDF)

---

## How It Works

### Root Causes Tab
1. Enter a free-text root cause description
2. BERTopic analyzes it using SBERT embeddings
3. Returns **top 3 matching categories** with confidence scores
4. Click a suggestion or select "No Match"
5. Fill in NCR Number, Status, Retain/Eliminate decision
6. Click **Save Root Cause Record**

### Corrective Actions Tab
Same flow as Root Causes:
1. Enter corrective action description
2. Get 3 suggestions from SBERT
3. Select one or "No Match"
4. Fill in NCR Number, Status, Effectiveness
5. Save the record

### Records Browser Tab
- View all saved records with filters (Type, NCR Number, Status)
- Summary metrics (Total records, by type, unique NCRs)
- Export as **Excel** (with summary sheet), **CSV**, or **HTML** (print-to-PDF)
- Clear all records with one click

---

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the App
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

### 3. Deploy on Streamlit Community Cloud
```
1. Push app.py and requirements.txt to GitHub
2. Go to https://share.streamlit.io → Create app
3. Select your GitHub repo and branch
4. Deploy
```

---

## Configuration

### Customize Root Cause Categories

Edit the `ROOT_CAUSE_CATEGORIES` list in `app.py`:

```python
ROOT_CAUSE_CATEGORIES = [
    "Your Category 1",
    "Your Category 2",
    # ... add more
]
```

### Customize Corrective Action Categories

Edit the `CORRECTIVE_ACTION_CATEGORIES` list:

```python
CORRECTIVE_ACTION_CATEGORIES = [
    "Design Modification",
    "Documentation Update",
    "Equipment Maintenance",
    # ... customize to your needs
]
```

### Change Confidence Threshold

In the `get_top_suggestions()` function, adjust `confidence_threshold`:

```python
def get_top_suggestions(..., confidence_threshold=0.30):
    # Lower = more suggestions but lower quality
    # Higher = stricter matching
```

---

## Model Details

- **Embedding Model**: `paraphrase-multilingual-mpnet-base-v2`
  - Supports English and Spanish
  - Better for multilingual environments
  - Swap for `all-mpnet-base-v2` if English-only

- **Method**: SBERT cosine similarity against per-category centroids
  - Centroids computed from category names
  - Embeddings cached for performance

- **Output**: Top 3 matching categories + "No Match" option

---

## Database & Persistence

All records are stored in **session state** (in-memory).

### To Add Database Persistence:

Replace the in-memory `st.session_state.records` with:
- **SQL**: SQLite, PostgreSQL, MySQL
- **NoSQL**: Firebase, MongoDB, DynamoDB
- **Cloud**: Azure Blob Storage, Google Cloud Storage, AWS S3

Example with SQLite:
```python
import sqlite3
conn = sqlite3.connect("ncr_records.db")
# Save/load records using SQL
```

---

## Customization Ideas

1. **Add more fields**:
   - Investigation details
   - Risk assessment
   - Containment actions
   - CAPA decision

2. **Connect to training data**:
   - Load category centroids from Azure Blob Storage
   - Update centroids on-the-fly as new records are saved

3. **Add analytics dashboard**:
   - Category frequency charts
   - Trends over time
   - Effectiveness metrics

4. **Enable offline mode**:
   - Cache embeddings locally
   - Use pre-computed centroids file

---

## Troubleshooting

### "Model loading is slow"
- First run downloads SBERT model (~500MB)
- Streamlit caches it with `@st.cache_resource`
- Subsequent runs are instant

### "Suggestions don't match my input"
- Adjust `confidence_threshold` lower (0.20–0.25)
- Add more example descriptions to category lists
- Check text is clean (no special characters, extra spaces)

### "Records disappear after refresh"
- They're stored in session state (in-memory)
- To persist: export to Excel before closing, or add database

---

## File Structure

```
.
├── app.py                 # Main Streamlit app
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| **Streamlit** | Web UI framework |
| **Sentence-Transformers** | SBERT embeddings (paraphrase-multilingual-mpnet-base-v2) |
| **scikit-learn** | Cosine similarity computation |
| **pandas** | Data handling & export |
| **openpyxl** | Excel generation |

---

## License

Open source. Modify as needed for your organization.

---

## Support

For issues or feature requests, contact the development team.
