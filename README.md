# NCR Classification Dashboard
## Root Causes & Corrective Actions

Beautiful, simple Streamlit app powered by BERTopic (SBERT) for structured classification of:
- **Root Causes** → Select from 58 pre-defined categories, auto-assigned NCR ID
- **Corrective Actions** → Select from configurable categories, auto-assigned NCR ID
- **Records Browser** → View, filter, and export all saved records

---

## 🎨 Design

- **Clean white background** with navy blue accent colors (#1c3060)
- **Minimalist UI** — No clutter, only what matters
- **Three tabs**: Root Causes | Corrective Actions | Records Browser
- **One-click suggestions** — Enter description → BERTopic suggests 3 matches → Click one
- **Auto-generated NCR IDs** — No manual entry needed (NCR-00001, NCR-00002, etc.)
- **Export options** — Excel, CSV, HTML (print-to-PDF)

---

## How It Works

### Root Causes Tab
1. Enter a free-text root cause description
2. BERTopic analyzes it using SBERT embeddings
3. Returns **top 3 matching categories** with confidence scores
4. Click a suggestion or select "No Match"
5. Select **Status** (Open, Closed, In Progress)
6. Click **Save Root Cause Record**
7. ✓ **Auto-assigned NCR ID** (e.g., NCR-00001)

### Corrective Actions Tab
Same flow as Root Causes:
1. Enter corrective action description
2. Get 3 suggestions from SBERT
3. Select one or "No Match"
4. Select **Status** (Open, Closed, In Progress)
5. Select **Effectiveness** (Effective, Ineffective, Pending Review)
6. Save the record
7. ✓ **Auto-assigned NCR ID** (e.g., NCR-00002)

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

### Setup Training Data from Azure Blob Storage (Recommended)

For significantly improved confidence scores, load real training datasets from Azure:

#### **Step 1: Prepare Training Datasets**

**Root Causes Dataset** (`Root_Causes_Final_Dataset.xlsx`):
- Column A: `Description` (text descriptions of root causes)
- Column B: `Category` (root cause category)
- Minimum 50+ rows recommended

**Corrective Actions Dataset** (`Corrective_Actions_Final_Dataset.xlsx`):
- Column A: `Description` (text descriptions of actions)
- Column B: `Category` (action category)
- Minimum 50+ rows recommended

#### **Step 2: Upload to Azure Blob Storage**

1. Go to Azure Portal → Your Storage Account
2. Create container: `ncr-training-data`
3. Upload both Excel files
4. Generate SAS URLs for each file (set expiry: 1 year)

#### **Step 3: Configure Streamlit Secrets**

For **Streamlit Community Cloud**:
1. Go to `share.streamlit.io` → Your app → Settings → Secrets
2. Add:

```toml
TRAINING_DATA_RC_URL = "https://your-account.blob.core.windows.net/container/Root_Causes_Final_Dataset.xlsx?sp=r&st=..."
TRAINING_DATA_CA_URL = "https://your-account.blob.core.windows.net/container/Corrective_Actions_Final_Dataset.xlsx?sp=r&st=..."
```

For **Local development** (`.streamlit/secrets.toml`):
```toml
TRAINING_DATA_RC_URL = "https://..."
TRAINING_DATA_CA_URL = "https://..."
```

#### **How It Works**

- App loads training data on startup
- Computes centroids from real descriptions (not just category names)
- **Confidence scores improve dramatically** (0.60+ instead of 0.15-0.35)
- Falls back to category names if URLs not configured

---

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

### Customize NCR ID Format

Edit the `generate_ncr_id()` function to change the numbering scheme:

```python
# Default: NCR-00001
def generate_ncr_id():
    ncr_id = f"NCR-{st.session_state.ncr_counter:05d}"
    st.session_state.ncr_counter += 1
    return ncr_id

# Alternative: NCR-2025-001 (with year)
def generate_ncr_id():
    import datetime
    year = datetime.datetime.now().year
    ncr_id = f"NCR-{year}-{st.session_state.ncr_counter:03d}"
    st.session_state.ncr_counter += 1
    return ncr_id
```

---

## Model Details

- **Embedding Model**: `paraphrase-multilingual-mpnet-base-v2`
  - Supports English and Spanish
  - Better for multilingual environments
  - Swap for `all-mpnet-base-v2` if English-only

- **Training Data** (Optional but Recommended):
  - Load from Azure Blob Storage (see Configuration section)
  - Excel format: Description | Category columns
  - Computes per-category centroids from real text examples
  - **Without training data**: Uses category names only (lower confidence)
  - **With training data**: Uses real examples (higher confidence: 0.60+)

- **Method**: SBERT cosine similarity against per-category centroids
  - Matches user input to closest category centroid
  - Returns top 3 matches with confidence scores
  - "No Match" option always available

- **Output**: Top 3 matching categories + "No Match" option

---

## Database & Persistence

All records are stored in **session state** (in-memory).

Records saved include:
- DateTime
- Type (Root Cause or Corrective Action)
- **NCR Number** (auto-generated)
- Description
- Category
- Confidence score
- Status
- Effectiveness (for Corrective Actions only)

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

2. **Improve training data** ✅ (Now Implemented):
   - Load category centroids from Azure Blob Storage
   - Use real description examples instead of category names
   - Dramatically improves confidence scores
   - See Configuration section for setup

3. **Add analytics dashboard**:
   - Category frequency charts
   - Trends over time
   - Effectiveness metrics
   - Confidence score distribution

4. **Enable offline mode**:
   - Cache embeddings locally
   - Use pre-computed centroids file
   - Works without internet connection

5. **Custom NCR ID schemes**:
   - Timestamp-based: `NCR-20250517-001`
   - Type-based: `RC-001`, `CA-002`
   - Department-based: `RC-ENG-001`, `CA-MFG-002`

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

### "NCR IDs not incrementing"
- Check session state isn't being reset
- Counter should persist within a single browser session
- Clear browser cache if needed

---

## File Structure

```
.
├── app.py                 # Main Streamlit app
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── QUICKSTART.md         # Quick start guide
├── CONFIGURATION.md      # Advanced configuration
├── CHANGELOG.md          # What changed in latest version
└── DESIGN_SPEC.md        # Visual design specifications
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

## Latest Updates

✅ **Azure Blob Storage Integration** — Load training data from Azure for better confidence scores  
✅ **Auto-generated NCR IDs** — No manual entry required  
✅ **Navy blue design** — Matching original ETQ dashboard  
✅ **Simplified Root Causes form** — Removed Retain/Eliminate field  
✅ **Clean tab names** — Removed emoji icons  
✅ **Persistent record tracking** — All NCR IDs tracked and saved  
✅ **Separate exports** — Root Causes and CA in separate files  
✅ **PDF export option** — Print-to-PDF for all exports  

See `CHANGELOG.md` for full list of updates.

---

## License

Open source. Modify as needed for your organization.

---

## Support

For issues or feature requests, contact the development team.
