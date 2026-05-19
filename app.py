"""
NCR Unified Dashboard — Root Causes & Corrective Actions
=========================================================
Streamlit app with BERTopic (SBERT) trained on Azure Blob Storage data.
Descriptions → BERTopic suggestions → Save
Clean white background with red accents.
"""

import re, io, os, warnings, datetime
from zoneinfo import ZoneInfo
import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# AZURE BLOB STORAGE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

TRAINING_DATA_RC_URL = (
    "https://capstoneedward8879566973.blob.core.windows.net/"
    "azureml-blobstore-cf885e38-fe3d-4e8a-966e-"
    "9b7ad29fe76c/Root_Causes_Final_Dataset.xlsx?"
    "sp=r&st=2026-05-01T22:47:13Z&se=2027-05-"
    "01T07:02:13Z&spr=https&sv=2025-11-"
    "05&sr=b&sig=1JtAyqDXGPAbueJZBznmPnhCPqMvQF7Q7fEQvRgb%2B"
    "Q%3D"
)

TRAINING_DATA_CA_URL = (
    "https://capstoneedward8879566973.blob.core.windows.net/"
    "azureml-blobstore-cf885e38-fe3d-4e8a-966e-"
    "9b7ad29fe76c/Combined_Actions.xlsx?sp=r&st=2026-05-"
    "18T03:05:17Z&se=2027-05-18T11:20:17Z&spr=https&sv=2026-"
    "02-"
    "06&sr=b&sig=ZZLC11ETT0Mrt8erFLIr%2BT66yTurzd9dV02MeGHp8G"
    "c%3D"
)

# Categories to exclude from training and suggestions
EXCLUDED_CATEGORIES = {"Other", "Can not be determined", "Early Release"}

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & STYLING
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="NCR Classification Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Beautiful white + red theme
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600;700&family=Geist+Mono:wght@400;500;600&display=swap');

* {
    font-family: 'Geist', -apple-system, BlinkMacSystemFont, sans-serif;
}

html, body, [class*="css"] {
    background: #ffffff !important;
    color: #1f2937 !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e5e7eb !important;
}
section[data-testid="stSidebar"] * { color: #374151 !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2 { color: #1c3060 !important; font-weight: 700; }

/* ── Main container ── */
.main {
    background: #ffffff !important;
    border-top: 4px solid #dc2626 !important;
}

/* ── Tab styling ── */
[data-baseweb="tab-list"] {
    background: #f9fafb !important;
    border-bottom: 2px solid #e5e7eb !important;
    gap: 0 !important;
}
[data-baseweb="tab"] {
    color: #6b7280 !important;
    font-weight: 500 !important;
    padding: 16px 20px !important;
    border-bottom: 2px solid transparent !important;
}
[data-baseweb="tab"][aria-selected="true"] {
    color: #1c3060 !important;
    border-bottom-color: #dc2626 !important;
    background: #f0f4f8 !important;
}

/* ── Cards & containers ── */
.description-card {
    background: #f0f4f8;
    border: 1px solid #c8d4e8;
    border-top: 3px solid #dc2626;
    border-radius: 8px;
    padding: 20px;
    margin: 16px 0;
}

.suggestion-box {
    background: #1c3060;
    border: 2px solid #1c3060;
    border-left: 4px solid #dc2626;
    border-radius: 8px;
    padding: 14px 16px;
    margin: 8px 0;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
    color: #ffffff;
}
.suggestion-box:hover {
    background: #2a4a8a;
    border-color: #2a4a8a;
    transform: translateX(4px);
    box-shadow: 0 2px 8px rgba(28, 48, 96, 0.25);
}

.no-match-box {
    background: #e8edf8;
    border: 2px dashed #9ca3af;
    border-left: 4px solid #dc2626;
    border-radius: 8px;
    padding: 14px 16px;
    margin: 8px 0;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
    color: #6b7280;
}
.no-match-box:hover {
    background: #dde6f4;
    transform: translateX(4px);
}

.selection-badge {
    background: #dcfce7;
    border-left: 4px solid #16a34a;
    border-top: 2px solid #dc2626;
    border-radius: 4px;
    padding: 12px 14px;
    margin: 12px 0;
    font-weight: 500;
    color: #15803d;
}

/* ── Buttons ── */
.stButton > button {
    background: rgba(255, 255, 255, 0.9) !important;
    color: #000000 !important;
    border: 2px solid #1c3060 !important;
    font-weight: 600 !important;
    padding: 10px 24px !important;
    border-radius: 6px !important;
    transition: all 0.2s !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: #1c3060 !important;
    color: #ffffff !important;
    box-shadow: 0 4px 12px rgba(28, 48, 96, 0.3) !important;
}

/* ── Text input ── */
.stTextArea textarea {
    border: 1px solid #e5e7eb !important;
    border-top: 2px solid #dc2626 !important;
    border-radius: 6px !important;
    font-family: 'Geist', sans-serif !important;
    padding: 12px !important;
}
.stTextArea textarea:focus {
    border: 1px solid #e5e7eb !important;
    border-top: 3px solid #dc2626 !important;
    box-shadow: 0 0 0 3px rgba(220, 38, 38, 0.1) !important;
}

/* ── Dataframe ── */
.stDataFrame {
    border: 1px solid #e5e7eb !important;
    border-top: 3px solid #dc2626 !important;
    border-radius: 8px !important;
}

/* ── Headers ── */
h1 { color: #1f2937 !important; font-weight: 700; letter-spacing: -0.02em; }
h2 { color: #1f2937 !important; font-weight: 600; }
h3 { color: #374151 !important; font-weight: 600; }

/* ── Confidence colors ── */
.conf-high { color: #16a34a; font-weight: 700; font-family: 'Geist Mono', monospace; }
.conf-mid  { color: #d97706; font-weight: 700; font-family: 'Geist Mono', monospace; }
.conf-low  { color: #dc2626; font-weight: 700; font-family: 'Geist Mono', monospace; }

/* ── Saved count badge ── */
.saved-badge {
    display: inline-block;
    background: #dc2626;
    color: white;
    padding: 4px 12px;
    border-radius: 12px;
    font-size: 0.85rem;
    font-weight: 600;
    margin-left: 8px;
}

/* ── Selection boxes ── */
.sel-box {
    border-radius: 5px;
    padding: 12px 14px;
    font-weight: 600;
    font-size: 0.9rem;
    margin: 12px 0;
    border-left: 4px solid;
    background: rgba(255, 255, 255, 0.8);
    color: #000000 !important;
}
.sel-match { border-color: #1a8040; }
.sel-nomatch { border-color: #c62828; }

hr { border: 1px solid #e5e7eb !important; }

.stInfo { background: #f0fdf4 !important; border-left: 4px solid #16a34a !important; }
.stWarning { background: #fef3c7 !important; border-left: 4px solid #d97706 !important; }
.stError { background: #fee2e2 !important; border-left: 4px solid #dc2626 !important; }
.stSuccess { background: #f0fdf4 !important; border-left: 4px solid #16a34a !important; }

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE & INITIALIZATION
# ─────────────────────────────────────────────────────────────────────────────

if "records" not in st.session_state:
    st.session_state.records = []

if "ncr_counter" not in st.session_state:
    st.session_state.ncr_counter = 1

if "sbert_model" not in st.session_state:
    st.session_state.sbert_model = None

if "rc_categories" not in st.session_state:
    st.session_state.rc_categories = None
if "rc_centroids" not in st.session_state:
    st.session_state.rc_centroids = None

if "ca_categories" not in st.session_state:
    st.session_state.ca_categories = None
if "ca_centroids" not in st.session_state:
    st.session_state.ca_centroids = None

if "rc_selected_category" not in st.session_state:
    st.session_state.rc_selected_category = None

if "ca_selected_category" not in st.session_state:
    st.session_state.ca_selected_category = None

if "rc_data_loaded" not in st.session_state:
    st.session_state.rc_data_loaded = False

if "ca_data_loaded" not in st.session_state:
    st.session_state.ca_data_loaded = False

def generate_ncr_id():
    """Generate next NCR ID automatically."""
    ncr_id = f"NCR-{st.session_state.ncr_counter:05d}"
    st.session_state.ncr_counter += 1
    return ncr_id

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_sbert_model():
    """Load SBERT model once and cache it."""
    return SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

def clean_text(text):
    """Basic text cleaning."""
    text = str(text).lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def load_training_data_rc():
    """Load Root Causes training data from Azure Blob Storage."""
    try:
        with st.spinner("Loading Root Causes training data..."):
            df = pd.read_excel(TRAINING_DATA_RC_URL)
            
            # Filter out excluded categories
            df = df[~df['Root Cause Category_Final'].isin(EXCLUDED_CATEGORIES)].copy()
            
            # Limit to 2000 descriptions
            if len(df) > 2000:
                df = df.sample(n=2000, random_state=42)
            
            # Clean descriptions
            df['Description'] = df['Description'].apply(clean_text)
            
            # Remove empty descriptions
            df = df[df['Description'].str.len() > 0]
            
            return df
    except Exception as e:
        st.error(f"Error loading Root Causes data: {e}")
        return None

def load_training_data_ca():
    """Load Corrective Actions training data from Azure Blob Storage."""
    try:
        with st.spinner("Loading Corrective Actions training data..."):
            df = pd.read_excel(TRAINING_DATA_CA_URL)
            
            # Filter out excluded categories
            df = df[~df['Corrective Action Type_Final'].isin(EXCLUDED_CATEGORIES)].copy()
            
            # Limit to 2000 descriptions
            if len(df) > 2000:
                df = df.sample(n=2000, random_state=42)
            
            # Clean descriptions
            df['Description of Action'] = df['Description of Action'].apply(clean_text)
            
            # Remove empty descriptions
            df = df[df['Description of Action'].str.len() > 0]
            
            return df
    except Exception as e:
        st.error(f"Error loading Corrective Actions data: {e}")
        return None

def compute_centroids_from_data(descriptions, categories):
    """Compute SBERT embedding centroids for each unique category from data."""
    model = load_sbert_model()
    centroids = {}
    unique_categories = set(categories)
    
    with st.spinner("Computing category centroids..."):
        for cat in unique_categories:
            # Get all descriptions matching this category
            cat_texts = [d for d, c in zip(descriptions, categories) if c == cat]
            if cat_texts:
                embeddings = model.encode(cat_texts, show_progress_bar=False)
                centroids[cat] = embeddings.mean(axis=0)
            else:
                # Fallback: encode the category name itself
                centroids[cat] = model.encode([cat], show_progress_bar=False)[0]
    
    return centroids, list(unique_categories)

def get_top_suggestions(user_text, centroids, categories, top_n=3):
    """
    Get top N category suggestions using SBERT cosine similarity.
    Returns (category, confidence) tuples.
    """
    model = load_sbert_model()
    user_embedding = model.encode([user_text], show_progress_bar=False)[0]
    
    scores = []
    for cat in categories:
        if cat in centroids:
            similarity = cosine_similarity([user_embedding], [centroids[cat]])[0][0]
            scores.append((cat, float(similarity)))
    
    # Sort by confidence descending
    scores.sort(key=lambda x: x[1], reverse=True)
    
    # Return top N
    suggestions = scores[:top_n]
    
    return suggestions

# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP LAYOUT
# ─────────────────────────────────────────────────────────────────────────────

# Header
st.markdown("""
<div style="text-align: center; margin: 30px 0 40px 0; border-top: 4px solid #dc2626; padding-top: 30px;">
    <h1 style="margin: 0; font-size: 2.2rem;">🔍 NCR Classification</h1>
    <p style="color: #6b7280; margin: 8px 0 0 0; font-size: 1rem;">Root Causes & Corrective Actions</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["Root Causes", "Corrective Actions", "Saved Records"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ROOT CAUSES
# ═══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.markdown("### Classify a Root Cause")
    st.markdown("Enter a free-text root cause description. BERTopic will suggest the 3 best matching categories from training data.")
    
    # Initialize centroids from Azure data if needed
    if st.session_state.rc_categories is None:
        rc_df = load_training_data_rc()
        if rc_df is not None and len(rc_df) > 0:
            st.session_state.rc_centroids, st.session_state.rc_categories = compute_centroids_from_data(
                rc_df['Description'].tolist(),
                rc_df['Root Cause Category_Final'].tolist()
            )
            st.session_state.rc_data_loaded = True
            st.success(f"✓ Trained on {len(rc_df)} Root Cause descriptions with {len(st.session_state.rc_categories)} categories")
        else:
            st.warning("Unable to load Root Causes training data. Please check your data source.")
    
    # Input section
    rc_description = st.text_area(
        "Root Cause Description",
        placeholder="e.g., Operator assembled component incorrectly due to unclear work instruction...",
        height=100,
        label_visibility="collapsed"
    )
    
    # Classify & Analyze button (visual indicator)
    col_btn, col_space = st.columns([1, 4])
    with col_btn:
        st.button("🔍 Classify & Analyze", key="rc_classify_btn", use_container_width=True, type="primary", disabled=not rc_description.strip())
    
    if rc_description.strip() and st.session_state.rc_categories is not None:
        # Get suggestions
        suggestions = get_top_suggestions(
            rc_description,
            st.session_state.rc_centroids,
            st.session_state.rc_categories,
            top_n=3
        )
        
        st.markdown(
            "<p style='font-size:0.85rem; font-weight:600; color:#1c3060; margin: 16px 0 12px 0;'>"
            "Root Cause Category — select one:</p>",
            unsafe_allow_html=True
        )
        
        # Initialize session state for RC selection if needed
        if "rc_manual_category" not in st.session_state:
            st.session_state.rc_manual_category = None
        
        # Display suggestions as clickable buttons
        for idx, (cat, conf) in enumerate(suggestions):
            conf_pct = conf * 100
            conf_class = "conf-high" if conf_pct >= 80 else ("conf-mid" if conf_pct >= 60 else "conf-low")
            
            col_btn, col_conf = st.columns([4, 1])
            with col_btn:
                if st.button(cat, key=f"rc_cat_{idx}", use_container_width=True):
                    st.session_state.rc_selected_category = cat
                    st.session_state.rc_manual_category = cat
            with col_conf:
                st.markdown(f"<div class='{conf_class}'>{conf_pct:.1f}%</div>", unsafe_allow_html=True)
        
        # Custom category input
        custom_cat = st.text_input("Or type a different category:", value=st.session_state.rc_manual_category or "")
        if custom_cat.strip():
            st.session_state.rc_selected_category = custom_cat.strip()
        
        st.markdown("---")
        
        # Save button
        if st.session_state.rc_selected_category:
            st.markdown(
                f"<div class='selection-badge'>✓ Selected: {st.session_state.rc_selected_category}</div>",
                unsafe_allow_html=True
            )
            
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                if st.button("✅ Save Record", use_container_width=True):
                    record = {
                        "DateTime": datetime.datetime.now(ZoneInfo("US/Pacific")),
                        "Type": "Root Cause",
                        "NCR Number": generate_ncr_id(),
                        "Category": st.session_state.rc_selected_category,
                        "Description": rc_description
                    }
                    st.session_state.records.append(record)
                    st.success("✓ Root Cause record saved!")
                    st.session_state.rc_selected_category = None
                    st.session_state.rc_manual_category = None
                    st.rerun()
            
            with col2:
                if st.button("🔄 Clear", use_container_width=True):
                    st.session_state.rc_selected_category = None
                    st.session_state.rc_manual_category = None
                    st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CORRECTIVE ACTIONS
# ═══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown("### Classify a Corrective Action")
    st.markdown("Enter a free-text corrective action description. BERTopic will suggest the 3 best matching categories from training data.")
    
    # Initialize centroids from Azure data if needed
    if st.session_state.ca_categories is None:
        ca_df = load_training_data_ca()
        if ca_df is not None and len(ca_df) > 0:
            st.session_state.ca_centroids, st.session_state.ca_categories = compute_centroids_from_data(
                ca_df['Description of Action'].tolist(),
                ca_df['Corrective Action Type_Final'].tolist()
            )
            st.session_state.ca_data_loaded = True
            st.success(f"✓ Trained on {len(ca_df)} Corrective Action descriptions with {len(st.session_state.ca_categories)} categories")
        else:
            st.warning("Unable to load Corrective Actions training data. Please check your data source.")
    
    # Input section
    ca_description = st.text_area(
        "Corrective Action Description",
        placeholder="e.g., Revised work instructions to include step-by-step assembly guidance with visual aids...",
        height=100,
        label_visibility="collapsed"
    )
    
    # Classify & Analyze button
    col_btn, col_space = st.columns([1, 4])
    with col_btn:
        st.button("🔍 Classify & Analyze", key="ca_classify_btn", use_container_width=True, type="primary", disabled=not ca_description.strip())
    
    if ca_description.strip() and st.session_state.ca_categories is not None:
        # Get suggestions
        suggestions = get_top_suggestions(
            ca_description,
            st.session_state.ca_centroids,
            st.session_state.ca_categories,
            top_n=3
        )
        
        st.markdown(
            "<p style='font-size:0.85rem; font-weight:600; color:#1c3060; margin: 16px 0 12px 0;'>"
            "Corrective Action Category — select one:</p>",
            unsafe_allow_html=True
        )
        
        # Initialize session state for CA selection if needed
        if "ca_manual_category" not in st.session_state:
            st.session_state.ca_manual_category = None
        
        # Display suggestions as clickable buttons
        for idx, (cat, conf) in enumerate(suggestions):
            conf_pct = conf * 100
            conf_class = "conf-high" if conf_pct >= 80 else ("conf-mid" if conf_pct >= 60 else "conf-low")
            
            col_btn, col_conf = st.columns([4, 1])
            with col_btn:
                if st.button(cat, key=f"ca_cat_{idx}", use_container_width=True):
                    st.session_state.ca_selected_category = cat
                    st.session_state.ca_manual_category = cat
            with col_conf:
                st.markdown(f"<div class='{conf_class}'>{conf_pct:.1f}%</div>", unsafe_allow_html=True)
        
        # Custom category input
        custom_cat = st.text_input("Or type a different category:", value=st.session_state.ca_manual_category or "", key="ca_custom")
        if custom_cat.strip():
            st.session_state.ca_selected_category = custom_cat.strip()
        
        st.markdown("---")
        
        # Save button
        if st.session_state.ca_selected_category:
            st.markdown(
                f"<div class='selection-badge'>✓ Selected: {st.session_state.ca_selected_category}</div>",
                unsafe_allow_html=True
            )
            
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                if st.button("✅ Save Record", use_container_width=True, key="ca_save"):
                    record = {
                        "DateTime": datetime.datetime.now(ZoneInfo("US/Pacific")),
                        "Type": "Corrective Action",
                        "NCR Number": generate_ncr_id(),
                        "Category": st.session_state.ca_selected_category,
                        "Description": ca_description
                    }
                    st.session_state.records.append(record)
                    st.success("✓ Corrective Action record saved!")
                    st.session_state.ca_selected_category = None
                    st.session_state.ca_manual_category = None
                    st.rerun()
            
            with col2:
                if st.button("🔄 Clear", use_container_width=True, key="ca_clear"):
                    st.session_state.ca_selected_category = None
                    st.session_state.ca_manual_category = None
                    st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SAVED RECORDS
# ═══════════════════════════════════════════════════════════════════════════════

with tab3:
    if len(st.session_state.records) == 0:
        st.info("No records saved yet. Start by classifying Root Causes or Corrective Actions.")
    else:
        st.markdown(f"### Saved Records <span class='saved-badge'>{len(st.session_state.records)}</span>", unsafe_allow_html=True)
        
        # Convert records to DataFrame
        df = pd.DataFrame(st.session_state.records)
        
        st.markdown("---")
        
        # Filters
        st.markdown("#### Filters")
        cols = st.columns(2)
        with cols[0]:
            type_f = st.multiselect("Record Type", sorted(df["Type"].unique()), default=None)
        with cols[1]:
            ncr_f = st.multiselect("NCR Number", sorted(df["NCR Number"].unique()), default=None)
        
        # Apply filters
        fdf = df.copy()
        if type_f:
            fdf = fdf[fdf["Type"].isin(type_f)]
        if ncr_f:
            fdf = fdf[fdf["NCR Number"].isin(ncr_f)]
        
        st.markdown(f"**Showing {len(fdf)} of {len(df)} records**")
        
        # Display records
        fdf_display = fdf.copy()
        fdf_display = fdf_display.rename(columns={"Type": "Record Type"})
        display_cols = ["DateTime", "Record Type", "NCR Number", "Category", "Description"]
        st.dataframe(fdf_display[display_cols], use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.markdown("#### Export")
        
        # Separate data by type
        rc_data = fdf[fdf["Type"] == "Root Cause"].copy()
        ca_data = fdf[fdf["Type"] == "Corrective Action"].copy()
        
        # Export options - separate for RC and CA
        st.markdown("**Root Causes Export**")
        rc_cols = st.columns(3)
        
        with rc_cols[0]:
            if len(rc_data) > 0:
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine="openpyxl") as w:
                    rc_export_cols = ["DateTime", "NCR Number", "Category", "Description"]
                    rc_export = rc_data[rc_export_cols].copy()
                    rc_export["DateTime"] = rc_export["DateTime"].dt.strftime("%Y-%m-%d %H:%M:%S")
                    rc_export.to_excel(w, sheet_name="Root Causes", index=False)
                    # RC summary
                    rc_summary = rc_data["Category"].value_counts().reset_index()
                    rc_summary.columns = ["Category", "Count"]
                    rc_summary.to_excel(w, sheet_name="Summary", index=False)
                
                st.download_button(
                    "📥 Root Causes Excel",
                    data=buf.getvalue(),
                    file_name=f"Root_Causes_{datetime.date.today()}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            else:
                st.info("No Root Causes to export")
        
        with rc_cols[1]:
            if len(rc_data) > 0:
                rc_csv = rc_data[["DateTime", "NCR Number", "Category", "Description"]].to_csv(index=False)
                st.download_button(
                    "📥 Root Causes CSV",
                    data=rc_csv.encode("utf-8"),
                    file_name=f"Root_Causes_{datetime.date.today()}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.info("No Root Causes to export")
        
        with rc_cols[2]:
            if len(rc_data) > 0:
                # Generate PDF for Root Causes
                pdf_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Root Causes Report — {datetime.date.today()}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            color: #1f2937;
        }}
        h1 {{
            color: #1c3060;
            border-bottom: 2px solid #1c3060;
            padding-bottom: 10px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        th {{
            background: #1c3060;
            color: white;
            padding: 10px;
            text-align: left;
            font-weight: bold;
        }}
        td {{
            padding: 10px;
            border-bottom: 1px solid #e5e7eb;
        }}
        tr:nth-child(even) {{
            background: #f9fafb;
        }}
    </style>
</head>
<body>
    <h1>Root Causes Report</h1>
    <p><strong>Generated:</strong> {datetime.date.today()}</p>
    {rc_data[["DateTime", "NCR Number", "Category", "Description"]].to_html(index=False, border=0)}
</body>
</html>"""
                st.download_button(
                    "📥 Root Causes PDF",
                    data=pdf_html.encode("utf-8"),
                    file_name=f"Root_Causes_{datetime.date.today()}.html",
                    mime="text/html",
                    use_container_width=True,
                    help="Open in browser → Ctrl+P → Save as PDF"
                )
            else:
                st.info("No Root Causes to export")
        
        st.markdown("---")
        st.markdown("**Corrective Actions Export**")
        ca_cols = st.columns(3)
        
        with ca_cols[0]:
            if len(ca_data) > 0:
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine="openpyxl") as w:
                    ca_export_cols = ["DateTime", "NCR Number", "Category", "Description"]
                    ca_export = ca_data[ca_export_cols].copy()
                    ca_export["DateTime"] = ca_export["DateTime"].dt.strftime("%Y-%m-%d %H:%M:%S")
                    ca_export.to_excel(w, sheet_name="Corrective Actions", index=False)
                    # CA summary
                    ca_summary = ca_data["Category"].value_counts().reset_index()
                    ca_summary.columns = ["Category", "Count"]
                    ca_summary.to_excel(w, sheet_name="Summary", index=False)
                
                st.download_button(
                    "📥 Corrective Actions Excel",
                    data=buf.getvalue(),
                    file_name=f"Corrective_Actions_{datetime.date.today()}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            else:
                st.info("No Corrective Actions to export")
        
        with ca_cols[1]:
            if len(ca_data) > 0:
                ca_csv = ca_data[["DateTime", "NCR Number", "Category", "Description"]].to_csv(index=False)
                st.download_button(
                    "📥 Corrective Actions CSV",
                    data=ca_csv.encode("utf-8"),
                    file_name=f"Corrective_Actions_{datetime.date.today()}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.info("No Corrective Actions to export")
        
        with ca_cols[2]:
            if len(ca_data) > 0:
                # Generate PDF for Corrective Actions
                pdf_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Corrective Actions Report — {datetime.date.today()}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            color: #1f2937;
        }}
        h1 {{
            color: #1c3060;
            border-bottom: 2px solid #1c3060;
            padding-bottom: 10px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        th {{
            background: #1c3060;
            color: white;
            padding: 10px;
            text-align: left;
            font-weight: bold;
        }}
        td {{
            padding: 10px;
            border-bottom: 1px solid #e5e7eb;
        }}
        tr:nth-child(even) {{
            background: #f9fafb;
        }}
    </style>
</head>
<body>
    <h1>Corrective Actions Report</h1>
    <p><strong>Generated:</strong> {datetime.date.today()}</p>
    {ca_data[["DateTime", "NCR Number", "Category", "Description"]].to_html(index=False, border=0)}
</body>
</html>"""
                st.download_button(
                    "📥 Corrective Actions PDF",
                    data=pdf_html.encode("utf-8"),
                    file_name=f"Corrective_Actions_{datetime.date.today()}.html",
                    mime="text/html",
                    use_container_width=True,
                    help="Open in browser → Ctrl+P → Save as PDF"
                )
            else:
                st.info("No Corrective Actions to export")
        
        st.markdown("---")
        
        # Clear data
        if st.button("🗑️ Clear All Records", use_container_width=True):
            st.session_state.records = []
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #9ca3af; font-size: 0.85rem; margin-top: 30px;">
    <p>NCR Classification Dashboard • Powered by BERTopic & SBERT</p>
</div>
""", unsafe_allow_html=True)
