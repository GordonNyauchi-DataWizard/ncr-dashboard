"""
NCR Unified Dashboard — Root Causes & Corrective Actions
=========================================================
Streamlit app with BERTopic (SBERT) for both Root Causes and Corrective Actions.
Simplified UI: Description → BERTopic suggestions → Save
Clean white background with red accents.
"""

import re, io, os, warnings, datetime
import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

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
    border-bottom: 3px solid transparent !important;
}
[data-baseweb="tab"][aria-selected="true"] {
    color: #1c3060 !important;
    border-bottom-color: #1c3060 !important;
    background: #f0f4f8 !important;
}

/* ── Cards & containers ── */
.description-card {
    background: #f0f4f8;
    border: 1px solid #c8d4e8;
    border-radius: 8px;
    padding: 20px;
    margin: 16px 0;
}

.suggestion-box {
    background: #1c3060;
    border: 2px solid #1c3060;
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
    border-radius: 6px !important;
    font-family: 'Geist', sans-serif !important;
    padding: 12px !important;
}
.stTextArea textarea:focus {
    border: 2px solid #1c3060 !important;
    box-shadow: 0 0 0 3px rgba(28, 48, 96, 0.1) !important;
}

/* ── Dataframe ── */
.stDataFrame {
    border: 1px solid #e5e7eb !important;
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

def generate_ncr_id():
    """Generate next NCR ID automatically."""
    ncr_id = f"NCR-{st.session_state.ncr_counter:05d}"
    st.session_state.ncr_counter += 1
    return ncr_id

# ─────────────────────────────────────────────────────────────────────────────
# ROOT CAUSE CATEGORIES (58 categories from original app)
# ─────────────────────────────────────────────────────────────────────────────

ROOT_CAUSE_CATEGORIES = [
    "Documentation (Procedures, forms, drawings, instructions, protocols, ECRs)",
    "Documentation : Ambiguous/Unclear",
    "Documentation : Inadequate Instructions",
    "Documentation : Inadequate Standardization/Conflicting Documents",
    "Documentation : Lack of procedure or instruction",
    "Documentation : Lack of sufficient details",
    "Equipment",
    "Equipment : Calibration issues",
    "Equipment : Design and/or installation issues",
    "Equipment : Equipment Maintenance issues",
    "Equipment : Equipment Out of Specification",
    "Equipment : Equipment Utilization issues",
    "Equipment : Poorly Designed Interfaces",
    "Human Factor : Habits and Routine",
    "Human Factor : Misguided Practice",
    "Human Factor : Multitasking/Interruptions",
    "Human Factor : Work/Mental Load",
    "Human Factors",
    "Human Factors : Operator Error",
    "Labeling/Instructions for Use",
    "Labeling/Instructions for Use : Inadequate instructions for use",
    "Labeling/Instructions for Use : Inadequate label",
    "Labeling/Instructions for Use : Incorrect label",
    "Labeling/Instructions for Use : Translation error",
    "Materials (Incoming/ In Process)",
    "Materials : Material Controls issues",
    "Materials : Purchasing control issues",
    "Materials : Supplier Issues",
    "Planned NCR - NCR Module Only",
    "Process",
    "Process : Inadequate process control",
    "Process : Inadequate verification or validation",
    "Process : Lack of mistake proofing",
    "Process : Sterilization failure",
    "Product (Including software)",
    "Product : Inadequate or incorrect acceptance criteria",
    "Product : Inadequate product design/human factors",
    "Product : Shelf life and distribution issues",
    "Product : Software bug or inadequate design",
    "Supplier Issues",
    "System",
    "System : Complex or multiple changes",
    "System : Configuration issues",
    "System : IT Infrastructure/Software",
    "System : Inadequate interface design",
    "System : Inadequate Network Infrastructure",
    "System : Inadequate software functionality",
    "System : Network issues",
    "System : System failure",
    "System : Training and change control",
    "Training",
    "Training : Inadequate training",
    "Training : Inadequate training, unclear documentation",
    "Training : Inadequate understanding of procedure",
    "Training : Initial inadequate training",
    "Training : Lack of Training",
    "Training : Refresher training needed",
    "Training : Unclear documentation",
]

# ─────────────────────────────────────────────────────────────────────────────
# CORRECTIVE ACTION CATEGORIES (example set - adjust to match your data)
# ─────────────────────────────────────────────────────────────────────────────

CORRECTIVE_ACTION_CATEGORIES = [
    "Design Modification",
    "Documentation Update",
    "Equipment Maintenance",
    "Equipment Replacement",
    "Process Change",
    "Procedure Update",
    "Supplier Change",
    "Training Program",
    "Validation Study",
]

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

def compute_centroids(categories, descriptions):
    """Compute SBERT embedding centroids for each category."""
    model = load_sbert_model()
    centroids = {}
    
    for cat in categories:
        # Get all descriptions matching this category
        cat_texts = [d for d, c in zip(descriptions, categories) if c == cat]
        if cat_texts:
            embeddings = model.encode(cat_texts, show_progress_bar=False)
            centroids[cat] = embeddings.mean(axis=0)
        else:
            # Fallback: encode the category name itself
            centroids[cat] = model.encode([cat], show_progress_bar=False)[0]
    
    return centroids

def get_top_suggestions(user_text, centroids, categories, top_n=3):
    """
    Get top N category suggestions using SBERT cosine similarity.
    Returns (category, confidence) tuples (no filtering).
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
    
    # Return top N (no threshold filter)
    suggestions = scores[:top_n]
    
    return suggestions

# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP LAYOUT
# ─────────────────────────────────────────────────────────────────────────────

# Header
st.markdown("""
<div style="text-align: center; margin: 30px 0 40px 0;">
    <h1 style="margin: 0; font-size: 2.2rem;">🔍 NCR Classification</h1>
    <p style="color: #6b7280; margin: 8px 0 0 0; font-size: 1rem;">Root Causes & Corrective Actions</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["Root Causes", "Corrective Actions", "Records Browser"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ROOT CAUSES
# ═══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.markdown("### Classify a Root Cause")
    st.markdown("Enter a free-text root cause description. BERTopic will suggest the 3 best matching categories.")
    
    # Initialize centroids if needed
    if st.session_state.rc_categories is None:
        # Load sample data or use default categories
        st.session_state.rc_categories = ROOT_CAUSE_CATEGORIES
        # Generate dummy descriptions for centroid computation
        dummy_rc_descriptions = ROOT_CAUSE_CATEGORIES
        st.session_state.rc_centroids = compute_centroids(
            st.session_state.rc_categories,
            dummy_rc_descriptions
        )
    
    # Input section
    st.markdown('<div class="description-card">', unsafe_allow_html=True)
    rc_description = st.text_area(
        "Root Cause Description",
        placeholder="e.g., Operator assembled component incorrectly due to unclear work instruction...",
        height=100,
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    if rc_description.strip():
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
        if "rc_selected_category" not in st.session_state:
            st.session_state.rc_selected_category = None
        
        # 4-button layout: 3 suggestions + No Match
        btn_cols = st.columns(4)
        
        if suggestions:
            for i, (cat, conf) in enumerate(suggestions):
                # Confidence icon - aligned with 30% threshold
                icon = "🟢" if conf >= 0.50 else ("🟡" if conf >= 0.30 else "🔴")
                
                # Truncate category name if too long
                short = cat[:40] + "…" if len(cat) > 40 else cat
                
                # Button label with selection indicator
                is_selected = st.session_state.rc_selected_category == cat
                label = f"{'✓ ' if is_selected else ''}{icon} {short}\n[conf: {conf:.3f}]"
                
                with btn_cols[i]:
                    if st.button(
                        label,
                        key=f"rc_sug_{i}",
                        use_container_width=True,
                        type="primary" if is_selected else "secondary"
                    ):
                        st.session_state.rc_selected_category = cat
                        st.rerun()
        
        # No Match button
        is_nomatch = st.session_state.rc_selected_category == "No Match"
        with btn_cols[3]:
            if st.button(
                f"{'✓ ' if is_nomatch else ''}❌ No Match",
                key="rc_nomatch_btn",
                use_container_width=True,
                type="primary" if is_nomatch else "secondary"
            ):
                st.session_state.rc_selected_category = "No Match"
                st.rerun()
        
        # Show full selected category name
        if st.session_state.rc_selected_category:
            if st.session_state.rc_selected_category == "No Match":
                st.markdown(
                    '<div class="sel-box sel-nomatch">❌ No Match — root cause could not be categorised</div>',
                    unsafe_allow_html=True
                )
            else:
                # Get confidence for selected category
                sel_conf = next(
                    (conf for cat, conf in suggestions if cat == st.session_state.rc_selected_category),
                    None
                )
                conf_str = f" &nbsp;|&nbsp; Confidence: {sel_conf:.3f}" if sel_conf else ""
                st.markdown(
                    f'<div class="sel-box sel-match">✅ {st.session_state.rc_selected_category}{conf_str}</div>',
                    unsafe_allow_html=True
                )
            
            # Save and Clear buttons
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("---")
            
            save_col, clear_col, spacer_col = st.columns([1, 1, 3])
            
            with save_col:
                save_rc = st.button("💾 Save Record", type="primary", use_container_width=True)
            
            with clear_col:
                clear_rc = st.button("🗑️ Clear", use_container_width=True)
            
            if clear_rc:
                st.session_state.rc_selected_category = None
                st.rerun()
            
            if save_rc:
                status = st.selectbox("Status", ["Open", "Closed", "In Progress"], key="rc_status")
                ncr_id = generate_ncr_id()
                sel_conf = next(
                    (conf for cat, conf in suggestions if cat == st.session_state.rc_selected_category),
                    None
                ) if st.session_state.rc_selected_category != "No Match" else 0.0
                
                record = {
                    "DateTime": datetime.datetime.now(),
                    "Type": "Root Cause",
                    "NCR Number": ncr_id,
                    "Description": rc_description,
                    "Category": st.session_state.rc_selected_category,
                    "Confidence": sel_conf if sel_conf else 0.0,
                    "Status": status,
                }
                st.session_state.records.append(record)
                st.success(f"✅ Record saved for **{ncr_id}** — Category: **{st.session_state.rc_selected_category}**")
                st.session_state.rc_selected_category = None
                st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CORRECTIVE ACTIONS
# ═══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown("### Classify a Corrective Action")
    st.markdown("Enter a free-text corrective action description. BERTopic will suggest the 3 best matching categories.")
    
    # Initialize centroids if needed
    if st.session_state.ca_categories is None:
        st.session_state.ca_categories = CORRECTIVE_ACTION_CATEGORIES
        dummy_ca_descriptions = CORRECTIVE_ACTION_CATEGORIES
        st.session_state.ca_centroids = compute_centroids(
            st.session_state.ca_categories,
            dummy_ca_descriptions
        )
    
    # Input section
    st.markdown('<div class="description-card">', unsafe_allow_html=True)
    ca_description = st.text_area(
        "Corrective Action Description",
        placeholder="e.g., Revise work instruction and provide retraining to all operators...",
        height=100,
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    if ca_description.strip():
        # Get suggestions
        suggestions = get_top_suggestions(
            ca_description,
            st.session_state.ca_centroids,
            st.session_state.ca_categories,
            top_n=3
        )
        
        st.markdown(
            "<p style='font-size:0.85rem; font-weight:600; color:#1c3060; margin: 16px 0 12px 0;'>"
            "Corrective Action Type — select one:</p>",
            unsafe_allow_html=True
        )
        
        # Initialize session state for CA selection if needed
        if "ca_selected_category" not in st.session_state:
            st.session_state.ca_selected_category = None
        
        # 4-button layout: 3 suggestions + No Match
        btn_cols = st.columns(4)
        
        if suggestions:
            for i, (cat, conf) in enumerate(suggestions):
                # Confidence icon - aligned with 30% threshold
                icon = "🟢" if conf >= 0.50 else ("🟡" if conf >= 0.30 else "🔴")
                
                # Truncate category name if too long
                short = cat[:40] + "…" if len(cat) > 40 else cat
                
                # Button label with selection indicator
                is_selected = st.session_state.ca_selected_category == cat
                label = f"{'✓ ' if is_selected else ''}{icon} {short}\n[conf: {conf:.3f}]"
                
                with btn_cols[i]:
                    if st.button(
                        label,
                        key=f"ca_sug_{i}",
                        use_container_width=True,
                        type="primary" if is_selected else "secondary"
                    ):
                        st.session_state.ca_selected_category = cat
                        st.rerun()
        
        # No Match button
        is_nomatch = st.session_state.ca_selected_category == "No Match"
        with btn_cols[3]:
            if st.button(
                f"{'✓ ' if is_nomatch else ''}❌ No Match",
                key="ca_nomatch_btn",
                use_container_width=True,
                type="primary" if is_nomatch else "secondary"
            ):
                st.session_state.ca_selected_category = "No Match"
                st.rerun()
        
        # Show full selected category name
        if st.session_state.ca_selected_category:
            if st.session_state.ca_selected_category == "No Match":
                st.markdown(
                    '<div class="sel-box sel-nomatch">❌ No Match — corrective action could not be categorised</div>',
                    unsafe_allow_html=True
                )
            else:
                # Get confidence for selected category
                sel_conf = next(
                    (conf for cat, conf in suggestions if cat == st.session_state.ca_selected_category),
                    None
                )
                conf_str = f" &nbsp;|&nbsp; Confidence: {sel_conf:.3f}" if sel_conf else ""
                st.markdown(
                    f'<div class="sel-box sel-match">✅ {st.session_state.ca_selected_category}{conf_str}</div>',
                    unsafe_allow_html=True
                )
            
            # Save and Clear buttons
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("---")
            
            save_col, clear_col, spacer_col = st.columns([1, 1, 3])
            
            with save_col:
                save_ca = st.button("💾 Save Record", type="primary", use_container_width=True, key="save_ca")
            
            with clear_col:
                clear_ca = st.button("🗑️ Clear", use_container_width=True, key="clear_ca")
            
            if clear_ca:
                st.session_state.ca_selected_category = None
                st.rerun()
            
            if save_ca:
                status = st.selectbox("Status", ["Open", "Closed", "In Progress"], key="ca_status")
                effective = st.radio(
                    "Effectiveness",
                    ["Effective", "Ineffective", "Pending Review"],
                    horizontal=True,
                    key="ca_effective"
                )
                
                ncr_id = generate_ncr_id()
                sel_conf = next(
                    (conf for cat, conf in suggestions if cat == st.session_state.ca_selected_category),
                    None
                ) if st.session_state.ca_selected_category != "No Match" else 0.0
                
                record = {
                    "DateTime": datetime.datetime.now(),
                    "Type": "Corrective Action",
                    "NCR Number": ncr_id,
                    "Description": ca_description,
                    "Category": st.session_state.ca_selected_category,
                    "Confidence": sel_conf if sel_conf else 0.0,
                    "Status": status,
                    "Effectiveness": effective,
                }
                st.session_state.records.append(record)
                st.success(f"✅ Record saved for **{ncr_id}** — Category: **{st.session_state.ca_selected_category}**")
                st.session_state.ca_selected_category = None
                st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — RECORDS BROWSER & EXPORT
# ═══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown("### 📁 Saved Records")
    
    if not st.session_state.records:
        st.info("No records saved yet. Add root causes or corrective actions to get started.")
    else:
        df = pd.DataFrame(st.session_state.records)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Root Causes", len(df[df["Type"] == "Root Cause"]))
        with col3:
            st.metric("Corrective Actions", len(df[df["Type"] == "Corrective Action"]))
        with col4:
            st.metric("Unique NCRs", df["NCR Number"].nunique())
        
        st.markdown("---")
        
        # Filters
        st.markdown("#### Filters")
        cols = st.columns(3)
        with cols[0]:
            type_f = st.multiselect("Record Type", sorted(df["Type"].unique()), default=None)
        with cols[1]:
            ncr_f = st.multiselect("NCR Number", sorted(df["NCR Number"].unique()), default=None)
        with cols[2]:
            status_f = st.multiselect("Status", sorted(df["Status"].unique()), default=None)
        
        # Apply filters
        fdf = df.copy()
        if type_f:
            fdf = fdf[fdf["Type"].isin(type_f)]
        if ncr_f:
            fdf = fdf[fdf["NCR Number"].isin(ncr_f)]
        if status_f:
            fdf = fdf[fdf["Status"].isin(status_f)]
        
        st.markdown(f"**Showing {len(fdf)} of {len(df)} records**")
        
        # Display records
        display_cols = ["DateTime", "Type", "NCR Number", "Category", "Description", "Status"]
        st.dataframe(fdf[display_cols], use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.markdown("#### Export")
        
        # Separate data by type
        rc_data = fdf[fdf["Type"] == "Root Cause"].copy()
        ca_data = fdf[fdf["Type"] == "Corrective Action"].copy()
        
        # Export options
        cols = st.columns(3)
        
        with cols[0]:
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as w:
                # Root Causes sheet
                if len(rc_data) > 0:
                    rc_export_cols = ["DateTime", "NCR Number", "Category", "Description", "Status"]
                    rc_data[rc_export_cols].to_excel(w, sheet_name="Root Causes", index=False)
                    # RC summary
                    rc_summary = rc_data["Category"].value_counts().reset_index()
                    rc_summary.columns = ["Category", "Count"]
                    rc_summary.to_excel(w, sheet_name="RC Summary", index=False)
                
                # Corrective Actions sheet
                if len(ca_data) > 0:
                    ca_export_cols = ["DateTime", "NCR Number", "Category", "Description", "Status", "Effectiveness"]
                    ca_data[ca_export_cols].to_excel(w, sheet_name="Corrective Actions", index=False)
                    # CA summary
                    ca_summary = ca_data["Category"].value_counts().reset_index()
                    ca_summary.columns = ["Category", "Count"]
                    ca_summary.to_excel(w, sheet_name="CA Summary", index=False)
            
            st.download_button(
                "📥 Excel (Root Causes & CA)",
                data=buf.getvalue(),
                file_name=f"NCR_Records_{datetime.date.today()}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        
        with cols[1]:
            # CSV - Root Causes only
            if len(rc_data) > 0:
                rc_csv = rc_data[["DateTime", "NCR Number", "Category", "Description", "Status"]].to_csv(index=False)
                st.download_button(
                    "📥 Root Causes CSV",
                    data=rc_csv.encode("utf-8"),
                    file_name=f"Root_Causes_{datetime.date.today()}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.info("No Root Causes to export")
        
        with cols[2]:
            # CSV - Corrective Actions only
            if len(ca_data) > 0:
                ca_csv = ca_data[["DateTime", "NCR Number", "Category", "Description", "Status", "Effectiveness"]].to_csv(index=False)
                st.download_button(
                    "📥 Corrective Actions CSV",
                    data=ca_csv.encode("utf-8"),
                    file_name=f"Corrective_Actions_{datetime.date.today()}.csv",
                    mime="text/csv",
                    use_container_width=True
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
