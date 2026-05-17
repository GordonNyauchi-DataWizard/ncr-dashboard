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
section[data-testid="stSidebar"] h2 { color: #dc2626 !important; font-weight: 700; }

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
    color: #dc2626 !important;
    border-bottom-color: #dc2626 !important;
    background: #fef2f2 !important;
}

/* ── Cards & containers ── */
.description-card {
    background: #fef2f2;
    border: 1px solid #fecaca;
    border-radius: 8px;
    padding: 20px;
    margin: 16px 0;
}

.suggestion-box {
    background: #ffffff;
    border: 2px solid #dc2626;
    border-radius: 8px;
    padding: 14px 16px;
    margin: 8px 0;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
}
.suggestion-box:hover {
    background: #fef2f2;
    transform: translateX(4px);
    box-shadow: 0 2px 8px rgba(220, 38, 38, 0.15);
}

.no-match-box {
    background: #f3f4f6;
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
    background: #e5e7eb;
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
    background: #dc2626 !important;
    color: white !important;
    border: none !important;
    font-weight: 600 !important;
    padding: 10px 24px !important;
    border-radius: 6px !important;
    transition: all 0.2s !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: #b91c1c !important;
    box-shadow: 0 4px 12px rgba(220, 38, 38, 0.3) !important;
}

/* ── Text input ── */
.stTextArea textarea {
    border: 1px solid #e5e7eb !important;
    border-radius: 6px !important;
    font-family: 'Geist', sans-serif !important;
    padding: 12px !important;
}
.stTextArea textarea:focus {
    border: 2px solid #dc2626 !important;
    box-shadow: 0 0 0 3px rgba(220, 38, 38, 0.1) !important;
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

def get_top_suggestions(user_text, centroids, categories, top_n=3, confidence_threshold=0.30):
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
    
    # Filter by threshold and return top N
    suggestions = [(cat, conf) for cat, conf in scores if conf >= confidence_threshold][:top_n]
    
    return suggestions

def confidence_color(conf):
    """Return HTML color class for confidence level."""
    if conf >= 0.60:
        return "conf-high"
    elif conf >= 0.40:
        return "conf-mid"
    else:
        return "conf-low"

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
tab1, tab2, tab3 = st.tabs(["🔴 Root Causes", "🟠 Corrective Actions", "📁 Records Browser"])

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
        
        st.markdown("#### Select Root Cause Category")
        
        selected_rc = None
        
        # Suggestion buttons
        for cat, conf in suggestions:
            conf_class = confidence_color(conf)
            cols = st.columns([0.95, 0.05])
            with cols[0]:
                if st.button(
                    f"**{cat}** — Confidence: {conf:.1%}",
                    key=f"rc_btn_{cat}_{conf}",
                    use_container_width=True
                ):
                    selected_rc = cat
        
        # No Match button
        cols = st.columns([0.95, 0.05])
        with cols[0]:
            if st.button(
                "⊘ No Match",
                key="rc_nomatch",
                use_container_width=True
            ):
                selected_rc = "No Match"
        
        # Show selection
        if selected_rc:
            st.markdown(f"""
            <div class="selection-badge">
                ✓ Selected: <strong>{selected_rc}</strong>
            </div>
            """, unsafe_allow_html=True)
            
            # Save section
            st.markdown("---")
            st.markdown("#### Save Record")
            
            cols = st.columns([1, 1])
            with cols[0]:
                ncr_number = st.text_input("NCR Number", placeholder="e.g., NCR-26-00001")
            with cols[1]:
                status = st.selectbox("Status", ["Open", "Closed", "In Progress"])
            
            retain_elim = st.radio(
                "Retain or Eliminate",
                ["Retain", "Eliminate"],
                horizontal=True
            )
            
            if st.button("💾 Save Root Cause Record", use_container_width=True):
                if not ncr_number.strip():
                    st.error("Please enter an NCR Number")
                else:
                    record = {
                        "DateTime": datetime.datetime.now(),
                        "Type": "Root Cause",
                        "NCR Number": ncr_number,
                        "Description": rc_description,
                        "Category": selected_rc,
                        "Confidence": suggestions[0][1] if suggestions and selected_rc != "No Match" else 0.0,
                        "Retain or Eliminate": retain_elim,
                        "Status": status,
                    }
                    st.session_state.records.append(record)
                    st.success(f"✓ Root cause saved to NCR {ncr_number}")
                    st.session_state.rc_description = ""  # Clear input

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
        
        st.markdown("#### Select Corrective Action Type")
        
        selected_ca = None
        
        # Suggestion buttons
        for cat, conf in suggestions:
            conf_class = confidence_color(conf)
            cols = st.columns([0.95, 0.05])
            with cols[0]:
                if st.button(
                    f"**{cat}** — Confidence: {conf:.1%}",
                    key=f"ca_btn_{cat}_{conf}",
                    use_container_width=True
                ):
                    selected_ca = cat
        
        # No Match button
        cols = st.columns([0.95, 0.05])
        with cols[0]:
            if st.button(
                "⊘ No Match",
                key="ca_nomatch",
                use_container_width=True
            ):
                selected_ca = "No Match"
        
        # Show selection
        if selected_ca:
            st.markdown(f"""
            <div class="selection-badge">
                ✓ Selected: <strong>{selected_ca}</strong>
            </div>
            """, unsafe_allow_html=True)
            
            # Save section
            st.markdown("---")
            st.markdown("#### Save Record")
            
            cols = st.columns([1, 1])
            with cols[0]:
                ncr_number = st.text_input("NCR Number", placeholder="e.g., NCR-26-00001", key="ca_ncr")
            with cols[1]:
                status = st.selectbox("Status", ["Open", "Closed", "In Progress"], key="ca_status")
            
            effective = st.radio(
                "Effectiveness",
                ["Effective", "Ineffective", "Pending Review"],
                horizontal=True,
                key="ca_effective"
            )
            
            if st.button("💾 Save Corrective Action Record", use_container_width=True):
                if not ncr_number.strip():
                    st.error("Please enter an NCR Number")
                else:
                    record = {
                        "DateTime": datetime.datetime.now(),
                        "Type": "Corrective Action",
                        "NCR Number": ncr_number,
                        "Description": ca_description,
                        "Category": selected_ca,
                        "Confidence": suggestions[0][1] if suggestions and selected_ca != "No Match" else 0.0,
                        "Effectiveness": effective,
                        "Status": status,
                    }
                    st.session_state.records.append(record)
                    st.success(f"✓ Corrective action saved to NCR {ncr_number}")

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
        display_cols = ["DateTime", "Type", "NCR Number", "Category", "Status", "Confidence"]
        st.dataframe(fdf[display_cols], use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.markdown("#### Export")
        
        # Export options
        cols = st.columns(3)
        
        with cols[0]:
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as w:
                fdf.to_excel(w, sheet_name="Records", index=False)
                # Add summary sheet
                summary = fdf["Category"].value_counts().reset_index()
                summary.columns = ["Category", "Count"]
                summary.to_excel(w, sheet_name="Summary", index=False)
            
            st.download_button(
                "📥 Excel",
                data=buf.getvalue(),
                file_name=f"NCR_Records_{datetime.date.today()}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        
        with cols[1]:
            st.download_button(
                "📥 CSV",
                data=fdf.to_csv(index=False).encode("utf-8"),
                file_name=f"NCR_Records_{datetime.date.today()}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with cols[2]:
            html_doc = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>NCR Records — {datetime.date.today()}</title>
    <style>
        body {{
            font-family: 'Geist', Arial, sans-serif;
            margin: 30px;
            color: #1f2937;
            background: #ffffff;
        }}
        h1 {{
            color: #dc2626;
            border-bottom: 3px solid #dc2626;
            padding-bottom: 10px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        th {{
            background: #dc2626;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }}
        td {{
            padding: 10px 12px;
            border-bottom: 1px solid #e5e7eb;
        }}
        tr:nth-child(even) {{
            background: #f9fafb;
        }}
    </style>
</head>
<body>
    <h1>NCR Classification Records</h1>
    <p><strong>Generated:</strong> {datetime.date.today()}</p>
    {fdf.to_html(index=False, border=0)}
</body>
</html>"""
            
            st.download_button(
                "📥 HTML",
                data=html_doc.encode("utf-8"),
                file_name=f"NCR_Records_{datetime.date.today()}.html",
                mime="text/html",
                use_container_width=True,
                help="Open in browser → Ctrl+P → Save as PDF"
            )
        
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
