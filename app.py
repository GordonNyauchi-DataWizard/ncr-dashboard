"""
NCR Classification Dashboard — CLEAN Edition
Azure data with caching, no loading messages after first load
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from io import BytesIO
import requests
import warnings
warnings.filterwarnings('ignore')

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================================
# CONFIG
# ============================================================================

EMBEDDING_MODEL_NAME = 'paraphrase-multilingual-mpnet-base-v2'

ROOT_CAUSE_CATEGORIES = [
    "Communication: Inadequate communication",
    "Design: Inadequate design",
    "Design: Component design deficiency",
    "Design: Process design deficiency",
    "Documentation: Lack of sufficient detail",
    "Documentation: Outdated documentation",
    "Equipment: Failure/defect",
    "Equipment: Inadequate maintenance",
    "Equipment: Lack of maintenance",
    "Human Factors: Operator error",
    "Human Factors: Lack of awareness",
    "Human Factors: Inadequate training",
    "Inspection: Inadequate inspection",
    "Inspection: Lack of inspection",
    "Material: Defective material",
    "Material: Incompatible material",
    "Process: Inadequate process control",
    "Process: Lack of mistake proofing",
    "Process: Inadequate control",
    "Procedure: Inadequate procedure",
    "Procedure: Lack of procedure",
    "Supplier: Supplier quality issue",
    "System: Inadequate software functionality",
    "System: Software issue",
    "System: Data/information error",
    "Test/Analysis: Inadequate test",
    "Test/Analysis: Inadequate analysis",
    "Validation: Inadequate validation",
    "Validation: Inadequate acceptance criteria",
]

CORRECTIVE_ACTION_CATEGORIES = [
    "Design Modification",
    "Documentation Update",
    "Equipment Maintenance",
    "Equipment Replacement",
    "Process Change",
    "Procedure Update",
    "Supplier Change",
    "Training Program",
    "Validation Study"
]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_resource
def load_sbert_model():
    """Load SBERT model once and cache it."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

@st.cache_resource
def load_azure_data(sas_url, column_name, category_column):
    """Load training data from Azure and cache it."""
    try:
        response = requests.get(sas_url, timeout=60)
        response.raise_for_status()
        df = pd.read_excel(BytesIO(response.content))
        
        descriptions = df[column_name].dropna().astype(str).tolist()
        categories = df[category_column].dropna().astype(str).tolist()
        
        valid_pairs = [(d, c) for d, c in zip(descriptions, categories) if d.strip() and c.strip()]
        
        if valid_pairs:
            descriptions = [d for d, c in valid_pairs]
            categories = [c for d, c in valid_pairs]
            return descriptions, categories
        return None, None
    except Exception as e:
        return None, None

@st.cache_resource
def train_sbert_centroids(descriptions, categories, show_progress=True):
    """Train SBERT centroids and cache them."""
    try:
        model = load_sbert_model()
        
        if show_progress:
            embeddings = model.encode(descriptions, show_progress_bar=True, batch_size=64)
        else:
            embeddings = model.encode(descriptions, show_progress_bar=False, batch_size=64)
        
        cat_centroids = {}
        for cat in set(categories):
            cat_mask = np.array(categories) == cat
            cat_embeddings = embeddings[cat_mask]
            cat_centroids[cat] = cat_embeddings.mean(axis=0)
        
        return cat_centroids
        
    except Exception as e:
        return None

def clean_text(text):
    """Basic text cleaning."""
    import re
    text = str(text).lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def get_top_suggestions(user_text, model, cat_centroids, top_n=3):
    """Get top N suggestions using SBERT."""
    if cat_centroids is None:
        return []
    
    try:
        user_embedding = model.encode([user_text], show_progress_bar=False)[0]
        
        suggestions = []
        for category, centroid in cat_centroids.items():
            similarity = cosine_similarity([user_embedding], [centroid])[0][0]
            suggestions.append((category, float(similarity)))
        
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return suggestions[:top_n]
    except:
        return []

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="NCR Classification Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .header-container {
        border-top: 4px solid #dc2626;
        padding-top: 20px;
        margin-bottom: 30px;
    }
    .stButton > button {
        background: rgba(255,255,255,0.9);
        color: #000;
        border: 2px solid #1c3060;
        border-radius: 6px;
        font-weight: 500;
    }
    .stButton > button:hover {
        background: #dc2626;
        color: #fff;
        border: 2px solid #dc2626;
    }
    .sel-box {
        background: #f0f4f8;
        border-left: 4px solid #dc2626;
        padding: 12px;
        border-radius: 4px;
        margin: 16px 0;
    }
    body {
        background: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE - TRACK LOADING STATUS
# ============================================================================

if "records" not in st.session_state:
    st.session_state.records = []
if "ncr_counter" not in st.session_state:
    st.session_state.ncr_counter = 1
if "rc_selected" not in st.session_state:
    st.session_state.rc_selected = None
if "ca_selected" not in st.session_state:
    st.session_state.ca_selected = None
if "rc_loaded" not in st.session_state:
    st.session_state.rc_loaded = False
if "ca_loaded" not in st.session_state:
    st.session_state.ca_loaded = False

# ============================================================================
# HEADER
# ============================================================================

st.markdown(
    "<div class='header-container'><h1>🔍 NCR Classification Dashboard</h1>"
    "<p style='color:#666; font-size:0.95rem;'>Root Causes & Corrective Actions</p></div>",
    unsafe_allow_html=True
)

# ============================================================================
# LOAD MODEL
# ============================================================================

model = load_sbert_model()

# ============================================================================
# TABS
# ============================================================================

tab1, tab2, tab3 = st.tabs(["Root Causes", "Corrective Actions", "Records"])

# ============================================================================
# TAB 1: ROOT CAUSES
# ============================================================================

with tab1:
    # Load and cache Azure data (only show messages if not loaded yet)
    if not st.session_state.rc_loaded:
        sas_url = st.secrets.get("TRAINING_DATA_RC_URL")
        
        if sas_url:
            with st.spinner("Loading Root Causes..."):
                descriptions, categories = load_azure_data(sas_url, "Description", "Root Cause Category_Final")
                
                if descriptions and categories:
                    with st.spinner("Training model..."):
                        rc_centroids = train_sbert_centroids(descriptions, categories, show_progress=False)
                    
                    if rc_centroids:
                        st.session_state.rc_centroids = rc_centroids
                        st.session_state.rc_loaded = True
                        st.success(f"✅ Ready! Trained on {len(descriptions)} examples")
                else:
                    st.error("❌ Could not load Root Causes data")
        else:
            st.error("❌ TRAINING_DATA_RC_URL not in Secrets")
    
    # Show form only if loaded
    if st.session_state.rc_loaded:
        st.markdown("<p style='font-weight:500; color:#1c3060;'>Describe the root cause:</p>", unsafe_allow_html=True)
        rc_desc = st.text_area("Root Cause", height=100, label_visibility="collapsed", 
                               placeholder="Enter root cause description...")
        
        if rc_desc:
            suggestions = get_top_suggestions(clean_text(rc_desc), model, st.session_state.rc_centroids, 3)
            
            st.markdown("<p style='font-weight:500; color:#1c3060; margin-top:16px;'>Select category:</p>", unsafe_allow_html=True)
            
            cols = st.columns(4)
            for idx, (cat, conf) in enumerate(suggestions):
                icon = "🟢" if conf >= 0.50 else ("🟡" if conf >= 0.30 else "🔴")
                label = f"{'✓ ' if st.session_state.rc_selected == cat else ''}{icon} {cat[:30]}\n[{conf:.3f}]"
                with cols[idx]:
                    if st.button(label, key=f"rc_{idx}", use_container_width=True):
                        st.session_state.rc_selected = cat
            
            with cols[3]:
                if st.button("✗ No Match", key="rc_nomatch", use_container_width=True):
                    st.session_state.rc_selected = "No Match"
            
            if st.session_state.rc_selected:
                st.markdown(f"<div class='sel-box'><strong>Selected:</strong> {st.session_state.rc_selected}</div>", unsafe_allow_html=True)
                
                c1, c2, c3 = st.columns(3)
                with c1:
                    status = st.selectbox("Status", ["Open", "Closed", "In Progress"], key="rc_status")
                with c2:
                    if st.button("💾 Save", key="rc_save", use_container_width=True):
                        ncr = f"NCR-{st.session_state.ncr_counter:05d}"
                        st.session_state.ncr_counter += 1
                        st.session_state.records.append({
                            "DateTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Type": "Root Cause",
                            "NCR": ncr,
                            "Category": st.session_state.rc_selected,
                            "Description": rc_desc,
                            "Status": status
                        })
                        st.success(f"✅ Saved as {ncr}")
                        st.session_state.rc_selected = None
                with c3:
                    if st.button("🔄 Clear", key="rc_clear", use_container_width=True):
                        st.session_state.rc_selected = None

# ============================================================================
# TAB 2: CORRECTIVE ACTIONS
# ============================================================================

with tab2:
    # Load and cache Azure data (only show messages if not loaded yet)
    if not st.session_state.ca_loaded:
        sas_url = st.secrets.get("TRAINING_DATA_CA_URL")
        
        if sas_url:
            with st.spinner("Loading Corrective Actions..."):
                descriptions, categories = load_azure_data(sas_url, "Description of Action", "Corrective Action Type_Final")
                
                if descriptions and categories:
                    with st.spinner("Training model..."):
                        ca_centroids = train_sbert_centroids(descriptions, categories, show_progress=False)
                    
                    if ca_centroids:
                        st.session_state.ca_centroids = ca_centroids
                        st.session_state.ca_loaded = True
                        st.success(f"✅ Ready! Trained on {len(descriptions)} examples")
                else:
                    st.error("❌ Could not load Corrective Actions data")
        else:
            st.error("❌ TRAINING_DATA_CA_URL not in Secrets")
    
    # Show form only if loaded
    if st.session_state.ca_loaded:
        st.markdown("<p style='font-weight:500; color:#1c3060;'>Describe the corrective action:</p>", unsafe_allow_html=True)
        ca_desc = st.text_area("Corrective Action", height=100, label_visibility="collapsed",
                               placeholder="Enter corrective action description...")
        
        if ca_desc:
            suggestions = get_top_suggestions(clean_text(ca_desc), model, st.session_state.ca_centroids, 3)
            
            st.markdown("<p style='font-weight:500; color:#1c3060; margin-top:16px;'>Select category:</p>", unsafe_allow_html=True)
            
            cols = st.columns(4)
            for idx, (cat, conf) in enumerate(suggestions):
                icon = "🟢" if conf >= 0.50 else ("🟡" if conf >= 0.30 else "🔴")
                label = f"{'✓ ' if st.session_state.ca_selected == cat else ''}{icon} {cat[:30]}\n[{conf:.3f}]"
                with cols[idx]:
                    if st.button(label, key=f"ca_{idx}", use_container_width=True):
                        st.session_state.ca_selected = cat
            
            with cols[3]:
                if st.button("✗ No Match", key="ca_nomatch", use_container_width=True):
                    st.session_state.ca_selected = "No Match"
            
            if st.session_state.ca_selected:
                st.markdown(f"<div class='sel-box'><strong>Selected:</strong> {st.session_state.ca_selected}</div>", unsafe_allow_html=True)
                
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    status = st.selectbox("Status", ["Open", "Closed", "In Progress"], key="ca_status")
                with c2:
                    effectiveness = st.selectbox("Effectiveness", ["Effective", "Ineffective", "Pending"], key="ca_eff")
                with c3:
                    if st.button("💾 Save", key="ca_save", use_container_width=True):
                        ncr = f"NCR-{st.session_state.ncr_counter:05d}"
                        st.session_state.ncr_counter += 1
                        st.session_state.records.append({
                            "DateTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Type": "Corrective Action",
                            "NCR": ncr,
                            "Category": st.session_state.ca_selected,
                            "Description": ca_desc,
                            "Status": status,
                            "Effectiveness": effectiveness
                        })
                        st.success(f"✅ Saved as {ncr}")
                        st.session_state.ca_selected = None
                with c4:
                    if st.button("🔄 Clear", key="ca_clear", use_container_width=True):
                        st.session_state.ca_selected = None

# ============================================================================
# TAB 3: RECORDS
# ============================================================================

with tab3:
    st.subheader("📊 Records")
    
    if st.session_state.records:
        df = pd.DataFrame(st.session_state.records)
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total", len(df))
        c2.metric("Root Causes", len(df[df['Type'] == 'Root Cause']))
        c3.metric("Corrective Actions", len(df[df['Type'] == 'Corrective Action']))
        c4.metric("NCRs", df['NCR'].nunique())
        
        st.divider()
        st.dataframe(df, use_container_width=True)
        st.divider()
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button("📄 CSV", df.to_csv(index=False), "ncr.csv", "text/csv")
        with c2:
            output = BytesIO()
            df.to_excel(output, sheet_name='Records', index=False, engine='openpyxl')
            st.download_button("📊 Excel", output.getvalue(), "ncr.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        with c3:
            if st.button("🗑️ Clear All", use_container_width=True):
                st.session_state.records = []
                st.rerun()
    else:
        st.info("📭 No records yet.")
