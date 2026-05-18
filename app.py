"""
NCR Classification Dashboard v2 — BERTopic Edition
Root Causes & Corrective Actions Classification
Uses BERTopic (SBERT + UMAP + HDBSCAN) with Azure Blob Storage training data
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
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN

# ============================================================================
# CONFIG
# ============================================================================

EMBEDDING_MODEL_NAME = 'paraphrase-multilingual-mpnet-base-v2'
RANDOM_SEED = 42

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
    # Add more as needed
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
def load_training_data_from_azure(sas_url):
    """Load training dataset from Azure Blob Storage SAS URL."""
    try:
        if not sas_url:
            return None
        response = requests.get(sas_url, timeout=30)
        response.raise_for_status()
        df = pd.read_excel(BytesIO(response.content))
        return df
    except Exception as e:
        st.warning(f"Could not load training data: {str(e)}")
        return None

def clean_text(text):
    """Basic text cleaning."""
    import re
    text = str(text).lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def train_bertopic_model(descriptions, categories, n_topics=None):
    """Train BERTopic model and return model + category centroids."""
    try:
        model = load_sbert_model()
        
        if n_topics is None:
            n_topics = len(set(categories))
        
        # Pre-compute SBERT embeddings
        st.info("🔄 Computing SBERT embeddings...")
        embeddings = model.encode(descriptions, show_progress_bar=False, batch_size=64)
        
        # Set up BERTopic components
        umap_model = UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric='cosine',
            random_state=RANDOM_SEED
        )
        
        hdbscan_model = HDBSCAN(
            min_cluster_size=15,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )
        
        # Train BERTopic
        st.info("🔄 Training BERTopic model...")
        bt_model = BERTopic(
            nr_topics=n_topics,
            embedding_model=model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            calculate_probabilities=True,
            verbose=False
        )
        
        topics, probs = bt_model.fit_transform(descriptions, embeddings=embeddings)
        
        # Build category centroids
        st.info("🔄 Building category centroids...")
        cat_centroids = {}
        for cat in set(categories):
            cat_mask = np.array(categories) == cat
            cat_embeddings = embeddings[cat_mask]
            cat_centroids[cat] = cat_embeddings.mean(axis=0)
        
        st.success(f"✅ BERTopic trained on {len(descriptions)} examples")
        return model, embeddings, cat_centroids, bt_model
        
    except Exception as e:
        st.error(f"Error training BERTopic: {str(e)}")
        return None, None, None, None

def get_top_suggestions(user_text, model, cat_centroids, categories, top_n=3):
    """Get top N category suggestions using SBERT cosine similarity."""
    try:
        user_embedding = model.encode([user_text])[0]
        
        suggestions = []
        for category in sorted(set(categories)):
            if category in cat_centroids:
                centroid = cat_centroids[category]
                similarity = cosine_similarity([user_embedding], [centroid])[0][0]
                suggestions.append((category, float(similarity)))
        
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return suggestions[:top_n]
    except Exception as e:
        st.error(f"Error getting suggestions: {str(e)}")
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

# Custom CSS
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
    .stTextArea textarea:focus {
        border: 2px solid #dc2626 !important;
        box-shadow: 0 0 0 3px rgba(220,38,38,0.1) !important;
    }
    .sel-box {
        background: rgba(255,255,255,0.8);
        color: #000;
        border: 2px solid #1c3060;
        border-left: 4px solid #dc2626;
        padding: 12px;
        border-radius: 6px;
        margin-top: 10px;
    }
    hr {
        border: 1px solid #dc2626 !important;
        opacity: 0.3;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="header-container">', unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: #1c3060;'>🔍 NCR Classification</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>Root Causes & Corrective Actions</p>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'records' not in st.session_state:
    st.session_state.records = []

if 'ncr_counter' not in st.session_state:
    st.session_state.ncr_counter = 1

if 'sbert_model' not in st.session_state:
    st.session_state.sbert_model = None

if 'rc_centroids' not in st.session_state:
    st.session_state.rc_centroids = None

if 'ca_centroids' not in st.session_state:
    st.session_state.ca_centroids = None

if 'rc_selected_category' not in st.session_state:
    st.session_state.rc_selected_category = None

if 'ca_selected_category' not in st.session_state:
    st.session_state.ca_selected_category = None

# ============================================================================
# TABS
# ============================================================================

tab1, tab2, tab3 = st.tabs(["Root Causes", "Corrective Actions", "Records Browser"])

# ============================================================================
# TAB 1: ROOT CAUSES
# ============================================================================

with tab1:
    # Initialize model and centroids
    if st.session_state.rc_centroids is None:
        sas_url = st.secrets.get("TRAINING_DATA_RC_URL")
        
        if sas_url:
            st.info("🔄 Loading Root Causes training data from Azure...")
            df_training = load_training_data_from_azure(sas_url)
            
            if df_training is not None:
                # Find correct columns
                desc_col, cat_col = None, None
                if "Description" in df_training.columns:
                    desc_col = "Description"
                if "Root Cause Category_Final" in df_training.columns:
                    cat_col = "Root Cause Category_Final"
                
                if desc_col and cat_col:
                    # Clean data
                    descriptions = df_training[desc_col].dropna().astype(str).tolist()
                    categories = df_training[cat_col].dropna().astype(str).tolist()
                    valid_pairs = [(d, c) for d, c in zip(descriptions, categories) if d.strip() and c.strip()]
                    
                    if valid_pairs:
                        descriptions = [d for d, c in valid_pairs]
                        categories = [c for d, c in valid_pairs]
                        
                        st.success(f"✅ Loaded {len(valid_pairs)} Root Cause training examples")
                        
                        # Train BERTopic
                        model, embeddings, cat_centroids, bt_model = train_bertopic_model(
                            descriptions, categories, n_topics=len(ROOT_CAUSE_CATEGORIES)
                        )
                        
                        if cat_centroids:
                            st.session_state.rc_centroids = cat_centroids
                            st.session_state.sbert_model = model
        else:
            st.warning("⚠️ TRAINING_DATA_RC_URL not configured in Streamlit Secrets")
    
    st.markdown(
        "<p style='font-size:0.9rem; font-weight:500; color:#1c3060; margin-bottom:12px;'>"
        "Describe the root cause:</p>",
        unsafe_allow_html=True
    )
    
    rc_description = st.text_area(
        "Root Cause Description",
        height=120,
        label_visibility="collapsed",
        placeholder="Enter a detailed description of the root cause..."
    )
    
    if rc_description and st.session_state.rc_centroids and st.session_state.sbert_model:
        suggestions = get_top_suggestions(
            clean_text(rc_description),
            st.session_state.sbert_model,
            st.session_state.rc_centroids,
            ROOT_CAUSE_CATEGORIES,
            top_n=3
        )
        
        st.markdown("<p style='font-size:0.9rem; font-weight:500; color:#1c3060; margin-top:16px;'>"
                   "Root Cause Category — select one:</p>", unsafe_allow_html=True)
        
        cols = st.columns(4)
        
        for idx, (category, conf) in enumerate(suggestions):
            icon = "🟢" if conf >= 0.50 else ("🟡" if conf >= 0.30 else "🔴")
            short = category[:40] + "…" if len(category) > 40 else category
            label = f"{'✓ ' if st.session_state.rc_selected_category == category else ''}{icon} {short}\n[conf: {conf:.3f}]"
            
            with cols[idx]:
                if st.button(label, key=f"rc_btn_{idx}", use_container_width=True):
                    st.session_state.rc_selected_category = category
        
        # No Match button
        with cols[3]:
            if st.button("✗ No Match", key="rc_no_match", use_container_width=True):
                st.session_state.rc_selected_category = "No Match"
        
        # Display selection
        if st.session_state.rc_selected_category:
            sel_box_html = f"""
            <div class="sel-box">
                <strong>Selected:</strong> {st.session_state.rc_selected_category}
            </div>
            """
            st.markdown(sel_box_html, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                status = st.selectbox("Status", ["Open", "Closed", "In Progress"], key="rc_status")
            
            with col2:
                if st.button("💾 Save Record", key="rc_save", use_container_width=True):
                    ncr_id = f"NCR-{st.session_state.ncr_counter:05d}"
                    st.session_state.ncr_counter += 1
                    
                    record = {
                        "DateTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Type": "Root Cause",
                        "NCR Number": ncr_id,
                        "Description": rc_description,
                        "Category": st.session_state.rc_selected_category,
                        "Status": status
                    }
                    st.session_state.records.append(record)
                    st.success(f"✅ Saved as {ncr_id}")
                    st.session_state.rc_selected_category = None
            
            with col3:
                if st.button("🔄 Clear", key="rc_clear", use_container_width=True):
                    st.session_state.rc_selected_category = None

# ============================================================================
# TAB 2: CORRECTIVE ACTIONS
# ============================================================================

with tab2:
    # Initialize model and centroids
    if st.session_state.ca_centroids is None:
        sas_url = st.secrets.get("TRAINING_DATA_CA_URL")
        
        if sas_url:
            st.info("🔄 Loading Corrective Actions training data from Azure...")
            df_training = load_training_data_from_azure(sas_url)
            
            if df_training is not None:
                # Find correct columns
                desc_col, cat_col = None, None
                if "Description of Action" in df_training.columns:
                    desc_col = "Description of Action"
                if "Corrective Action Type_Final" in df_training.columns:
                    cat_col = "Corrective Action Type_Final"
                
                if desc_col and cat_col:
                    # Clean data
                    descriptions = df_training[desc_col].dropna().astype(str).tolist()
                    categories = df_training[cat_col].dropna().astype(str).tolist()
                    valid_pairs = [(d, c) for d, c in zip(descriptions, categories) if d.strip() and c.strip()]
                    
                    if valid_pairs:
                        descriptions = [d for d, c in valid_pairs]
                        categories = [c for d, c in valid_pairs]
                        
                        st.success(f"✅ Loaded {len(valid_pairs)} Corrective Action training examples")
                        
                        # Train BERTopic
                        model, embeddings, cat_centroids, bt_model = train_bertopic_model(
                            descriptions, categories, n_topics=len(CORRECTIVE_ACTION_CATEGORIES)
                        )
                        
                        if cat_centroids:
                            st.session_state.ca_centroids = cat_centroids
                            st.session_state.sbert_model = model
        else:
            st.warning("⚠️ TRAINING_DATA_CA_URL not configured in Streamlit Secrets")
    
    st.markdown(
        "<p style='font-size:0.9rem; font-weight:500; color:#1c3060; margin-bottom:12px;'>"
        "Describe the corrective action:</p>",
        unsafe_allow_html=True
    )
    
    ca_description = st.text_area(
        "Corrective Action Description",
        height=120,
        label_visibility="collapsed",
        placeholder="Enter a detailed description of the corrective action..."
    )
    
    if ca_description and st.session_state.ca_centroids and st.session_state.sbert_model:
        suggestions = get_top_suggestions(
            clean_text(ca_description),
            st.session_state.sbert_model,
            st.session_state.ca_centroids,
            CORRECTIVE_ACTION_CATEGORIES,
            top_n=3
        )
        
        st.markdown("<p style='font-size:0.9rem; font-weight:500; color:#1c3060; margin-top:16px;'>"
                   "Corrective Action Type — select one:</p>", unsafe_allow_html=True)
        
        cols = st.columns(4)
        
        for idx, (category, conf) in enumerate(suggestions):
            icon = "🟢" if conf >= 0.50 else ("🟡" if conf >= 0.30 else "🔴")
            short = category[:40] + "…" if len(category) > 40 else category
            label = f"{'✓ ' if st.session_state.ca_selected_category == category else ''}{icon} {short}\n[conf: {conf:.3f}]"
            
            with cols[idx]:
                if st.button(label, key=f"ca_btn_{idx}", use_container_width=True):
                    st.session_state.ca_selected_category = category
        
        # No Match button
        with cols[3]:
            if st.button("✗ No Match", key="ca_no_match", use_container_width=True):
                st.session_state.ca_selected_category = "No Match"
        
        # Display selection
        if st.session_state.ca_selected_category:
            sel_box_html = f"""
            <div class="sel-box">
                <strong>Selected:</strong> {st.session_state.ca_selected_category}
            </div>
            """
            st.markdown(sel_box_html, unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                status = st.selectbox("Status", ["Open", "Closed", "In Progress"], key="ca_status")
            
            with col2:
                effectiveness = st.selectbox("Effectiveness", ["Effective", "Ineffective", "Pending Review"], key="ca_effectiveness")
            
            with col3:
                if st.button("💾 Save Record", key="ca_save", use_container_width=True):
                    ncr_id = f"NCR-{st.session_state.ncr_counter:05d}"
                    st.session_state.ncr_counter += 1
                    
                    record = {
                        "DateTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Type": "Corrective Action",
                        "NCR Number": ncr_id,
                        "Description": ca_description,
                        "Category": st.session_state.ca_selected_category,
                        "Status": status,
                        "Effectiveness": effectiveness
                    }
                    st.session_state.records.append(record)
                    st.success(f"✅ Saved as {ncr_id}")
                    st.session_state.ca_selected_category = None
            
            with col4:
                if st.button("🔄 Clear", key="ca_clear", use_container_width=True):
                    st.session_state.ca_selected_category = None

# ============================================================================
# TAB 3: RECORDS BROWSER
# ============================================================================

with tab3:
    st.subheader("📊 Records Browser")
    
    if st.session_state.records:
        df_records = pd.DataFrame(st.session_state.records)
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Records", len(df_records))
        col2.metric("Root Causes", len(df_records[df_records['Type'] == 'Root Cause']))
        col3.metric("Corrective Actions", len(df_records[df_records['Type'] == 'Corrective Action']))
        col4.metric("Unique NCRs", df_records['NCR Number'].nunique())
        
        st.divider()
        
        # Filters
        col1, col2 = st.columns(2)
        with col1:
            record_types = st.multiselect("Record Type", df_records['Type'].unique())
        with col2:
            ncr_numbers = st.multiselect("NCR Number", df_records['NCR Number'].unique())
        
        # Apply filters
        filtered_df = df_records
        if record_types:
            filtered_df = filtered_df[filtered_df['Type'].isin(record_types)]
        if ncr_numbers:
            filtered_df = filtered_df[filtered_df['NCR Number'].isin(ncr_numbers)]
        
        # Display table
        st.dataframe(filtered_df[['DateTime', 'Type', 'NCR Number', 'Category', 'Description']], use_container_width=True)
        
        st.divider()
        
        # Exports
        st.subheader("📥 Export Records")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Export as Excel"):
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    filtered_df[['DateTime', 'NCR Number', 'Category', 'Description']].to_excel(writer, sheet_name='Records', index=False)
                st.download_button(
                    label="Download Excel",
                    data=output.getvalue(),
                    file_name="ncr_records.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        
        with col2:
            if st.button("📄 Export as CSV"):
                csv = filtered_df[['DateTime', 'NCR Number', 'Category', 'Description']].to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="ncr_records.csv",
                    mime="text/csv"
                )
        
        with col3:
            if st.button("🗑️ Clear All Records"):
                st.session_state.records = []
                st.rerun()
    else:
        st.info("📭 No records yet. Create records in the Root Causes or Corrective Actions tabs.")
