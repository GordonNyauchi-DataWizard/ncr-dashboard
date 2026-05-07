"""
NCR Root Cause Classifier — ETQ-style Dashboard
================================================
- BERTopic (SBERT) suggests top 3 categories + No Match
- One root cause = one record
- No manual dropdown, no Problem Description field
- High-contrast light theme matching ETQ aesthetics

FIXES APPLIED:
1. Least Frequent Category: Now correctly handles ties (shows all tied categories)
2. NCR Volume Trend: Improved timestamp parsing and error handling
3. Pie Chart: Ensures all labels display with proper formatting
4. Data consistency: Better session state initialization
"""

import re, io, os, warnings, datetime
import streamlit as st
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_FILE_VARIANTS = [
    "Root_Causes_Final_Dataset.xlsx",
    "Root_Causes.xlsx",
    "Root_Causes_Final.xlsx",
    "root_causes.xlsx",
]

st.set_page_config(
    page_title="NCR Root Cause Classifier",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background: #f2f4f8 !important;
    color: #1c2b4a !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #1c3060 !important;
    border-right: 3px solid #0f2040 !important;
}
section[data-testid="stSidebar"] * { color: #e8eef8 !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #ffffff !important; }
section[data-testid="stSidebar"] .stMetric label { color: #a8c0e8 !important; }
section[data-testid="stSidebar"] .stMetric [data-testid="stMetricValue"] { color: #ffffff !important; }

/* ── Top banner ── */
.etq-banner {
    background: linear-gradient(90deg, #1c3060 0%, #2a4a8a 100%);
    color: #ffffff;
    padding: 10px 20px;
    border-radius: 6px;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.etq-banner .ncr-id { font-family: 'JetBrains Mono', monospace; font-size: 1rem; font-weight: 700; color: #a8d0ff; }
.etq-banner .ncr-title { font-size: 0.8rem; color: #c8daff; }

/* ── Workflow bar ── */
.workflow-bar {
    display: flex;
    margin-bottom: 18px;
    border-radius: 6px;
    overflow: hidden;
    border: 1px solid #c8d4e8;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
}
.wf-step {
    flex: 1;
    text-align: center;
    padding: 7px 2px;
    font-size: 0.62rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    background: #e8edf8;
    color: #6080aa;
    border-right: 1px solid #c8d4e8;
}
.wf-step:last-child { border-right: none; }
.wf-step.active { background: #1c3060; color: #ffffff; }
.wf-step.done   { background: #1a6640; color: #ffffff; }

/* ── Section cards ── */
.etq-card {
    background: #ffffff;
    border: 1px solid #c8d4e8;
    border-radius: 6px;
    margin-bottom: 14px;
    overflow: hidden;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.etq-card-header {
    background: #dde6f4;
    border-bottom: 2px solid #b0c4e0;
    padding: 7px 16px;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #1c3060;
}
.etq-card-body { padding: 16px; }

/* ── BERTopic suggestion buttons ── */
.cat-row { display: flex; gap: 10px; flex-wrap: wrap; margin: 10px 0 16px 0; }

/* ── Selection display ── */
.sel-box {
    border-radius: 5px;
    padding: 10px 14px;
    font-weight: 600;
    font-size: 0.9rem;
    margin-top: 8px;
    border-left: 4px solid;
}
.sel-match    { background: #e8f4ec; border-color: #1a8040; color: #0f4020; }
.sel-nomatch  { background: #fdecea; border-color: #c62828; color: #7f0000; }

/* ── Confidence badge colors ── */
.conf-g { color: #1a8040; font-weight: 700; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; }
.conf-y { color: #c07800; font-weight: 700; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; }
.conf-r { color: #c62828; font-weight: 700; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; }

/* ── Analytics metric card ── */
.metric-card {
    background: #ffffff;
    border: 1px solid #c8d4e8;
    border-top: 3px solid #1c3060;
    border-radius: 6px;
    padding: 16px;
    text-align: center;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.metric-value { font-size: 2.2rem; font-weight: 700; color: #1c3060; font-family: 'JetBrains Mono', monospace; }
.metric-label { font-size: 0.7rem; color: #607090; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 4px; }

/* ── Table ── */
.stDataFrame { border: 1px solid #c8d4e8 !important; border-radius: 6px !important; }

hr { border-color: #c8d4e8 !important; }
</style>
""", unsafe_allow_html=True)

# ── All 58 Root Cause Categories ─────────────────────────────────────────────
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
    "Quality Systems",
    "Quality Systems : CAPA Effectiveness issues",
    "Quality Systems : Change control issues",
    "Quality Systems : Maintenance of records",
    "Quality Systems : Opportunity for Improvement identified",
    "Quality Systems : Post-market surveillance issues",
    "Quality Systems : Preventive Maintenance",
    "Quality Systems : Top Management Review",
    "Quality Systems : Training/Competency",
    "Quality Systems : Validation/Verification",
    "Retest/Rework",
    "Supplier or External",
    "Supervision & Management Factors",
    "Supervision & Management Factors : Inadequate Management",
    "Supervision & Management Factors : Inadequate Management of Change",
    "Supervision & Management Factors : Inadequate Scheduling",
    "Supervision & Management Factors : Inadequate Staffing",
    "Supervision & Management Factors : Inadequate Supervision",
    "Supervision & Management Factors : Inadequate Training",
    "Supervision & Management Factors : Lack of accountability",
    "Training : Inadequate Retraining Frequency",
    "Training : Inadequate Training Content",
    "Training : Lack of competency assessment",
    "Other",
    "Cannot be determined",
]

# ────────────────────────────────────────────────────────────────────────────────
# SESSION STATE & HELPER FUNCTIONS
# ────────────────────────────────────────────────────────────────────────────────

if "records" not in st.session_state:
    st.session_state.records = []

if "model" not in st.session_state:
    st.session_state.model = None

if "category_centroids" not in st.session_state:
    st.session_state.category_centroids = {}

def suggest_categories(text, confidence_threshold=0.30):
    """
    SBERT-based category suggestion.
    Returns top 3 matches + No Match option.
    """
    if not text.strip():
        return []
    
    try:
        from sentence_transformers import util
        if st.session_state.model is None or not st.session_state.category_centroids:
            return []
        
        text_emb = st.session_state.model.encode(text, convert_to_tensor=True)
        
        scores_dict = {}
        for cat in ROOT_CAUSE_CATEGORIES:
            if cat in ["Other", "Cannot be determined"]:
                continue
            if cat not in st.session_state.category_centroids:
                continue
            
            cat_emb = st.session_state.category_centroids[cat]
            sim = util.pytorch_cos_sim(text_emb, cat_emb).item()
            if sim >= confidence_threshold:
                scores_dict[cat] = sim
        
        top_3 = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)[:3]
        return [cat for cat, _ in top_3]
    except Exception as e:
        st.warning(f"Category suggestion error: {e}")
        return []

def load_model():
    """Load SBERT model and build category centroids from training data."""
    if st.session_state.model is not None and st.session_state.category_centroids:
        return
    
    try:
        from sentence_transformers import SentenceTransformer
        
        with st.spinner("Loading SBERT model..."):
            st.session_state.model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
        
        training_file = None
        for fn in TRAINING_FILE_VARIANTS:
            full_path = os.path.join(_HERE, fn)
            if os.path.exists(full_path):
                training_file = full_path
                break
        
        if training_file:
            try:
                with st.spinner("Building category centroids from training data..."):
                    df_train = pd.read_excel(training_file)
                    
                    if "Root_Cause_Description" in df_train.columns and "Root_Cause_Category" in df_train.columns:
                        for cat in df_train["Root_Cause_Category"].unique():
                            cat_texts = df_train[df_train["Root_Cause_Category"] == cat]["Root_Cause_Description"].dropna()
                            if len(cat_texts) > 0:
                                embeddings = st.session_state.model.encode(cat_texts.tolist(), convert_to_tensor=True)
                                centroid = embeddings.mean(dim=0)
                                st.session_state.category_centroids[cat] = centroid
                        
                        st.success(f"✅ Trained from local file — {len(df_train)} rows, {len(st.session_state.category_centroids)} categories")
            except Exception as e:
                st.warning(f"Could not load training data: {e}. Using fallback category embeddings.")
                for cat in ROOT_CAUSE_CATEGORIES:
                    if cat not in ["Other", "Cannot be determined"]:
                        st.session_state.category_centroids[cat] = st.session_state.model.encode(cat, convert_to_tensor=True)
        else:
            st.warning("No training file found. Using fallback category embeddings.")
            for cat in ROOT_CAUSE_CATEGORIES:
                if cat not in ["Other", "Cannot be determined"]:
                    st.session_state.category_centroids[cat] = st.session_state.model.encode(cat, convert_to_tensor=True)
    
    except ImportError:
        st.warning("Install: `pip install sentence-transformers`")

# ─────────────────────────────────────────────────────────────────────────────────
# PAGE NAVIGATION
# ─────────────────────────────────────────────────────────────────────────────────

st.sidebar.markdown("# 🔍 **NCR Classifier**")
page = st.sidebar.radio("", ["✏️ Classify", "📊 Analytics", "📁 Records Browser"])

# ─────────────────────────────────────────────────────────────────────────────────
# PAGE 1 — CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────────

if page == "✏️ Classify":
    load_model()
    
    st.markdown("## ✏️ Classify Root Cause")
    
    st.markdown("### NCR Details")
    c1, c2 = st.columns([2, 1])
    with c1:
        ncr_num = st.text_input("NCR Number", placeholder="e.g., NCR-2025-001", key="ncr_number")
    with c2:
        status_opt = ["Open", "In Progress", "Closed"]
        status = st.selectbox("Status", status_opt, key="status")
    
    st.markdown("### Root Cause Description")
    rc_text = st.text_area(
        "Enter the root cause description (free text, any language):",
        height=120,
        placeholder="Describe what caused the non-conformance...",
        key="rc_desc"
    )
    
    suggestions = suggest_categories(rc_text) if rc_text.strip() else []
    
    st.markdown("### ✨ BERTopic Suggestions")
    if suggestions:
        cols = st.columns(len(suggestions) + 1)
        selected_cat = None
        for i, cat in enumerate(suggestions):
            with cols[i]:
                if st.button(cat, use_container_width=True):
                    selected_cat = cat
        with cols[len(suggestions)]:
            if st.button("❌ No Match", use_container_width=True):
                selected_cat = "No Match"
        
        if selected_cat:
            if selected_cat == "No Match":
                st.markdown('<div class="sel-box sel-nomatch">❌ No Match selected</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="sel-box sel-match">✅ {selected_cat}</div>', unsafe_allow_html=True)
    else:
        selected_cat = None
    
    st.markdown("### Additional Information")
    c1, c2 = st.columns(2)
    with c1:
        investigation_info = st.text_area("Investigation Information", height=80, key="investigation_info")
    with c2:
        containment = st.text_area("Containment Actions", height=80, key="containment")
    
    c3, c4 = st.columns(2)
    with c3:
        risk_assessment = st.text_area("Risk Assessment", height=80, key="risk_assessment")
    with c4:
        capa_decision = st.text_area("CAPA Decision", height=80, key="capa_decision")
    
    retain_or_eliminate = st.radio("Retain or Eliminate", ["Retain", "Eliminate"], horizontal=True, key="retain_eliminate")
    rationale = st.text_area("Rationale", height=60, key="rationale")
    
    if st.button("💾 Save Root Cause Record", use_container_width=True, type="primary"):
        if not ncr_num or not rc_text or selected_cat is None:
            st.error("Please fill in NCR Number, Root Cause Description, and select a category.")
        else:
            record = {
                "Timestamp": datetime.datetime.now().isoformat(),
                "NCR Number": ncr_num,
                "Status": status,
                "RC Description": rc_text,
                "Root Cause Category": selected_cat,
                "Investigation Info": investigation_info,
                "Containment": containment,
                "Risk Assessment": risk_assessment,
                "Disposition / CAPA": capa_decision,
                "Retain or Eliminate": retain_or_eliminate,
                "Rationale": rationale,
            }
            st.session_state.records.append(record)
            st.success(f"✅ Saved! {len(st.session_state.records)} record(s) in session.")
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────────────
# PAGE 2 — ANALYTICS DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────────

elif page == "📊 Analytics":
    st.markdown("## 📊 Analytics Dashboard")
    
    if not st.session_state.records:
        st.info("No records yet. Please classify some root causes first.")
    else:
        df = pd.DataFrame(st.session_state.records)
        
        total   = len(df)
        n_ncrs  = df["NCR Number"].nunique()
        n_cats  = df["Root Cause Category"].nunique()
        matched = (df["Root Cause Category"] != "No Match").sum()
        match_pct = f"{matched/max(total,1)*100:.0f}%"
        
        m1, m2, m3, m4 = st.columns(4)
        for col, val, lbl in [
            (m1, total,     "Total RC Records"),
            (m2, n_ncrs,    "Unique NCRs"),
            (m3, n_cats,    "Categories Used"),
            (m4, match_pct, "BERTopic Match Rate"),
        ]:
            col.markdown(f'<div class="metric-card"><div class="metric-value">{val}</div><div class="metric-label">{lbl}</div></div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        try:
            import plotly.express as px
            T    = "plotly_white"
            FONT = dict(family="Inter")
            BLUE = "#1c3060"
            
            c1, c2 = st.columns([3, 2])
            with c1:
                st.markdown("**Top 15 Root Cause Categories**")
                vc = df[df["Root Cause Category"] != "No Match"]["Root Cause Category"].value_counts().head(15).reset_index()
                vc.columns = ["Category", "Count"]
                vc["Label"] = vc["Category"].str[:55]
                fig = px.bar(vc, x="Count", y="Label", orientation="h",
                             color="Count", color_continuous_scale=[[0,"#a8c0e8"],[1,"#1c3060"]],
                             template=T, height=420)
                fig.update_layout(yaxis=dict(autorange="reversed"), showlegend=False,
                                  margin=dict(l=0,r=0,t=10,b=10), font=FONT, yaxis_title="",
                                  plot_bgcolor="#f8fafc")
                st.plotly_chart(fig, use_container_width=True)
            
            with c2:
                st.markdown("**Category Group Breakdown**")
                df["Group"] = df["Root Cause Category"].str.split(" : ").str[0]
                gc = df["Group"].value_counts().reset_index()
                gc.columns = ["Group", "Count"]
                fig2 = px.pie(gc, names="Group", values="Count", template=T,
                              color_discrete_sequence=px.colors.sequential.Blues_r, height=420)
                # FIX: Ensure all labels display properly
                fig2.update_traces(textposition="auto", textinfo="label+percent")
                fig2.update_layout(font=FONT, margin=dict(l=0,r=0,t=10,b=10), showlegend=True)
                st.plotly_chart(fig2, use_container_width=True)
            
            # FIX: Improved timestamp handling for NCR Volume Trend
            if "Timestamp" in df.columns:
                st.markdown("**NCR Volume Trend**")
                try:
                    # Parse timestamps safely
                    df["Date"] = pd.to_datetime(df["Timestamp"], errors='coerce').dt.date
                    df_trend = df.dropna(subset=["Date"])
                    
                    if len(df_trend) > 0:
                        trend = df_trend.groupby("Date").size().reset_index(name="Count")
                        trend = trend.sort_values("Date")
                        fig3 = px.line(trend, x="Date", y="Count", template=T, height=220,
                                       markers=True,
                                       color_discrete_sequence=["#1c3060"])
                        fig3.update_layout(margin=dict(l=0,r=0,t=10,b=10), font=FONT,
                                           plot_bgcolor="#f8fafc", hovermode="x unified")
                        st.plotly_chart(fig3, use_container_width=True)
                    else:
                        st.warning("No valid timestamps found in records. Trend chart cannot be displayed.")
                except Exception as e:
                    st.warning(f"Could not process timestamps: {e}")
            
            c3, c4 = st.columns(2)
            with c3:
                st.markdown("**Retain vs. Eliminate**")
                re_c = df["Retain or Eliminate"].value_counts().reset_index()
                re_c.columns = ["Decision", "Count"]
                fig4 = px.bar(re_c, x="Decision", y="Count", color="Decision",
                              color_discrete_map={"Retain": "#1c3060", "Eliminate": "#c62828"},
                              template=T, height=250)
                fig4.update_layout(showlegend=False, margin=dict(l=0,r=0,t=10,b=10),
                                   font=FONT, plot_bgcolor="#f8fafc")
                st.plotly_chart(fig4, use_container_width=True)
            
            with c4:
                st.markdown("**NCR Status**")
                sc = df.drop_duplicates("NCR Number")["Status"].value_counts().reset_index()
                sc.columns = ["Status", "Count"]
                fig5 = px.bar(sc, x="Status", y="Count", color="Status",
                              color_discrete_sequence=px.colors.sequential.Blues_r,
                              template=T, height=250)
                fig5.update_layout(showlegend=False, margin=dict(l=0,r=0,t=10,b=10),
                                   font=FONT, plot_bgcolor="#f8fafc")
                st.plotly_chart(fig5, use_container_width=True)
        
        except ImportError:
            st.warning("Install plotly: `pip install plotly`")
        
        st.markdown("---")
        st.markdown("**📋 Descriptive Summary**")
        
        # FIX: Proper handling of "Least Frequent Category" when there are ties
        no_match_n = (df["Root Cause Category"] == "No Match").sum()
        
        # Get most frequent
        if matched > 0:
            most_freq = df[df["Root Cause Category"] != "No Match"]["Root Cause Category"].value_counts().idxmax()
        else:
            most_freq = "N/A"
        
        # FIX: Get ALL least frequent categories (not just one when there's a tie)
        if matched > 0:
            vc = df[df["Root Cause Category"] != "No Match"]["Root Cause Category"].value_counts()
            min_count = vc.min()
            least_freq_list = vc[vc == min_count].index.tolist()
            # If multiple are tied, show all; otherwise show the single one
            least_freq = ", ".join(least_freq_list) if len(least_freq_list) > 1 else least_freq_list[0] if least_freq_list else "N/A"
        else:
            least_freq = "N/A"
        
        stats = pd.DataFrame({
            "Metric": [
                "Total Root Cause Records",
                "Unique NCR Numbers",
                "Unique Categories Used",
                "Most Frequent Category",
                "Least Frequent Category",
                "Avg Root Causes per NCR",
                "No Match Records",
                "Eliminate Decisions",
            ],
            "Value": [
                total,
                n_ncrs,
                n_cats,
                most_freq,
                least_freq,
                f"{total/max(n_ncrs,1):.1f}",
                f"{no_match_n} ({no_match_n/max(total,1)*100:.1f}%)",
                f"{(df['Retain or Eliminate']=='Eliminate').sum()} ({(df['Retain or Eliminate']=='Eliminate').sum()/max(total,1)*100:.1f}%)",
            ]
        })
        st.dataframe(stats, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────────
# PAGE 3 — RECORDS BROWSER & EXPORT
# ─────────────────────────────────────────────────────────────────────────────────

elif page == "📁 Records Browser":
    st.markdown("## 📁 Saved NCR Records")
    
    up = st.file_uploader("📂 Load existing records (CSV or Excel)", type=["csv", "xlsx"])
    if up:
        df_up = pd.read_csv(up) if up.name.endswith(".csv") else pd.read_excel(up)
        action = st.radio("Action", ["Append to existing records", "Replace existing records"], horizontal=True)
        if st.button("Load File"):
            st.session_state.records = df_up.to_dict("records") if "Replace" in action \
                else st.session_state.records + df_up.to_dict("records")
            st.success(f"Loaded {len(df_up)} records.")
            st.rerun()
    
    if not st.session_state.records:
        st.info("No records yet. Save NCR entries first.")
    else:
        df = pd.DataFrame(st.session_state.records)
        
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            ncr_f = st.multiselect("Filter NCR Number", sorted(df["NCR Number"].unique()))
        with fc2:
            cat_f = st.multiselect("Filter Category", sorted(df["Root Cause Category"].unique()))
        with fc3:
            sta_f = st.multiselect("Filter Status", sorted(df["Status"].unique()))
        
        fdf = df.copy()
        if ncr_f: fdf = fdf[fdf["NCR Number"].isin(ncr_f)]
        if cat_f: fdf = fdf[fdf["Root Cause Category"].isin(cat_f)]
        if sta_f: fdf = fdf[fdf["Status"].isin(sta_f)]
        
        st.markdown(f"Showing **{len(fdf)}** of **{len(df)}** records")
        st.dataframe(fdf, use_container_width=True, height=380)
        
        st.markdown("---")
        st.markdown("### 💾 Export")
        e1, e2, e3 = st.columns(3)
        
        with e1:
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as w:
                fdf.to_excel(w, sheet_name="NCR_Records", index=False)
                fdf["Root Cause Category"].value_counts().rename_axis("Category").reset_index(
                    name="Count").to_excel(w, sheet_name="Category_Summary", index=False)
            st.download_button("📥 Download Excel", data=buf.getvalue(),
                file_name=f"NCR_Records_{datetime.date.today()}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True)
        
        with e2:
            st.download_button("📥 Download CSV", data=fdf.to_csv(index=False).encode("utf-8"),
                file_name=f"NCR_Records_{datetime.date.today()}.csv",
                mime="text/csv", use_container_width=True)
        
        with e3:
            html_doc = f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<style>body{{font-family:Inter,Arial,sans-serif;font-size:11px;margin:20px;color:#1c2b4a}}
h2{{color:#1c3060;border-bottom:2px solid #1c3060;padding-bottom:6px}}
table{{border-collapse:collapse;width:100%}}
th{{background:#1c3060;color:white;padding:7px 10px;text-align:left;font-size:10px;text-transform:uppercase;letter-spacing:0.06em}}
td{{border:1px solid #c8d4e8;padding:6px 10px}}
tr:nth-child(even){{background:#f0f4f8}}
</style></head><body>
<h2>NCR Root Cause Records — {datetime.date.today()}</h2>
{fdf.to_html(index=False,border=0)}
</body></html>"""
            st.download_button("📥 Download PDF-ready HTML", data=html_doc.encode("utf-8"),
                file_name=f"NCR_Records_{datetime.date.today()}.html",
                mime="text/html", use_container_width=True,
                help="Open in browser → Ctrl+P → Save as PDF")
        
        st.markdown("---")
        if st.button("🗑️ Clear All Records"):
            st.session_state.records = []
            st.rerun()
