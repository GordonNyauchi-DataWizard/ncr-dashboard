"""
NCR Root Cause Classifier — ETQ-style Dashboard
================================================
- BERTopic (SBERT) suggests top 3 categories + No Match
- One root cause = one record
- No manual dropdown, no Problem Description field
- High-contrast light theme matching ETQ aesthetics
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
    "Production Environment",
    "Production Environment : Inadequate pest control",
    "Production Environment : Inadequate static control",
    "Production Environment : Room out of specification",
    "Supervision & Management Factors",
    "Supervision & Management Factors : Inadequate Management Oversight",
    "Supervision & Management Factors : Personnel Performance",
    "Training",
    "Training : Frequency not adequate",
    "Training : Inadequate Retraining Frequency",
    "Training : Inadequate Training Content",
    "Training : Inadequate Training Curriculum",
    "Training : Inadequate Training Method",
    "Training : Inadequate Training/Certification",
    "Training : Management Oversight",
    "Training : Training/Certification not assigned",
    "Work Environment",
    "Work Environment : Improper Environmental or Logistical Conditions",
    "Work Environment : Work Environment Issues",
    "Work Environment : Workplace Layout/Ergonomics",
]

RETAIN_ELIMINATE   = ["Retain", "Eliminate"]
NCR_STATUS_OPTIONS = ["Open", "In Progress", "Closed", "On Hold"]
NO_MATCH_PAT       = re.compile(r'^\s*(other|cannot be determined)\s*$', re.IGNORECASE)
CONF_THRESHOLD     = 0.30

# ── Session state ─────────────────────────────────────────────────────────────
_defaults = {
    "records":      [],
    "model_loaded": False,
    "sbert_model":  None,
    "cat_names":    None,
    "cat_matrix":   None,
    "train_source": None,
    "rc_desc":      "",
    "rc_category":  None,
    "suggestions":  [],
    "retain_elim":  "Retain",
    "rationale":    "",
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Helpers ───────────────────────────────────────────────────────────────────
def clean_text(text):
    t = str(text).lower()
    t = re.sub(r'\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b', '', t)
    t = re.sub(r'\b\d+\.?\d*\s*(mm|cm|inch|inches|lbf|units?|pcs?)?\b', '', t)
    t = re.sub(r'[^a-z\s]', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()

def find_training_file():
    for name in TRAINING_FILE_VARIANTS:
        p = os.path.join(_HERE, name)
        if os.path.exists(p):
            return p
    return None

@st.cache_resource(show_spinner=False)
def load_model_and_centroids():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
    path  = find_training_file()
    if path:
        df_raw   = pd.read_excel(path)
        desc_col = next((c for c in df_raw.columns if "description" in c.lower()), df_raw.columns[0])
        cat_col  = next(
            (c for c in df_raw.columns if "root cause" in c.lower() and "category" in c.lower()),
            next((c for c in df_raw.columns if "category" in c.lower()), df_raw.columns[1])
        )
        df = df_raw[[desc_col, cat_col]].copy()
        df.columns = ["description", "topic_label"]
        df["topic_label"] = df["topic_label"].astype(str).str.strip()
        df = df[~df["topic_label"].isin(["", "nan", "None", "NaN"])]
        df = df[~df["topic_label"].apply(lambda x: bool(NO_MATCH_PAT.match(x)))]
        df["clean_desc"] = df["description"].fillna("").apply(clean_text)
        df = df[df["clean_desc"].str.len() > 5].reset_index(drop=True)
        labels = df["topic_label"].values
        embs   = model.encode(df["clean_desc"].tolist(), normalize_embeddings=True,
                               show_progress_bar=False, batch_size=64)
        cat_names  = sorted(set(labels))
        cat_matrix = np.vstack([embs[labels == c].mean(axis=0) for c in cat_names])
        norms = np.linalg.norm(cat_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        cat_matrix /= norms
        source = f"✅ Trained from **{os.path.basename(path)}** — {len(df):,} rows, {len(cat_names)} categories"
    else:
        cat_names  = ROOT_CAUSE_CATEGORIES
        cat_matrix = model.encode(cat_names, normalize_embeddings=True, show_progress_bar=False)
        source     = "⚠️ Fallback mode — add Root_Causes_Final_Dataset.xlsx for best accuracy"
    return model, list(cat_names), cat_matrix, source

def ensure_model():
    if not st.session_state.model_loaded:
        with st.spinner("🤖 Loading BERTopic SBERT model and building category centroids…"):
            model, cat_names, cat_matrix, source = load_model_and_centroids()
            st.session_state.sbert_model  = model
            st.session_state.cat_names    = cat_names
            st.session_state.cat_matrix   = cat_matrix
            st.session_state.train_source = source
            st.session_state.model_loaded = True

def get_top3(text):
    if not st.session_state.model_loaded or not text or len(text.strip()) < 6:
        return []
    q   = st.session_state.sbert_model.encode([clean_text(text)], normalize_embeddings=True)
    sim = (q @ np.array(st.session_state.cat_matrix).T)[0]
    out = []
    for idx in sim.argsort()[::-1]:
        cat  = st.session_state.cat_names[idx]
        conf = round(float(sim[idx]), 4)
        if NO_MATCH_PAT.match(str(cat)):
            continue
        out.append({"category": cat, "confidence": conf})
        if len(out) == 3:
            break
    return out

def reset_form():
    st.session_state.rc_desc     = ""
    st.session_state.rc_category = None
    st.session_state.suggestions = []
    st.session_state.retain_elim = "Retain"
    st.session_state.rationale   = ""

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏭 NCR Root Cause System")
    st.markdown("---")
    page = st.radio("Navigation", [
        "📝 New NCR Entry",
        "📊 Analytics Dashboard",
        "📁 Records Browser",
    ])
    st.markdown("---")
    if st.session_state.model_loaded:
        st.markdown(st.session_state.train_source)
    else:
        tpath = find_training_file()
        if tpath:
            st.success(f"📂 Found: `{os.path.basename(tpath)}`")
        else:
            st.warning("No training file found.\nFallback mode will be used.")
        if st.button("🚀 Load Model", type="primary", use_container_width=True):
            ensure_model()
            st.rerun()
    st.markdown("---")
    st.metric("Saved Records", len(st.session_state.records))

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — NCR ENTRY
# ─────────────────────────────────────────────────────────────────────────────
if page == "📝 New NCR Entry":
    ensure_model()

    # Auto-generate NCR number
    ncr_num_default = f"NCR-{datetime.datetime.now().strftime('%y')}-{str(len(st.session_state.records)+1).zfill(5)}"

    # ── Top banner (ETQ style) ────────────────────────────────────────────────
    st.markdown(f"""
    <div class="etq-banner">
        <div>
            <div class="ncr-id">📋 {ncr_num_default}</div>
            <div class="ncr-title">Nonconformance Report — Root Cause Analysis</div>
        </div>
        <div style="font-size:0.75rem; color:#a8c8f0;">Investigation Phase</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Workflow bar ──────────────────────────────────────────────────────────
    st.markdown("""
    <div class="workflow-bar">
        <div class="wf-step done">Identification</div>
        <div class="wf-step active">Investigation</div>
        <div class="wf-step">Containment</div>
        <div class="wf-step">Approval</div>
        <div class="wf-step">Implementation</div>
        <div class="wf-step">Effectiveness</div>
        <div class="wf-step">Closed</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Left / Right column layout (mimics ETQ) ───────────────────────────────
    left_col, right_col = st.columns([1, 2])

    with left_col:
        # Problem Description nav panel (read-only labels like ETQ left panel)
        st.markdown("""
        <div class="etq-card">
            <div class="etq-card-header">Problem Description</div>
            <div class="etq-card-body">
                <p style="font-size:0.78rem; color:#445566; margin:4px 0;">▸ Investigation Information</p>
                <p style="font-size:0.78rem; color:#445566; margin:4px 0;">▸ Containment</p>
                <p style="font-size:0.78rem; color:#445566; margin:4px 0;">▸ Risk Assessment</p>
                <p style="font-size:0.78rem; font-weight:700; color:#1c3060; margin:4px 0;">▸ Root Cause Analysis Subform</p>
                <p style="font-size:0.78rem; color:#445566; margin:4px 0;">▸ Results of Investigation</p>
                <p style="font-size:0.78rem; color:#445566; margin:4px 0;">▸ CAPA Decision</p>
                <p style="font-size:0.78rem; color:#445566; margin:4px 0;">▸ Investigation Approvers</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # NCR ID fields
        st.markdown('<div class="etq-card"><div class="etq-card-header">NCR Identification</div><div class="etq-card-body">', unsafe_allow_html=True)
        ncr_num      = st.text_input("NCR Number", value=ncr_num_default)
        ncr_status   = st.selectbox("Status", NCR_STATUS_OPTIONS)
        ncr_date     = st.date_input("Date Initiated", value=datetime.date.today())
        initiated_by = st.text_input("Initiated By", placeholder="Name / ID")
        department   = st.text_input("Department / Area")
        st.markdown('</div></div>', unsafe_allow_html=True)

    with right_col:
        # ── ROOT CAUSE ANALYSIS SUBFORM ───────────────────────────────────────
        st.markdown('<div class="etq-card"><div class="etq-card-header">🔍 Root Cause Analysis Subform</div><div class="etq-card-body">', unsafe_allow_html=True)

        st.markdown(
            "<p style='font-size:0.82rem; color:#445566; margin-bottom:10px;'>"
            "Enter the root cause description. BERTopic will return the 3 best matching categories "
            "plus a <b>No Match</b> option. Click to select, then save the record."
            "</p>",
            unsafe_allow_html=True
        )

        # Description field
        rc_desc = st.text_area(
            "Root Cause Description ✱",
            value=st.session_state.rc_desc,
            height=100,
            placeholder="e.g. Operator assembled component incorrectly due to unclear work instruction…",
            key="rc_desc_widget",
        )

        # Update suggestions when description changes
        if rc_desc != st.session_state.rc_desc:
            st.session_state.rc_desc     = rc_desc
            st.session_state.rc_category = None
            st.session_state.suggestions = get_top3(rc_desc) if len(rc_desc.strip()) > 8 else []

        if rc_desc and len(rc_desc.strip()) > 8 and not st.session_state.suggestions and st.session_state.model_loaded:
            st.session_state.suggestions = get_top3(rc_desc)

        suggestions = st.session_state.suggestions

        # ── Category selection (4 buttons: 3 suggestions + No Match) ─────────
        if suggestions:
            st.markdown(
                "<p style='font-size:0.8rem; font-weight:600; color:#1c3060; margin: 12px 0 6px 0;'>"
                "Root Cause Category — select one:</p>",
                unsafe_allow_html=True
            )

            btn_cols = st.columns(4)
            conf_colors = ["🟢", "🟡", "🔴"]

            for i, sug in enumerate(suggestions):
                conf   = sug["confidence"]
                icon   = "🟢" if conf >= 0.50 else ("🟡" if conf >= 0.35 else "🔴")
                is_sel = st.session_state.rc_category == sug["category"]
                # Short label for button, full name shown below
                short  = sug["category"][:40] + "…" if len(sug["category"]) > 40 else sug["category"]
                label  = f"{'✓ ' if is_sel else ''}{icon} {short}\n[conf: {conf:.3f}]"
                with btn_cols[i]:
                    if st.button(label, key=f"sug_{i}", use_container_width=True,
                                  type="primary" if is_sel else "secondary"):
                        st.session_state.rc_category = sug["category"]
                        st.rerun()

            # No Match button
            is_nm = st.session_state.rc_category == "No Match"
            with btn_cols[3]:
                if st.button(
                    f"{'✓ ' if is_nm else ''}❌ No Match\n[below threshold]",
                    key="no_match_btn",
                    use_container_width=True,
                    type="primary" if is_nm else "secondary"
                ):
                    st.session_state.rc_category = "No Match"
                    st.rerun()

            # Show full selected category name
            if st.session_state.rc_category:
                if st.session_state.rc_category == "No Match":
                    st.markdown(
                        '<div class="sel-box sel-nomatch">❌ No Match — root cause could not be categorised</div>',
                        unsafe_allow_html=True
                    )
                else:
                    # Show full confidence for selected
                    sel_conf = next((s["confidence"] for s in suggestions if s["category"] == st.session_state.rc_category), None)
                    conf_str = f" &nbsp;|&nbsp; Confidence: {sel_conf:.3f}" if sel_conf else ""
                    st.markdown(
                        f'<div class="sel-box sel-match">✅ {st.session_state.rc_category}{conf_str}</div>',
                        unsafe_allow_html=True
                    )

        elif rc_desc and len(rc_desc.strip()) > 8 and not st.session_state.model_loaded:
            st.info("Click **Load Model** in the sidebar to get BERTopic suggestions.")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Return or Eliminate + Rationale ──────────────────────────────────
        re_col, rat_col = st.columns([1, 2])
        with re_col:
            re_val = st.radio(
                "Return or Eliminate Root Cause",
                RETAIN_ELIMINATE,
                index=RETAIN_ELIMINATE.index(st.session_state.retain_elim),
                horizontal=True,
                key="re_radio",
            )
            st.session_state.retain_elim = re_val

        with rat_col:
            rat_val = st.text_input(
                "Rationale for Return or Eliminate",
                value=st.session_state.rationale,
                placeholder="Explain why this root cause is retained or eliminated…",
                key="rat_input",
            )
            st.session_state.rationale = rat_val

        st.markdown('</div></div>', unsafe_allow_html=True)

        # ── Investigation details ─────────────────────────────────────────────
        st.markdown('<div class="etq-card"><div class="etq-card-header">Investigation Details</div><div class="etq-card-body">', unsafe_allow_html=True)
        inv_col1, inv_col2 = st.columns(2)
        with inv_col1:
            inv_info    = st.text_area("Investigation Information", height=70, key="inv_info", placeholder="Summary of investigation…")
            containment = st.text_area("Containment", height=70, key="contain_f", placeholder="Containment actions taken…")
        with inv_col2:
            risk_assess = st.text_area("Risk Assessment", height=70, key="risk_f", placeholder="Risk level and impact…")
            disposition = st.text_area("CAPA / Disposition Decision", height=70, key="disp_f", placeholder="Corrective action decision…")
        st.markdown('</div></div>', unsafe_allow_html=True)

        # ── Action buttons ────────────────────────────────────────────────────
        st.markdown("---")
        ab1, ab2, ab3 = st.columns([1, 1, 3])
        with ab1:
            save_btn = st.button("💾 Save Record", type="primary", use_container_width=True)
        with ab2:
            clear_btn = st.button("🗑️ Clear", use_container_width=True)

        if clear_btn:
            reset_form()
            st.rerun()

        if save_btn:
            if not st.session_state.rc_desc.strip():
                st.error("❌ Please enter a root cause description.")
            elif not st.session_state.rc_category:
                st.error("❌ Please select a category (or No Match) before saving.")
            else:
                ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.records.append({
                    "NCR Number":          ncr_num,
                    "Date Initiated":      str(ncr_date),
                    "Initiated By":        initiated_by,
                    "Department":          department,
                    "Status":              ncr_status,
                    "RC Description":      st.session_state.rc_desc,
                    "Root Cause Category": st.session_state.rc_category,
                    "Retain or Eliminate": st.session_state.retain_elim,
                    "Rationale":           st.session_state.rationale,
                    "Investigation Info":  inv_info,
                    "Containment":         containment,
                    "Risk Assessment":     risk_assess,
                    "Disposition / CAPA":  disposition,
                    "Timestamp":           ts,
                })
                st.success(f"✅ Record saved for **{ncr_num}** — Category: **{st.session_state.rc_category}**")
                reset_form()
                st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 Analytics Dashboard":
    st.markdown("## 📊 NCR Root Cause Analytics")

    if not st.session_state.records:
        st.info("No records yet — save some NCR entries first, or load an existing file below.")
        up = st.file_uploader("Load existing records (CSV or Excel)", type=["csv", "xlsx"])
        if up:
            df_up = pd.read_csv(up) if up.name.endswith(".csv") else pd.read_excel(up)
            st.session_state.records = df_up.to_dict("records")
            st.rerun()
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
                fig2.update_layout(font=FONT, margin=dict(l=0,r=0,t=10,b=10))
                st.plotly_chart(fig2, use_container_width=True)

            if "Timestamp" in df.columns:
                st.markdown("**NCR Volume Trend**")
                df["Date"] = pd.to_datetime(df["Timestamp"]).dt.date
                trend = df.groupby("Date").size().reset_index(name="Count")
                fig3 = px.area(trend, x="Date", y="Count", template=T, height=220,
                               color_discrete_sequence=["#1c3060"])
                fig3.update_layout(margin=dict(l=0,r=0,t=10,b=10), font=FONT,
                                   plot_bgcolor="#f8fafc")
                st.plotly_chart(fig3, use_container_width=True)

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
        no_match_n = (df["Root Cause Category"] == "No Match").sum()
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
                df[df["Root Cause Category"] != "No Match"]["Root Cause Category"].value_counts().idxmax() if matched > 0 else "N/A",
                df[df["Root Cause Category"] != "No Match"]["Root Cause Category"].value_counts().idxmin() if matched > 0 else "N/A",
                f"{total/max(n_ncrs,1):.1f}",
                f"{no_match_n} ({no_match_n/max(total,1)*100:.1f}%)",
                f"{(df['Retain or Eliminate']=='Eliminate').sum()} ({(df['Retain or Eliminate']=='Eliminate').sum()/max(total,1)*100:.1f}%)",
            ]
        })
        st.dataframe(stats, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — RECORDS BROWSER & EXPORT
# ─────────────────────────────────────────────────────────────────────────────
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
