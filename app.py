"""
NCR Root Cause Classifier Dashboard
====================================
Mimics ETQ Root Cause Analysis subform.
Uses BERTopic (SBERT cosine similarity) to suggest categories from free-text.

SETUP — just three steps:
  1. Put Root_Causes_Final_Dataset.xlsx in the SAME folder as this app.py
  2. pip install -r requirements.txt
  3. streamlit run app.py

The app auto-discovers and trains centroids from your Excel on startup.
No manual upload of training data required — ever.
"""

import re, io, os, warnings, datetime
import streamlit as st
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ── Where to look for training data (same folder as app.py) ──────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_FILE_VARIANTS = [
    "Root_Causes_Final_Dataset.xlsx",
    "Root_Causes.xlsx",
    "Root_Causes_Final.xlsx",
    "root_causes.xlsx",
]

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NCR Root Cause Classifier",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background: #0f1117; color: #e0e6f0; }
section[data-testid="stSidebar"] { background: #161b27 !important; border-right: 1px solid #1e2a3a; }
.card { background: #161b27; border: 1px solid #1e2d45; border-radius: 8px; padding: 20px; margin-bottom: 16px; }
.card-header { font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.12em;
               color: #5088c5; border-bottom: 1px solid #1e2d45; padding-bottom: 8px; margin-bottom: 14px; }
.workflow-bar { display: flex; gap: 0; margin-bottom: 20px; border-radius: 8px;
                overflow: hidden; border: 1px solid #1e2d45; }
.wf-step { flex: 1; text-align: center; padding: 8px 4px; font-size: 0.72rem; font-weight: 600;
           text-transform: uppercase; letter-spacing: 0.06em; background: #161b27; color: #445566;
           border-right: 1px solid #1e2d45; }
.wf-step:last-child { border-right: none; }
.wf-step.active { background: #1a2744; color: #60a8f8; }
.wf-step.done   { background: #1a3a28; color: #4ade80; }
.metric-card { background: #161b27; border: 1px solid #1e2d45; border-radius: 8px; padding: 16px; text-align: center; }
.metric-value { font-size: 2rem; font-weight: 700; color: #60a8f8; font-family: 'IBM Plex Mono', monospace; }
.metric-label { font-size: 0.75rem; color: #7090b0; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 4px; }
hr { border-color: #1e2d45; }
</style>
""", unsafe_allow_html=True)

# ── All 58 root cause categories (from your training data) ───────────────────
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

# ── Session state ─────────────────────────────────────────────────────────────
_defaults = {
    "records":      [],
    "model_loaded": False,
    "sbert_model":  None,
    "cat_names":    None,
    "cat_matrix":   None,
    "train_source": None,
    "rc_entries":   [{"desc":"","category":"","retain_eliminate":"Retain","rationale":"","_suggestions":[]}],
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Text cleaner (matches the notebook's clean() function) ───────────────────
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

# ── Core: load SBERT + build centroids (cached — runs once per server) ────────
@st.cache_resource(show_spinner=False)
def load_model_and_centroids():
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
    path  = find_training_file()

    if path:
        # Build real centroids from training data
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
        embs   = model.encode(
            df["clean_desc"].tolist(),
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=64,
        )
        cat_names  = sorted(set(labels))
        cat_matrix = np.vstack([embs[labels == c].mean(axis=0) for c in cat_names])
        norms      = np.linalg.norm(cat_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        cat_matrix /= norms
        source = f"✅ Trained from **{os.path.basename(path)}** — {len(df):,} rows, {len(cat_names)} categories"
    else:
        # Fallback: encode the 58 category names directly
        cat_names  = ROOT_CAUSE_CATEGORIES
        cat_matrix = model.encode(cat_names, normalize_embeddings=True, show_progress_bar=False)
        source     = "⚠️ Fallback mode — place **Root_Causes_Final_Dataset.xlsx** next to app.py for full accuracy"

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

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏭 NCR Root Cause System")
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
            st.warning("Training file not found in app folder.\nFallback mode will load.")
        if st.button("🚀 Load Model", type="primary"):
            ensure_model()
            st.rerun()

    st.markdown("---")
    st.metric("Saved Records", len(st.session_state.records))

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — NCR ENTRY
# ─────────────────────────────────────────────────────────────────────────────
if page == "📝 New NCR Entry":
    ensure_model()

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

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown('<div class="card"><div class="card-header">NCR Identification</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            ncr_num = st.text_input(
                "NCR Number",
                value=f"NCR-{datetime.datetime.now().strftime('%y')}-{str(len(st.session_state.records)+1).zfill(5)}",
            )
        with c2:
            ncr_status = st.selectbox("Status", NCR_STATUS_OPTIONS)
        ncr_date     = st.date_input("Date Initiated", value=datetime.date.today())
        initiated_by = st.text_input("Initiated By", placeholder="Name / Employee ID")
        department   = st.text_input("Department / Area")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="card"><div class="card-header">Problem Description</div>', unsafe_allow_html=True)
        problem_desc = st.text_area(
            "_pd", height=170,
            placeholder="Describe the nonconformance event in detail…",
            label_visibility="collapsed",
        )
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Root Cause Analysis Subform ───────────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">🔍 Root Cause Analysis Subform — BERTopic Powered</div>', unsafe_allow_html=True)
    st.caption("Type a root cause description → BERTopic suggests the top 3 matching categories. Click a suggestion or pick manually from the dropdown.")

    rc_entries = st.session_state.rc_entries
    to_remove  = None

    for idx, entry in enumerate(rc_entries):
        st.markdown(f"**Root Cause #{idx + 1}**")

        desc_val = st.text_area(
            f"_rc_{idx}",
            value=entry["desc"],
            height=80,
            placeholder="e.g. Operator assembled component incorrectly due to unclear SOP…",
            key=f"rc_desc_{idx}",
            label_visibility="collapsed",
        )

        if desc_val != entry["desc"]:
            rc_entries[idx]["desc"]     = desc_val
            rc_entries[idx]["category"] = ""
            rc_entries[idx]["_suggestions"] = get_top3(desc_val) if len(desc_val.strip()) > 8 else []

        # Lazy fetch on first render
        if desc_val and len(desc_val.strip()) > 8 and not entry.get("_suggestions") and st.session_state.model_loaded:
            rc_entries[idx]["_suggestions"] = get_top3(desc_val)

        suggestions = entry.get("_suggestions", [])

        if suggestions:
            st.markdown("**BERTopic Suggestions** — click the best match:")
            scols = st.columns(len(suggestions))
            for si, sug in enumerate(suggestions):
                conf = sug["confidence"]
                badge = "🟢" if conf >= 0.50 else ("🟡" if conf >= 0.35 else "🔴")
                is_sel = entry.get("category") == sug["category"]
                label  = f"{'✓ ' if is_sel else ''}{badge} {sug['category']}  [{conf:.3f}]"
                with scols[si]:
                    if st.button(label, key=f"sug_{idx}_{si}",
                                  use_container_width=True,
                                  type="primary" if is_sel else "secondary"):
                        rc_entries[idx]["category"] = sug["category"]
                        st.rerun()
        elif desc_val and len(desc_val.strip()) > 8 and not st.session_state.model_loaded:
            st.info("Click **Load Model** in the sidebar to get BERTopic suggestions.")

        # Manual override
        all_opts = ["(select manually)"] + ROOT_CAUSE_CATEGORIES
        cur_cat  = entry.get("category", "")
        sel_i    = all_opts.index(cur_cat) if cur_cat in all_opts else 0
        man_cat  = st.selectbox(
            f"Root Cause Category #{idx + 1} (manual override)",
            all_opts, index=sel_i, key=f"rc_cat_{idx}",
        )
        if man_cat != "(select manually)":
            rc_entries[idx]["category"] = man_cat

        if entry.get("category"):
            st.success(f"✅ **{entry['category']}**")

        c_re, c_rat = st.columns([1, 2])
        with c_re:
            re_val = st.radio(
                f"Return or Eliminate #{idx + 1}", RETAIN_ELIMINATE,
                index=RETAIN_ELIMINATE.index(entry.get("retain_eliminate","Retain")),
                key=f"re_{idx}", horizontal=True,
            )
            rc_entries[idx]["retain_eliminate"] = re_val
        with c_rat:
            rat_val = st.text_input(
                f"Rationale #{idx + 1}",
                value=entry.get("rationale",""),
                placeholder="Explain rationale for retaining or eliminating this root cause…",
                key=f"rat_{idx}",
            )
            rc_entries[idx]["rationale"] = rat_val

        if idx > 0:
            if st.button(f"✕ Remove Root Cause #{idx + 1}", key=f"rm_{idx}"):
                to_remove = idx

        if idx < len(rc_entries) - 1:
            st.markdown('<hr style="border-color:#1e2d45;margin:10px 0;">', unsafe_allow_html=True)

    if to_remove is not None:
        rc_entries.pop(to_remove)
        st.rerun()

    st.session_state.rc_entries = rc_entries

    if st.button("➕ Add Root Cause", type="secondary"):
        st.session_state.rc_entries.append(
            {"desc":"","category":"","retain_eliminate":"Retain","rationale":"","_suggestions":[]}
        )
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    # Investigation details
    st.markdown('<div class="card"><div class="card-header">Investigation Details</div>', unsafe_allow_html=True)
    ca, cb = st.columns(2)
    with ca:
        inv_info    = st.text_area("Investigation Information", height=80, key="inv_info")
        containment = st.text_area("Containment", height=80, key="contain_f")
    with cb:
        risk_assess = st.text_area("Risk Assessment", height=80, key="risk_f")
        disposition = st.text_area("CAPA / Disposition Decision", height=80, key="disp_f")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    b1, b2, _ = st.columns([1, 1, 4])
    with b1:
        save_btn  = st.button("💾 Save Record", type="primary", use_container_width=True)
    with b2:
        clear_btn = st.button("🗑️ Clear Form", use_container_width=True)

    if clear_btn:
        st.session_state.rc_entries = [{"desc":"","category":"","retain_eliminate":"Retain","rationale":"","_suggestions":[]}]
        st.rerun()

    if save_btn:
        valid = [e for e in rc_entries if e.get("desc","").strip() and e.get("category","")]
        if not valid:
            st.error("❌ Enter at least one root cause description and select a category.")
        else:
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for i, e in enumerate(valid):
                st.session_state.records.append({
                    "NCR Number":          ncr_num,
                    "Date Initiated":      str(ncr_date),
                    "Initiated By":        initiated_by,
                    "Department":          department,
                    "Status":              ncr_status,
                    "Problem Description": problem_desc,
                    "Root Cause #":        i + 1,
                    "RC Description":      e.get("desc",""),
                    "Root Cause Category": e.get("category",""),
                    "Retain or Eliminate": e.get("retain_eliminate","Retain"),
                    "Rationale":           e.get("rationale",""),
                    "Investigation Info":  inv_info    if i == 0 else "",
                    "Containment":         containment if i == 0 else "",
                    "Risk Assessment":     risk_assess if i == 0 else "",
                    "Disposition / CAPA":  disposition if i == 0 else "",
                    "Timestamp":           ts,
                })
            st.success(f"✅ Saved {len(valid)} root cause(s) for **{ncr_num}**")
            st.session_state.rc_entries = [{"desc":"","category":"","retain_eliminate":"Retain","rationale":"","_suggestions":[]}]
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 Analytics Dashboard":
    st.markdown("## 📊 NCR Root Cause Analytics")

    if not st.session_state.records:
        st.info("No records yet — save some NCR entries first, or load an existing file below.")
        up = st.file_uploader("Load existing records (CSV or Excel)", type=["csv","xlsx"])
        if up:
            df_up = pd.read_csv(up) if up.name.endswith(".csv") else pd.read_excel(up)
            st.session_state.records = df_up.to_dict("records")
            st.rerun()
    else:
        df = pd.DataFrame(st.session_state.records)
        total   = len(df)
        n_ncrs  = df["NCR Number"].nunique()
        n_cats  = df["Root Cause Category"].nunique()
        top_cat = df["Root Cause Category"].value_counts().idxmax()

        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(f'<div class="metric-card"><div class="metric-value">{total}</div><div class="metric-label">Total RC Entries</div></div>', unsafe_allow_html=True)
        m2.markdown(f'<div class="metric-card"><div class="metric-value">{n_ncrs}</div><div class="metric-label">Unique NCRs</div></div>', unsafe_allow_html=True)
        m3.markdown(f'<div class="metric-card"><div class="metric-value">{n_cats}</div><div class="metric-label">Category Types</div></div>', unsafe_allow_html=True)
        short = (top_cat[:40]+"…") if len(top_cat)>40 else top_cat
        m4.markdown(f'<div class="metric-card"><div class="metric-value" style="font-size:0.9rem;padding-top:10px;">{short}</div><div class="metric-label">Top Root Cause</div></div>', unsafe_allow_html=True)

        st.markdown("---")
        try:
            import plotly.express as px
            T    = "plotly_dark"
            BG   = "rgba(0,0,0,0)"
            FONT = dict(family="IBM Plex Sans")

            c1, c2 = st.columns([3, 2])
            with c1:
                st.markdown("**Top 15 Root Cause Categories**")
                vc = df["Root Cause Category"].value_counts().head(15).reset_index()
                vc.columns = ["Category","Count"]
                vc["Label"] = vc["Category"].str[:55]
                fig = px.bar(vc, x="Count", y="Label", orientation="h",
                             color="Count", color_continuous_scale="Blues",
                             template=T, height=420)
                fig.update_layout(yaxis=dict(autorange="reversed"), showlegend=False,
                                  paper_bgcolor=BG, plot_bgcolor=BG,
                                  margin=dict(l=0,r=0,t=10,b=10), font=FONT, yaxis_title="")
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                st.markdown("**Category Group Breakdown**")
                df["Group"] = df["Root Cause Category"].str.split(" : ").str[0]
                gc = df["Group"].value_counts().reset_index()
                gc.columns = ["Group","Count"]
                fig2 = px.pie(gc, names="Group", values="Count", template=T,
                              color_discrete_sequence=px.colors.sequential.Blues_r, height=420)
                fig2.update_layout(paper_bgcolor=BG, font=FONT, margin=dict(l=0,r=0,t=10,b=10))
                st.plotly_chart(fig2, use_container_width=True)

            if "Timestamp" in df.columns:
                st.markdown("**NCR Volume Trend**")
                df["Date"] = pd.to_datetime(df["Timestamp"]).dt.date
                trend = df.groupby("Date").size().reset_index(name="Count")
                fig3 = px.line(trend, x="Date", y="Count", markers=True, template=T, height=240)
                fig3.update_traces(line_color="#60a8f8")
                fig3.update_layout(paper_bgcolor=BG, plot_bgcolor=BG, font=FONT,
                                   margin=dict(l=0,r=0,t=10,b=10))
                st.plotly_chart(fig3, use_container_width=True)

            c3, c4 = st.columns(2)
            with c3:
                st.markdown("**Retain vs. Eliminate**")
                re_c = df["Retain or Eliminate"].value_counts().reset_index()
                re_c.columns = ["Decision","Count"]
                fig4 = px.bar(re_c, x="Decision", y="Count", color="Decision",
                              color_discrete_map={"Retain":"#4a7ab8","Eliminate":"#4ade80"},
                              template=T, height=260)
                fig4.update_layout(showlegend=False, paper_bgcolor=BG, plot_bgcolor=BG,
                                   margin=dict(l=0,r=0,t=10,b=10), font=FONT)
                st.plotly_chart(fig4, use_container_width=True)

            with c4:
                st.markdown("**NCR Status**")
                sc = df.drop_duplicates("NCR Number")["Status"].value_counts().reset_index()
                sc.columns = ["Status","Count"]
                fig5 = px.bar(sc, x="Status", y="Count", color="Status",
                              template=T, height=260)
                fig5.update_layout(showlegend=False, paper_bgcolor=BG, plot_bgcolor=BG,
                                   margin=dict(l=0,r=0,t=10,b=10), font=FONT)
                st.plotly_chart(fig5, use_container_width=True)

        except ImportError:
            st.warning("Install plotly for charts: `pip install plotly`")

        st.markdown("---")
        st.markdown("**📋 Descriptive Summary**")
        elim_pct = f"{(df['Retain or Eliminate']=='Eliminate').sum()/max(total,1)*100:.1f}%"
        stats = pd.DataFrame({
            "Metric": [
                "Total Root Cause Entries",
                "Unique NCR Numbers",
                "Unique Categories Used",
                "Most Frequent Category",
                "Least Frequent Category",
                "Avg Root Causes per NCR",
                "Entries — Eliminate Decision",
            ],
            "Value": [
                total, n_ncrs, n_cats, top_cat,
                df["Root Cause Category"].value_counts().idxmin(),
                f"{total/max(n_ncrs,1):.1f}",
                elim_pct,
            ]
        })
        st.dataframe(stats, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — RECORDS BROWSER & EXPORT
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📁 Records Browser":
    st.markdown("## 📁 Saved NCR Records")

    up = st.file_uploader("📂 Load existing records file (CSV or Excel)", type=["csv","xlsx"])
    if up:
        df_up = pd.read_csv(up) if up.name.endswith(".csv") else pd.read_excel(up)
        action = st.radio("Action", ["Append to existing records","Replace existing records"], horizontal=True)
        if st.button("Load File"):
            if "Replace" in action:
                st.session_state.records = df_up.to_dict("records")
            else:
                st.session_state.records += df_up.to_dict("records")
            st.success(f"Loaded {len(df_up)} records.")
            st.rerun()

    if not st.session_state.records:
        st.info("No records yet.")
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
        st.markdown("### 💾 Export Records")
        e1, e2, e3 = st.columns(3)

        with e1:
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as w:
                fdf.to_excel(w, sheet_name="NCR_Records", index=False)
                fdf["Root Cause Category"].value_counts().rename_axis("Category").reset_index(
                    name="Count"
                ).to_excel(w, sheet_name="Category_Summary", index=False)
            st.download_button(
                "📥 Download Excel (.xlsx)", data=buf.getvalue(),
                file_name=f"NCR_Records_{datetime.date.today()}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

        with e2:
            st.download_button(
                "📥 Download CSV (.csv)",
                data=fdf.to_csv(index=False).encode("utf-8"),
                file_name=f"NCR_Records_{datetime.date.today()}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with e3:
            html_doc = f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<style>body{{font-family:Arial,sans-serif;font-size:11px;margin:20px}}
h2{{color:#1a2744}}table{{border-collapse:collapse;width:100%}}
th{{background:#1a2744;color:white;padding:6px 8px;text-align:left}}
td{{border:1px solid #ccc;padding:5px 8px}}tr:nth-child(even){{background:#f0f4f8}}
</style></head><body>
<h2>NCR Root Cause Records — {datetime.date.today()}</h2>
{fdf.to_html(index=False,border=0)}
</body></html>"""
            st.download_button(
                "📥 Download PDF-ready HTML",
                data=html_doc.encode("utf-8"),
                file_name=f"NCR_Records_{datetime.date.today()}.html",
                mime="text/html",
                use_container_width=True,
                help="Open in browser → Ctrl+P → Save as PDF",
            )

        st.markdown("---")
        if st.button("🗑️ Clear All Records"):
            st.session_state.records = []
            st.rerun()
