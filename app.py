import streamlit as st
import pandas as pd
from datetime import datetime
import json

# Page Configuration
st.set_page_config(
    page_title="NCR Classification",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
        .header {
            text-align: center;
            padding: 30px 0;
            border-bottom: 3px solid #dc3545;
            margin-bottom: 30px;
        }
        .header h1 {
            color: #1a1a1a;
            margin: 0;
            font-size: 2.5em;
        }
        .header p {
            color: #999;
            font-size: 1em;
            margin: 10px 0 0 0;
        }
        .section-title {
            border-bottom: 2px solid #dc3545;
            padding-bottom: 10px;
            margin-top: 30px;
            margin-bottom: 20px;
        }
        .success-box {
            background: #f0fff4;
            border-left: 4px solid #22c55e;
            padding: 15px;
            border-radius: 4px;
            margin: 20px 0;
        }
        .info-box {
            background: #fff5f5;
            border-left: 4px solid #dc3545;
            padding: 15px;
            border-radius: 4px;
            margin: 20px 0;
        }
        .error-box {
            background: #fef2f2;
            border-left: 4px solid #ef4444;
            padding: 15px;
            border-radius: 4px;
            margin: 20px 0;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'records' not in st.session_state:
    st.session_state.records = []

# Header
st.markdown("""
    <div class="header">
        <h1>🔍 NCR Classification</h1>
        <p>Root Causes & Corrective Actions</p>
    </div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["Root Causes", "Corrective Actions", "Saved Records"])

# ============================================================================
# TAB 1: ROOT CAUSES
# ============================================================================
with tab1:
    st.markdown("### Classify a Root Cause")
    
    st.markdown("""
        <div class="info-box">
            <strong>💡 Tip:</strong> Provide detailed descriptions for more accurate categorization and better insights.
        </div>
    """, unsafe_allow_html=True)
    
    # Root Cause Input
    root_cause_description = st.text_area(
        "Root Cause Description",
        placeholder="e.g., Operator assembled component incorrectly due to unclear work instruction...",
        height=150,
        key="root_cause_input"
    )
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        classify_button = st.button("🔍 Classify & Analyze", use_container_width=True)
    with col2:
        st.empty()
    
    # Placeholder for classification results
    if classify_button:
        if root_cause_description.strip():
            with st.spinner("Classifying..."):
                # Simulate classification (replace with actual model)
                st.markdown("""
                    <div class="success-box">
                        <strong>✅ Classification Successful!</strong><br>
                        BERTopic has suggested the following categories:
                    </div>
                """, unsafe_allow_html=True)
                
                # Display suggested categories
                categories_data = {
                    "Category": ["Process Error", "Human Error", "Equipment Failure"],
                    "Confidence": [0.92, 0.85, 0.78]
                }
                
                df_categories = pd.DataFrame(categories_data)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🏆 Top Category", "Process Error", "92%")
                with col2:
                    st.metric("🥈 Second", "Human Error", "85%")
                with col3:
                    st.metric("🥉 Third", "Equipment Failure", "78%")
                
                # Category table
                st.dataframe(df_categories, use_container_width=True, hide_index=True)
                
                # Store for corrective action
                st.session_state.last_classification = {
                    "description": root_cause_description,
                    "category": "Process Error"
                }
        else:
            st.warning("⚠️ Please enter a root cause description to classify.")

# ============================================================================
# TAB 2: CORRECTIVE ACTIONS
# ============================================================================
with tab2:
    st.markdown("### Define Corrective Actions")
    
    st.markdown("""
        <div class="info-box">
            <strong>💡 Tip:</strong> Define clear, measurable corrective actions to address the root cause.
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        root_cause_text = st.text_input(
            "Root Cause (from classification)",
            value=st.session_state.last_classification.get("description", "") if 'last_classification' in st.session_state else "",
            disabled=True
        )
    
    with col2:
        category = st.selectbox(
            "Category",
            ["Process Error", "Human Error", "Equipment Failure", "Other"],
            index=0 if 'last_classification' not in st.session_state else ["Process Error", "Human Error", "Equipment Failure", "Other"].index(st.session_state.last_classification.get("category", "Process Error"))
        )
    
    corrective_action = st.text_area(
        "Corrective Action Description",
        placeholder="Describe the corrective action to be taken...",
        height=150
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        save_button = st.button("💾 Save Action", use_container_width=True)
    
    if save_button:
        if root_cause_text and corrective_action:
            record = {
                "id": f"NCR_{len(st.session_state.records) + 1:03d}",
                "root_cause": root_cause_text,
                "category": category,
                "action": corrective_action,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status": "Open"
            }
            
            st.session_state.records.append(record)
            
            st.markdown("""
                <div class="success-box">
                    <strong>✅ Corrective Action Saved!</strong><br>
                    NCR ID: {}<br>
                    Timestamp: {}
                </div>
            """.format(record["id"], record["timestamp"]), unsafe_allow_html=True)
            
            st.success(f"Saved as {record['id']}")
        else:
            st.error("⚠️ Please fill in all fields before saving.")

# ============================================================================
# TAB 3: SAVED RECORDS
# ============================================================================
with tab3:
    st.markdown("### Saved NCR Records")
    
    if st.session_state.records:
        # Filter options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            filter_category = st.selectbox(
                "Filter by Category",
                ["All"] + list(set([r["category"] for r in st.session_state.records]))
            )
        
        with col2:
            filter_status = st.selectbox(
                "Filter by Status",
                ["All", "Open", "Closed"]
            )
        
        with col3:
            st.empty()
        
        # Filter records
        filtered_records = st.session_state.records
        
        if filter_category != "All":
            filtered_records = [r for r in filtered_records if r["category"] == filter_category]
        
        if filter_status != "All":
            filtered_records = [r for r in filtered_records if r["status"] == filter_status]
        
        # Display records
        if filtered_records:
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total NCRs", len(st.session_state.records))
            with col2:
                st.metric("Open", len([r for r in st.session_state.records if r["status"] == "Open"]))
            with col3:
                st.metric("Closed", len([r for r in st.session_state.records if r["status"] == "Closed"]))
            with col4:
                st.metric("Filtered", len(filtered_records))
            
            st.markdown("---")
            
            # Display as dataframe
            df_records = pd.DataFrame(filtered_records)
            
            # Reorder columns
            df_records = df_records[["id", "root_cause", "category", "status", "timestamp"]]
            
            st.dataframe(
                df_records,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "id": st.column_config.TextColumn("NCR ID", width="medium"),
                    "root_cause": st.column_config.TextColumn("Root Cause", width="large"),
                    "category": st.column_config.TextColumn("Category", width="medium"),
                    "status": st.column_config.TextColumn("Status", width="small"),
                    "timestamp": st.column_config.TextColumn("Timestamp", width="medium")
                }
            )
            
            # Expand for details
            st.markdown("### Record Details")
            
            selected_id = st.selectbox(
                "Select a record to view details",
                [r["id"] for r in filtered_records]
            )
            
            selected_record = next(r for r in filtered_records if r["id"] == selected_id)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.text_input("Root Cause", value=selected_record["root_cause"], disabled=True)
                st.text_input("Category", value=selected_record["category"], disabled=True)
            
            with col2:
                st.text_input("Status", value=selected_record["status"], disabled=True)
                st.text_input("Timestamp", value=selected_record["timestamp"], disabled=True)
            
            st.text_area(
                "Corrective Action",
                value=selected_record["action"],
                disabled=True,
                height=120
            )
            
            # Export option
            st.markdown("---")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv = df_records.to_csv(index=False)
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name=f"ncr_records_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                json_data = json.dumps(filtered_records, indent=2)
                st.download_button(
                    label="📥 Download as JSON",
                    data=json_data,
                    file_name=f"ncr_records_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )
        
        else:
            st.info("No records match the selected filters.")
    
    else:
        st.info("📝 No NCR records saved yet. Go to 'Corrective Actions' to create one.")

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    st.markdown("---")
    
    # Clear all data option
    if st.button("🗑️ Clear All Records", use_container_width=True):
        st.session_state.records = []
        st.success("All records cleared!")
        st.rerun()
    
    st.markdown("---")
    
    st.markdown("### 📊 Statistics")
    
    if st.session_state.records:
        total_ncrs = len(st.session_state.records)
        categories = [r["category"] for r in st.session_state.records]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total NCRs", total_ncrs)
        with col2:
            st.metric("Unique Categories", len(set(categories)))
        
        # Category breakdown
        st.markdown("**Category Breakdown:**")
        category_counts = pd.Series(categories).value_counts()
        st.bar_chart(category_counts)
    
    st.markdown("---")
    
    st.markdown("### 📖 Help")
    st.info("""
        **How to use:**
        
        1. **Root Causes** - Enter and classify root causes
        2. **Corrective Actions** - Define actions for each root cause
        3. **Saved Records** - View and manage all NCR records
        
        Data is stored in your session and will be cleared when you refresh.
    """)
    
    st.markdown("---")
    
    st.markdown("**Version:** 1.0.0")
