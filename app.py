import streamlit as st
import pandas as pd
from groq import Groq
import PyPDF2
import io
import time
import base64
import random
import re
import plotly.express as px
import plotly.graph_objects as go

# ==========================================
# 0. CORE PERFORMANCE & SECURITY CONFIG
# ==========================================
st.set_page_config(
    page_title="InsightVibe AI Enterprise Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Robust Secret Key Check (Always check keys FIRST)
if "GROQ_API_KEY" not in st.secrets:
    st.error("⚠️ CRITICAL ERROR: GROQ_API_KEY is not defined in `.streamlit/secrets.toml`. "
             "The application cannot function without this key.")
    st.markdown("""
        ### Setup Instructions:
        1. Create a `.streamlit` folder in your project root.
        2. Create a `secrets.toml` file inside it.
        3. Add this line (with your actual key):
        `GROQ_API_KEY = "gsk_xxxx..." `
    """)
    st.stop()

# Initialize Groq client
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# Global Model Definitions - Keep these up-to-date!
MODEL_PREMIUM = "llama-3.3-70b-versatile"
MODEL_FAST = "llama-3.1-8b-instant"

# Context Length Management (Important for LLM stability)
CONTEXT_THRESHOLD_CHARS = 15000 


# ==========================================
# 1. PREMIUM UI: GLOBAL THEME & VISUALS
# ==========================================
def load_global_theme():
    """
    Sets up the modern, vibrant dark theme. 
    Overrides Streamlit's default components with custom CSS.
    """
    st.markdown("""
    <style>
        /* Base Page Setup */
        .main {
            background: linear-gradient(135deg, #07090f 0%, #0c111d 50%, #07090f 100%);
            color: #ececec;
            font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        }

        /* Sidebar Styling: Clean & Modern */
        [data-testid="stSidebar"] {
            background-color: #07090f;
            border-right: 1px solid #1f2937;
        }
        
        /* Premium Main Header (Neon Glow) */
        .premium-header {
            font-size: 3rem;
            font-weight: 800;
            background: linear-gradient(90deg, #00ffff, #0099ff, #00ffff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
            text-transform: uppercase;
            letter-spacing: 2px;
            text-shadow: 0 0 15px rgba(0, 255, 255, 0.4);
        }

        /* Sub-header Caption */
        .premium-caption {
            font-size: 1.1rem;
            color: #8da0b6;
            margin-top: -10px;
            margin-bottom: 2rem;
        }

        /* Styled Containers / Boxes */
        .st-emotion-cache-16ids9v, .st-emotion-cache-zt5igj, [data-testid="stExpander"] {
            background-color: rgba(12, 17, 29, 0.7) !important;
            border-radius: 12px !important;
            border: 1px solid #1f2937 !important;
            padding: 1.5rem !important;
        }

        /* Metrics (Dynamic stats) */
        [data-testid="stMetricValue"] {
            color: #00ffff !important;
            font-size: 2.2rem !important;
        }
        
        /* Global Button Override */
        .stButton>button {
            background: linear-gradient(90deg, #00ccff, #00ffff);
            color: #000;
            font-weight: 700;
            border-radius: 20px;
            border: none;
            transition: all 0.3s ease;
            text-transform: uppercase;
        }
        
        .stButton>button:hover {
            box-shadow: 0 0 10px rgba(0, 255, 255, 0.7);
            transform: translateY(-2px);
            color: #000;
        }

        /* Chat Input Bar (Top AI Look) */
        .stChatInputContainer {
            background-color: rgba(12, 17, 29, 0.9) !important;
            border: 1px solid #334155 !important;
            border-radius: 30px !important;
        }
        
        /* Fix the alignment and display issues */
        [data-testid="stVerticalBlock"] > div:has(div.premium-header) {
            margin-bottom: 0px !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # Custom Header Component
    st.markdown('<p class="premium-header">INSIGHTVIBE AI</p>', unsafe_allow_html=True)
    st.markdown('<p class="premium-caption">Advanced Multimodal Analysis Engine & Enterprise Intelligence Dashboard</p>', unsafe_allow_html=True)

# Function to serve the lively background animation
def load_lively_background():
    """
    Applies a dynamic, lively CSS particle background.
    """
    st.markdown("""
        <style>
        .background-container {
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background: radial-gradient(circle, #0a0e17 0%, #030509 100%);
            z-index: -2;
            overflow: hidden;
        }
        .particle {
            position: absolute;
            width: 3px;
            height: 3px;
            background: rgba(0, 255, 255, 0.4);
            border-radius: 50%;
            animation: float 20s infinite linear;
            z-index: -1;
        }
        @keyframes float {
            0% { transform: translateY(100vh) translateX(0vw); }
            100% { transform: translateY(-10vh) translateX(90vw); }
        }
        </style>
        <div class="background-container">
            <div class="particle" style="left: 10%; animation-duration: 25s; width: 4px; height: 4px;"></div>
            <div class="particle" style="left: 20%; animation-duration: 18s;"></div>
            <div class="particle" style="left: 35%; animation-duration: 22s; background: rgba(0, 153, 255, 0.3);"></div>
            <div class="particle" style="left: 50%; animation-duration: 15s;"></div>
            <div class="particle" style="left: 65%; animation-duration: 28s; width: 5px; height: 5px;"></div>
            <div class="particle" style="left: 80%; animation-duration: 20s;"></div>
            <div class="particle" style="left: 90%; animation-duration: 21s; background: rgba(0, 153, 255, 0.3);"></div>
        </div>
    """, unsafe_allow_html=True)


# ==========================================
# 2. STATE MANAGEMENT & HELPER FUNCTIONS
# ==========================================
def initialize_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "extracted_data_raw" not in st.session_state:
        st.session_state.extracted_data_raw = ""
    if "extracted_data_filename" not in st.session_state:
        st.session_state.extracted_data_filename = ""
    if "csv_dataframe" not in st.session_state:
        st.session_state.csv_dataframe = None
    if "data_insight_vibe" not in st.session_state:
        # Structured auto-analysis
        st.session_state.data_insight_vibe = {}

def get_file_type(uploaded_file):
    if uploaded_file.name.lower().endswith('.pdf'):
        return "PDF"
    elif uploaded_file.name.lower().endswith('.csv'):
        return "CSV"
    return "UNKNOWN"

# ==========================================
# 3. DATA PROCESSING ENGINES
# ==========================================
def process_pdf_core(uploaded_file):
    """Parses a PDF file and returns its raw text."""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
        raw_text = ""
        for page in pdf_reader.pages:
            raw_text += page.extract_text() or ""
        return raw_text
    except Exception as e:
        st.error(f"⚠️ Error Processing PDF: {str(e)}")
        return ""

def process_csv_core(uploaded_file):
    """Reads a CSV file, creates a summary, and stores the dataframe."""
    try:
        df = pd.read_csv(uploaded_file)
        # Store full dataframe for the advanced visualization engine
        st.session_state.csv_dataframe = df
        
        # Create a structural summary for the LLM
        summary = f"DATASET IDENTITY: {uploaded_file.name}\n"
        summary += f"DIMENSIONS: {df.shape[0]} Rows, {df.shape[1]} Columns.\n"
        summary += f"COLUMNS MAP: {list(df.columns)}\n"
        summary += f"SAMPLE (Head 3):\n{df.head(3).to_string()}"
        return summary
    except Exception as e:
        st.error(f"⚠️ Error Processing CSV: {str(e)}")
        return ""

# ==========================================
# 4. LLM ANALYTICS ENGINE (GROQ)
# ==========================================
def query_groq_streaming(user_prompt, context, system_context, model):
    """Handles standard query-answer streaming flow."""
    try:
        # Trim context if it's too long
        trimmed_context = context[:CONTEXT_THRESHOLD_CHARS]
        if len(context) > CONTEXT_THRESHOLD_CHARS:
            trimmed_context += "... [TEXT TRUNCATED FOR CONTEXT LIMIT]"
        
        response_stream = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_context},
                {"role": "system", "content": f"DATASET CONTEXT:\n\n{trimmed_context}"},
                *st.session_state.chat_history, # Persistent chat flow
                {"role": "user", "content": user_prompt}
            ],
            stream=True,
            temperature=0.5,
        )
        return response_stream
    except Exception as e:
        st.error(f"⚠️ Error Contacting Groq: {str(e)}")
        return None

# Specialized LLM-Powered Analysis
def generate_auto_analysis(context, file_type):
    """
    Requests a structured, multi-component analysis from the LLM based on file type.
    """
    if not context or st.session_state.data_insight_vibe:
        return # Avoid redundant runs

    model = MODEL_PREMIUM
    progress_bar = st.progress(0)
    status_text = st.empty()

    analysis_prompts = {
        "PDF": {
            "summary": "Create a professional, high-level summary (3-4 paragraphs) of this document.",
            "metrics": "Extract 5 key quantitative data points/metrics/dates (comma-separated, format: Key Value, e.g., 'Revenue: $5M, Date: 2023').",
            "entities": "Identify the top 10 key entities (Organizations, People, Locations, Topics) (comma-separated).",
            "actions": "Extract 5 proposed or clear Action Items/Next Steps mentioned in the text (comma-separated)."
        },
        "CSV": {
            "summary": "Explain the structure of this dataset, what it represents, and list important observations about its contents (3 paragraphs).",
            "metrics": "Extract 5 meaningful aggregate statistical insights (e.g., 'Avg Sales: $5000, Max Units: 200'). comma-separated.",
            "trends": "Based on the columns provided, suggest 3 significant trends or patterns the user should look for.",
            "cleanup": "Identify 3 potential data cleaning tasks or issues (e.g., 'Column Price has 50 nulls, Category column has duplicate labels')."
        }
    }

    vibe_results = {}
    total_tasks = len(analysis_prompts[file_type])
    current_task = 0

    trimmed_context = context[:CONTEXT_THRESHOLD_CHARS]

    # Optimized One-Shot Structural Extraction (More efficient than loop)
    consolidated_prompt = "You are a master systems analyst. Perform a mandatory structured extraction from this text. \n"
    consolidated_prompt += "Ensure your output is structured exactly with these headers and nothing else:\n"
    for k, v in analysis_prompts[file_type].items():
        consolidated_prompt += f"@@{k.upper()}: {v}\n"

    status_text.write("⚡ Executing Auto-Extraction Core...")
    progress_bar.progress(30)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Master Analyst. Mandatory structured extraction only. Use delimiter @@KEY:"},
                {"role": "system", "content": f"DATA:\n\n{trimmed_context}"},
                {"role": "user", "content": consolidated_prompt}
            ],
            temperature=0.2 # Keep it consistent and analytical
        )
        content = response.choices[0].message.content
        progress_bar.progress(80)

        # Parse the custom delimited output
        for key in analysis_prompts[file_type].keys():
            regex_pattern = f"@@{key.upper()}:\s*([\s\S]+?)(?=(@@|$))"
            match = re.search(regex_pattern, content)
            if match:
                extracted_value = match.group(1).strip()
                vibe_results[key] = extracted_value
            else:
                vibe_results[key] = f"[Extraction Failure for {key}]"
        
        progress_bar.progress(100)
        status_text.empty()
        progress_bar.empty()
        
    except Exception as e:
        status_text.write(f"⚠️ Error in Extraction: {str(e)}")
        progress_bar.progress(100)
        vibe_results = {k: "[ANALYSIS ERROR]" for k in analysis_prompts[file_type].keys()}

    st.session_state.data_insight_vibe = vibe_results


# ==========================================
# 5. SIDEBAR: DATA INGESTION & CONFIG
# ==========================================
def render_premium_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center;">
            <p style="font-size: 1.5rem; font-weight: 800; color: #00ffff; margin-bottom: 0;">⚡ DATA CONTROL</p>
            <p style="font-size: 0.9rem; color: #8da0b6; margin-top: 0;">Enterprise Knowledge Gateway</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Optimized File Uploader
        new_uploaded_file = st.file_uploader("Upload Data Vector (PDF, CSV)", type=["pdf", "csv"], key="primary_uploader")
        
        # Advanced Model Selection (Pre-filtered IDs)
        selected_engine = st.selectbox("AI Core Engine", [MODEL_PREMIUM, MODEL_FAST], index=0)
        
        if st.button("Generate One-Click Executive Vibe"):
             if st.session_state.extracted_data_raw:
                 # Manually trigger chat
                 sys_p = "Systems Analyst. Focus on structural impact."
                 user_p = "Give me a 'One-Click Executive Vibe' summary of the whole dataset with 3 critical impacts and 3 action items."
                 
                 st.session_state.chat_history.append({"role": "user", "content": user_p})
                 
                 # Placeholder for immediate UI feedback
                 completion = query_groq_streaming(user_p, st.session_state.extracted_data_raw, sys_p, selected_engine)
                 
                 resp_content = ""
                 if completion:
                     for chunk in completion:
                         if chunk.choices[0].delta.content: resp_content += chunk.choices[0].delta.content
                     
                     st.session_state.chat_history.append({"role": "assistant", "content": resp_content})
                     st.rerun() # Refresh to show chat
                 else:
                     st.error("Engine failure.")

        # Logic: If a new file is uploaded, process it instantly.
        if new_uploaded_file and new_uploaded_file.name != st.session_state.extracted_data_filename:
            with st.spinner("Executing Data Extraction Protocols..."):
                initialize_session_state() # Fresh state for new file
                
                ft = get_file_type(new_uploaded_file)
                st.session_state.extracted_data_filename = new_uploaded_file.name
                
                if ft == "PDF":
                    raw_text = process_pdf_core(new_uploaded_file)
                    st.session_state.extracted_data_raw = raw_text
                elif ft == "CSV":
                    matrix_summary = process_csv_core(new_uploaded_file)
                    # For CSV, the LLM context is the summary, not the raw text
                    st.session_state.extracted_data_raw = matrix_summary
                
                st.session_state.data_insight_vibe = {} # Reset vibe
                st.success(f"SUCCESS: {ft} Vector Mapped!")
                st.rerun()

        st.markdown("---")
        st.markdown("""
        <div style="font-size: 0.8rem; color: #506680; text-align: center;">
            InsightVibe AI v2.1 Platinum | Groq Compute Layer<br>
            Developer Node: AIML Year 3
        </div>
        """, unsafe_allow_html=True)
        return selected_engine


# ==========================================
# 6. MAIN WORKSPACE: ANALYTICS & CHAT
# ==========================================
def render_main_workspace(active_engine):
    
    # 6.1 State Verification (If no file is uploaded)
    if not st.session_state.extracted_data_raw:
        st.info("⚡ Awaiting Data Vector Ingestion. Please upload a PDF or CSV asset using the 'Data Control' panel in the sidebar.")
        
        # Show some cool sample stats to fill space
        col1, col2, col3 = st.columns(3)
        col1.metric("API Compute Nodes", "40", "+1")
        col2.metric("Compute Latency", "12ms", "-3%")
        col3.metric("Data Vector Status", "AWAITING", "...")
        st.stop()

    # Determine Active Vector Type
    active_filename = st.session_state.extracted_data_filename
    is_csv = active_filename.lower().endswith('.csv')
    ft = "CSV MATRIX" if is_csv else "PDF VECTOR"
    
    # Run the One-Time Auto-Analysis
    generate_auto_analysis(st.session_state.extracted_data_raw, "CSV" if is_csv else "PDF")
    vibe = st.session_state.data_insight_vibe

    # 6.2 Workspace Layout (Vibrant Dark Aesthetic)
    col1, col2 = st.columns([2, 1]) # 2/3 and 1/3 split

    with col1:
        # TABS: Clean navigation between analysis and chat
        tab_vibe, tab_raw = st.tabs(["⚡ AI-POWERED AUTO-INSIGHTS", "📋 RAW EXTRACTED VECTOR"])
        
        with tab_vibe:
            st.markdown(f"<h3 style='color:#00ffff;'>{vibe.get('metrics', '...')}</h3>", unsafe_allow_html=True)
            st.markdown(f"<p style='color:#8da0b6; font-size:1.1rem;'><b>Executive Vibe:</b> {vibe.get('summary', 'Running Protocols...')}</p>", unsafe_allow_html=True)
            
            # Show advanced structured data
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("---")
                if is_csv:
                    st.markdown("#### Suggested Trends & Patterns")
                    trends_str = vibe.get('trends', '')
                    for t in trends_str.split(','): st.write(f"📈 {t.strip()}")
                else:
                    st.markdown("#### Primary Key Entities (Orgs, People, etc.)")
                    entities_str = vibe.get('entities', '')
                    st.markdown(" ".join([f"<span style='background-color:#1f2937; color:#00ffff; border-radius:15px; padding: 5px 12px; margin: 3px; font-size:0.9rem; font-weight:700;'>{e.strip()}</span>" for e in entities_str.split(',')]), unsafe_allow_html=True)

            with col_b:
                st.markdown("---")
                if is_csv:
                    st.markdown("#### Data Cleanup & Optimization Tasks")
                    cleanup_str = vibe.get('cleanup', '')
                    for c in cleanup_str.split(','): st.write(f"🛠️ {c.strip()}")
                else:
                    st.markdown("#### Action Items & Strategic Next Steps")
                    actions_str = vibe.get('actions', '')
                    st.markdown(" ".join([f"<p style='color:#00ffff;'>⚡ {a.strip()}</p>" for a in actions_str.split(',')]), unsafe_allow_html=True)

        with tab_raw:
             if not is_csv:
                st.text_area("Live Text Matrix Inspection", st.session_state.extracted_data_raw, height=500)
             else:
                st.markdown("### Structural CSV Matrix Head (3 Sample Rows)")
                st.dataframe(st.session_state.csv_dataframe.head(3), use_container_width=True)
                st.markdown("---")
                st.text_area("Raw Matrix Summary provided to LLM", st.session_state.extracted_data_raw, height=300)

    # 6.3 ADVANCED DATA VISUALIZATION ENGINE (CSV ONLY)
    if is_csv and st.session_state.csv_dataframe is not None:
        with col2:
            df = st.session_state.csv_dataframe
            
            # Extract usable columns (Numerical vs Categorical)
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            st.markdown("""
            <p style="font-size: 1.5rem; font-weight: 800; color: #00ffff; margin-bottom: 0;">📊 ADVANCED VIZ ENGINE</p>
            """, unsafe_allow_html=True)
            
            if not numeric_cols:
                st.info("Visualization engine disabled: No numerical data found in CSV matrix.")
            else:
                st.markdown("---")
                chart_type = st.radio("Chart Type", ["📈 Trend Line", "📊 Distribution Bar", "🔘 Scatter Matrix"], horizontal=True, key="viz_chart_type")
                
                # Viz Selection Controls
                if chart_type != "🔘 Scatter Matrix":
                    x_axis = st.selectbox("X-Axis (Variable)", df.columns, key="viz_x_axis")
                    y_axis = st.selectbox("Y-Axis (Value)", numeric_cols, key="viz_y_axis")
                else:
                    st.info("Scatter Matrix generates relationship grids. This might take computational cycles for large datasets.")
                    sc_x = st.selectbox("X-Axis", numeric_cols, key="viz_scatter_x")
                    sc_y = st.selectbox("Y-Axis", numeric_cols, key="viz_scatter_y")
                    
                color_by = st.selectbox("Color / Segment By (Optional)", [None] + cat_cols, key="viz_color")
                
                # EXECUTE VIZ GENERATION
                try:
                    if chart_type == "📈 Trend Line":
                        fig = px.line(df, x=x_axis, y=y_axis, color=color_by, template="plotly_dark", 
                                      title=f"TREND PROTOCOL: {y_axis} by {x_axis}")
                        fig.update_layout(xaxis_title=x_axis, yaxis_title=y_axis)
                        st.plotly_chart(fig, use_container_width=True)
                    elif chart_type == "📊 Distribution Bar":
                         fig = px.bar(df, x=x_axis, y=y_axis, color=color_by, template="plotly_dark", barmode='group',
                                       title=f"DISTRIBUTION ANALYSIS: {y_axis} aggregated by {x_axis}")
                         fig.update_layout(xaxis_title=x_axis, yaxis_title=y_axis)
                         st.plotly_chart(fig, use_container_width=True)
                    elif chart_type == "🔘 Scatter Matrix":
                         fig = px.scatter(df, x=sc_x, y=sc_y, color=color_by, template="plotly_dark",
                                          title=f"RELATIONSHIP MAP: {sc_x} vs {sc_y}")
                         fig.update_layout(xaxis_title=sc_x, yaxis_title=sc_y)
                         st.plotly_chart(fig, use_container_width=True)
                except Exception as viz_e:
                    st.error(f"⚠️ Visualization Engine Error: {str(viz_e)}")

    # 6.4 PREMIUM CHAT INTERFACE (LOWER THIRD)
    st.markdown("---")
    st.markdown("""
    <p style="font-size: 1.5rem; font-weight: 800; color: #00ffff; margin-bottom: 0;">💬 CONTEXTUAL INTERROGATION NETWORK</p>
    """, unsafe_allow_html=True)
    
    # 6.4.1 Display Persistent Chat History
    chat_col_left, chat_col_right = st.columns([1, 15]) # Tiny profile column, large message column

    # Function to apply custom styles to chat entries based on type
    def render_message(role, content):
        if role == "user":
            st.markdown(f"""
            <div style="background-color: #1f2937; border-radius: 20px; padding: 15px 20px; margin-bottom: 10px; width: fit-content; max-width: 80%; align-self: flex-end; border: 1px solid #334155;">
                <p style="margin:0; color: #ececec;"><b>You:</b> {content}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background-color: #0c111d; border-radius: 20px; padding: 15px 20px; margin-bottom: 10px; border: 1px solid #1f2937;">
                <p style="margin:0; color: #00ffff;"><b>INSIGHTVibe AI:</b> {content}</p>
            </div>
            """, unsafe_allow_html=True)

    # Output messages
    for message in st.session_state.chat_history:
        render_message(message["role"], message["content"])

    # 6.4.2 PREMIUM CHAT INPUT BAR (Crazy lively display)
    if user_query := st.chat_input("Submit Interrogation Query (PDF/CSV context)..."):
        # 1. Immediate UI Feedback
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        render_message("user", user_query)
        
        # 2. Executing Streaming Query
        system_p = "You are InsightVibe AI. Use the supplied data context flawlessly to answer the user inquiry."
        completion = query_groq_streaming(user_query, st.session_state.extracted_data_raw, system_p, active_engine)
        
        # 3. Handling Streamed Response in real-time
        if completion:
             with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                
                # Streaming loop
                for chunk in completion:
                    if chunk.choices[0].delta.content is not None:
                        full_response += chunk.choices[0].delta.content
                        # Apply lively typing effect visual
                        message_placeholder.markdown(full_response + "▌")
                
                # Final output without the caret
                message_placeholder.markdown(full_response)
             
             # 4. Finalize state
             st.session_state.chat_history.append({"role": "assistant", "content": full_response})
             st.rerun() # Refresh to solidify UI

# ==========================================
# MAIN EXECUTION CORE
# ==========================================
if __name__ == "__main__":
    load_global_theme()
    load_lively_background()
    initialize_session_state()
    active_model = render_premium_sidebar()
    render_main_workspace(active_model)