import streamlit as st
import whisper
import google.generativeai as genai
import tempfile
import json
import os
import sqlite3
from sqlalchemy import create_engine, Column, Integer, String, Text, Float
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import pandas as pd
import plotly.express as px

# ==========================================
# 1. CONFIGURATION & DATABASE SETUP
# ==========================================
st.set_page_config(page_title="AI Meeting Minutes Generator", layout="wide")

# Setup SQLite Database (Creates a local file 'meetings.db')
Base = declarative_base()
engine = create_engine('sqlite:///meetings.db')
Session = sessionmaker(bind=engine)
session = Session()

class Meeting(Base):
    __tablename__ = 'meetings'
    id = Column(Integer, primary_key=True)
    filename = Column(String)
    upload_time = Column(String)
    transcript = Column(Text)
    summary = Column(Text)
    action_items = Column(Text)  # Stored as JSON string
    efficiency_score = Column(Float)
    bias_detected = Column(String)

# Create the database tables if they don't exist
Base.metadata.create_all(engine)

# ==========================================
# 2. AI PROCESSING FUNCTIONS
# ==========================================

@st.cache_resource
def load_whisper_model(size):
    """Load and cache the Whisper model to avoid reloading on every run."""
    return whisper.load_model(size)

def process_audio(audio_file_path, model_size):
    """
    Transcribes the audio file using OpenAI Whisper.
    """
    model = load_whisper_model(model_size)
    # Transcribe
    result = model.transcribe(audio_file_path)
    return result["text"]

def analyze_meeting_with_gemini(transcript, api_key):
    """
    Sends the transcript to Google Gemini to extract structured insights.
    """
    if not api_key:
        return None
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = f"""
    You are an expert Project Manager AI. Analyze the following meeting transcript.
    
    TRANSCRIPT:
    {transcript}
    
    OUTPUT FORMAT (Strict JSON only, do not include markdown formatting):
    {{
        "summary": "A concise executive summary of the meeting (max 3 sentences).",
        "action_items": [
            {{"task": "Description of task", "owner": "Name of person responsible", "deadline": "Date or ASAP", "confidence_score": 0.95}}
        ],
        "key_decisions": ["Decision 1", "Decision 2"],
        "efficiency_score": 85,  // An integer 0-100 based on clarity and productivity
        "bias_conflict_analysis": "A brief analysis of any detected conflict, bias, or tension."
    }}
    """
    
    try:
        response = model.generate_content(prompt)
        # Clean the response to ensure it is valid JSON
        text_response = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(text_response)
    except Exception as e:
        st.error(f"Error connecting to AI: {e}")
        return None

# ==========================================
# 3. STREAMLIT USER INTERFACE
# ==========================================

# --- SIDEBAR SETTINGS ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key Input
    google_api_key = st.text_input("Google Gemini API Key", type="password", help="Get your key at aistudio.google.com")
    
    # Model Selection
    model_size = st.selectbox("Whisper Model Size", ["tiny", "base", "small", "medium"], index=1, 
                              help="'tiny' is fast but less accurate. 'medium' is accurate but slow.")
    
    st.markdown("---")
    st.markdown("**System Status:**")
    if google_api_key:
        st.success("✅ AI Key Loaded")
    else:
        st.warning("⚠️ Waiting for API Key")

# --- MAIN PAGE ---
st.title("🎙️ AI Meeting Minutes Generator")
st.markdown("""
Upload a meeting recording to automatically generate **summaries, action items, and efficiency scores**.
""")

# File Upload Section
uploaded_file = st.file_uploader("📂 Upload Audio File (MP3, WAV, M4A)", type=["mp3", "wav", "m4a"])

if uploaded_file is not None:
    # 1. TEMPORARY FILE HANDLING
    # We must save the uploaded file to disk so Whisper can read it
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") 
    tfile.write(uploaded_file.read())
    tfile_path = tfile.name
    
    # Display Audio Player
    st.audio(uploaded_file)
    
    # 2. PROCESSING BUTTON
    if st.button("🚀 Generate Minutes"):
        if not google_api_key:
            st.error("Please enter your Google Gemini API Key in the sidebar first!")
        else:
            # A. TRANSCRIPTION
            with st.spinner("🎧 Transcribing Audio... (This may take a minute)"):
                try:
                    transcript_text = process_audio(tfile_path, model_size)
                    st.success("Transcription Complete!")
                except Exception as e:
                    st.error(f"Transcription Failed: {e}")
                    transcript_text = None
            
            # B. ANALYSIS
            if transcript_text:
                with st.spinner("🤖 Analyzing with Gemini AI..."):
                    analysis = analyze_meeting_with_gemini(transcript_text, google_api_key)
                
                # C. SAVE TO DATABASE
                if analysis:
                    new_meeting = Meeting(
                        filename=uploaded_file.name,
                        upload_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        transcript=transcript_text,
                        summary=analysis.get("summary"),
                        action_items=json.dumps(analysis.get("action_items")),
                        efficiency_score=analysis.get("efficiency_score"),
                        bias_detected=analysis.get("bias_conflict_analysis")
                    )
                    session.add(new_meeting)
                    session.commit()
                    
                    st.session_state['current_analysis'] = analysis
                    st.session_state['current_transcript'] = transcript_text
                    st.success("✅ Minutes Generated & Saved!")

    # Cleanup temp file
    if os.path.exists(tfile_path):
        os.remove(tfile_path)

# ==========================================
# 4. RESULTS DASHBOARD
# ==========================================
if 'current_analysis' in st.session_state:
    data = st.session_state['current_analysis']
    
    st.divider()
    st.header("📊 Meeting Insights")
    
    # Key Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Efficiency Score", f"{data.get('efficiency_score', 0)}/100")
    col2.metric("Action Items", len(data.get('action_items', [])))
    col3.metric("Decisions Made", len(data.get('key_decisions', [])))
    
    # Detailed Tabs
    tab1, tab2, tab3 = st.tabs(["📝 Summary & Tasks", "⚠️ Bias & Decisions", "📄 Full Transcript"])
    
    with tab1:
        st.subheader("Executive Summary")
        st.info(data.get("summary", "No summary available."))
        
        st.subheader("Action Items")
        items = data.get("action_items", [])
        if items:
            df = pd.DataFrame(items)
            st.dataframe(df, use_container_width=True)
        else:
            st.write("No action items detected.")
            
    with tab2:
        st.subheader("Key Decisions")
        decisions = data.get("key_decisions", [])
        for d in decisions:
            st.success(f"🎯 {d}")
            
        st.subheader("Bias & Conflict Analysis")
        st.warning(data.get("bias_conflict_analysis", "No analysis available."))

    with tab3:
        st.text_area("Transcript", st.session_state.get('current_transcript', ''), height=400)

# ==========================================
# 5. HISTORICAL DATA
# ==========================================
st.divider()
st.subheader("🗂️ History")

# Fetch all meetings from DB
previous_meetings = session.query(Meeting).order_by(Meeting.id.desc()).all()

if previous_meetings:
    for pm in previous_meetings:
        with st.expander(f"{pm.upload_time} - {pm.filename}"):
            st.write(f"**Summary:** {pm.summary}")
            if st.button("🗑️ Delete Record", key=pm.id):
                session.delete(pm)
                session.commit()
                st.rerun()
else:
    st.caption("No previous meetings found.")