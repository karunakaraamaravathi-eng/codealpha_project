"""
CodeAlpha Artificial Intelligence Internship - Unified AI Web Application
Main Entrypoint & Interactive Dashboard
"""

import sys
import os
import streamlit as st

# Add project root to Python module path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.translator import render_translator_ui
from modules.chatbot import render_chatbot_ui
from modules.music_gen import render_music_gen_ui
from modules.object_tracking import render_object_tracking_ui

# Configure Page
st.set_page_config(
    page_title="CodeAlpha AI Suite - Unified Production App",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Glassmorphism, Vibrant Gradients, Modern Cards)
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Modern Card Styles */
    .feature-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        height: 100%;
    }
    .feature-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 28px rgba(0, 0, 0, 0.08);
        border-color: #cbd5e1;
    }

    .badge-pill {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .badge-blue { background: #dbeafe; color: #1e40af; }
    .badge-green { background: #dcfce7; color: #166534; }
    .badge-purple { background: #f3e8ff; color: #6b21a8; }
    .badge-red { background: #fee2e2; color: #991b1b; }

    /* Clean Metric containers */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700;
        color: #1e293b;
    }

    /* Custom Header Banner */
    .hero-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
        border-radius: 20px;
        padding: 2.5rem;
        color: #ffffff;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.15);
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_overview():
    """Renders the Executive Overview & Landing Page for CodeAlpha AI Internship."""
    st.markdown("""
        <div class="hero-banner">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;">
                <div>
                    <span class="badge-pill" style="background:#3b82f6; color:white; margin-bottom:0.75rem;">Official Internship Submission</span>
                    <h1 style="color:white; margin:0.25rem 0 0.5rem 0; font-size:2.4rem; font-weight:800;">
                        CodeAlpha Artificial Intelligence Suite
                    </h1>
                    <p style="color:#94a3b8; font-size:1.05rem; max-width:700px; margin:0;">
                        A production-grade, unified multi-task AI web platform implementing all four domain projects: Machine Translation with TTS, NLP FAQ Chatbot, AI Music Composition, and Real-Time YOLOv8 Object Tracking.
                    </p>
                </div>
                <div style="text-align:right; margin-top:1rem;">
                    <span style="font-size:0.85rem; color:#94a3b8;">Developed for</span><br/>
                    <strong style="font-size:1.2rem; color:#38bdf8;">CodeAlpha AI Internship</strong>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Key Metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Tasks Built", "4 of 4", "100% Complete")
    with m2:
        st.metric("NLP Matching", "TF-IDF + Cosine", "High Accuracy")
    with m3:
        st.metric("Computer Vision", "YOLOv8 Nano", "Persistent IDs")
    with m4:
        st.metric("Symbolic Audio", "Music21 + WAV", "Pure Python Synth")

    st.markdown("### 🚀 Interactive Task Suite Overview")
    st.markdown("Select any task below or use the sidebar navigation to test each AI system interactively.")

    col1, col2 = st.columns(2, gap="medium")

    with col1:
        st.markdown("""
            <div class="feature-card">
                <span class="badge-pill badge-blue">Task 1: NLP & Speech</span>
                <h3 style="margin:0.75rem 0 0.5rem 0; color:#1e293b;">🌐 Language Translation Tool</h3>
                <p style="color:#64748b; font-size:0.92rem; line-height:1.5;">
                    Translate text seamlessly across 20+ global and Indian regional languages (Hindi, Telugu, Tamil, Spanish, French, etc.) with automated language detection and in-browser Text-to-Speech (gTTS) audio preview.
                </p>
                <div style="margin-top:1rem; font-size:0.85rem; color:#475569;">
                    <strong>Tech Stack:</strong> <code>deep-translator</code>, <code>gTTS</code>, <code>Streamlit</code>
                </div>
            </div>
        """, unsafe_allow_html=True)
        if st.button("👉 Launch Translation Tool", key="btn_nav_t1", use_container_width=True):
            st.session_state["nav_selection"] = "🌐 Task 1: Translation Tool"
            st.rerun()

        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

        st.markdown("""
            <div class="feature-card">
                <span class="badge-pill badge-purple">Task 3: Algorithmic Generative AI</span>
                <h3 style="margin:0.75rem 0 0.5rem 0; color:#1e293b;">🎵 AI Music Generation</h3>
                <p style="color:#64748b; font-size:0.92rem; line-height:1.5;">
                    Generate original symbolic melodies and chord patterns across multiple modal scales. Synthesize 44.1kHz audio in real time with customizable timbres and download standard MIDI files (.mid).
                </p>
                <div style="margin-top:1rem; font-size:0.85rem; color:#475569;">
                    <strong>Tech Stack:</strong> <code>music21</code>, <code>scipy.io.wavfile</code>, <code>numpy</code>, <code>Markov Chains</code>
                </div>
            </div>
        """, unsafe_allow_html=True)
        if st.button("👉 Launch AI Music Studio", key="btn_nav_t3", use_container_width=True):
            st.session_state["nav_selection"] = "🎵 Task 3: AI Music Studio"
            st.rerun()

    with col2:
        st.markdown("""
            <div class="feature-card">
                <span class="badge-pill badge-green">Task 2: Conversational NLP</span>
                <h3 style="margin:0.75rem 0 0.5rem 0; color:#1e293b;">💬 Intelligent FAQ Chatbot</h3>
                <p style="color:#64748b; font-size:0.92rem; line-height:1.5;">
                    Conversational agent using TF-IDF vectorization and Cosine Similarity to provide instantaneous answers to user questions regarding AI/ML concepts and CodeAlpha internship criteria with confidence scoring.
                </p>
                <div style="margin-top:1rem; font-size:0.85rem; color:#475569;">
                    <strong>Tech Stack:</strong> <code>scikit-learn</code>, <code>NLTK</code>, <code>Cosine Similarity</code>
                </div>
            </div>
        """, unsafe_allow_html=True)
        if st.button("👉 Launch FAQ Chatbot", key="btn_nav_t2", use_container_width=True):
            st.session_state["nav_selection"] = "💬 Task 2: FAQ Chatbot"
            st.rerun()

        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

        st.markdown("""
            <div class="feature-card">
                <span class="badge-pill badge-red">Task 4: Deep Learning Computer Vision</span>
                <h3 style="margin:0.75rem 0 0.5rem 0; color:#1e293b;">🎯 Real-Time Object Detection & Tracking</h3>
                <p style="color:#64748b; font-size:0.92rem; line-height:1.5;">
                    State-of-the-art YOLOv8 nano model processing video feeds and camera streams with persistent object IDs, bounding box coordinates, class labels, and confidence analytics.
                </p>
                <div style="margin-top:1rem; font-size:0.85rem; color:#475569;">
                    <strong>Tech Stack:</strong> <code>ultralytics (YOLOv8)</code>, <code>OpenCV</code>, <code>Pillow</code>
                </div>
            </div>
        """, unsafe_allow_html=True)
        if st.button("👉 Launch Object Tracking", key="btn_nav_t4", use_container_width=True):
            st.session_state["nav_selection"] = "🎯 Task 4: Object Tracking"
            st.rerun()


def main():
    # Sidebar Navigation
    with st.sidebar:
        st.markdown("""
            <div style="text-align:center; padding: 1rem 0;">
                <h2 style="margin:0; color:#1e293b; font-weight:800;">🤖 CodeAlpha AI</h2>
                <span style="font-size:0.85rem; color:#64748b;">Unified Internship Platform</span>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")

        nav_options = [
            "🏠 Executive Overview",
            "🌐 Task 1: Translation Tool",
            "💬 Task 2: FAQ Chatbot",
            "🎵 Task 3: AI Music Studio",
            "🎯 Task 4: Object Tracking"
        ]

        if "nav_selection" not in st.session_state:
            st.session_state["nav_selection"] = "🏠 Executive Overview"

        # Sync selection
        selected_nav = st.radio(
            "Navigation Menu",
            nav_options,
            index=nav_options.index(st.session_state["nav_selection"]),
            key="main_nav_radio"
        )
        st.session_state["nav_selection"] = selected_nav

        st.markdown("---")
        st.markdown("### 📌 Internship Info")
        st.markdown("""
            - **Domain:** Artificial Intelligence
            - **Organization:** CodeAlpha
            - **Status:** All 4 Tasks Implemented
            - **Ready for Submission:** Yes ✅
        """)

    # Main Router
    if selected_nav == "🏠 Executive Overview":
        render_overview()
    elif selected_nav == "🌐 Task 1: Translation Tool":
        render_translator_ui()
    elif selected_nav == "💬 Task 2: FAQ Chatbot":
        render_chatbot_ui()
    elif selected_nav == "🎵 Task 3: AI Music Studio":
        render_music_gen_ui()
    elif selected_nav == "🎯 Task 4: Object Tracking":
        render_object_tracking_ui()


if __name__ == "__main__":
    main()
