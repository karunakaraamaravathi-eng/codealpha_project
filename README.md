# 🤖 CodeAlpha Artificial Intelligence Suite

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?logo=opencv&logoColor=black)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Internship](https://img.shields.io/badge/CodeAlpha-AI%20Internship-orange)](https://www.codealpha.tech/)

**A unified, production-grade Artificial Intelligence web application featuring all 4 tasks from the CodeAlpha AI Internship curriculum in a modern, responsive dashboard.**

[Explore Tasks](#-curriculum-task-implementations) • [Quick Start](#-quick-start) • [Architecture](#-architecture) • [Internship Details](#-internship-submission)

</div>

---

## 📌 Executive Overview

This repository represents the official submission for the **CodeAlpha Artificial Intelligence Internship**. It combines Computer Vision, Natural Language Processing, Generative Audio AI, and Speech Synthesis into a single, cohesive web platform built with Streamlit and modern UI styling.

### 🌟 Key Highlights
- **100% Curriculum Coverage**: All 4 core internship tasks fully implemented and verified.
- **Modern Responsive UI**: Custom glassmorphism styling, gradient cards, real-time analytics, and smooth multi-tab navigation.
- **Zero-Crash Resilience**: Multi-tier API gateways and pure-Python mathematical fallbacks for seamless execution across any operating system.

---

## 🚀 Curriculum Task Implementations

### 🌐 Task 1: Language Translation Tool (`modules/translator.py`)
- **Features**: Translates text across 20+ languages (English, Hindi, Telugu, Tamil, Spanish, French, German, Japanese, etc.) with Auto-Detection.
- **Speech Synthesis**: Integrated Text-to-Speech (`gTTS`) generating instant in-browser audio previews.
- **Usability**: One-click clipboard copy, character counters, sample presets, and session translation history.

### 💬 Task 2: Intelligent FAQ Chatbot (`modules/chatbot.py`)
- **NLP Processing**: Text normalization, tokenization, stop-word elimination, and Unigram/Bigram representation.
- **Matching Engine**: Sublinear **TF-IDF Vectorization** and **Cosine Similarity** intent scoring.
- **Intelligence**: Dynamic confidence threshold slider (default: 30%) with graceful domain fallbacks and interactive query suggestions.
- **Knowledge Base**: Curated [`data/faqs.json`](data/faqs.json) covering AI, Machine Learning, Deep Learning, and CodeAlpha guidelines.

### 🎵 Task 3: AI Music Generation (`modules/music_gen.py`)
- **Symbolic Composition**: Algorithmic Markov Chain sequence transition generator powered by `music21`.
- **Musical Modes**: Major, Melancholic Minor, Blues Pentatonic, Dorian Jazz, and Oriental Insen scales.
- **Audio Synthesizer**: Pure Python 44.1kHz 16-bit WAV wave synthesizer with ADSR envelopes and harmonic timbres (Piano, Warm Synth Lead, Crystal Bell).
- **Exporting**: Direct in-browser audio playback, **Download MIDI (`.mid`)**, and **Download Audio (`.wav`)**.

### 🎯 Task 4: Real-Time Object Detection & Tracking (`modules/object_tracking.py`)
- **Vision Engine**: State-of-the-art **Ultralytics YOLOv8 nano** (`yolov8n.pt`) with OpenCV Centroid Tracker integration.
- **Persistent Tracking**: Assigns and preserves unique persistent tracking IDs ($ID_1, ID_2, \dots$) across frames with bounding boxes and confidence scores.
- **Multi-Source Support**: Processes uploaded images, MP4/AVI video files, and live webcam camera streams.

---

## 📂 Repository Architecture

```text
CodeAlpha_AI_Projects/
│
├── app.py                      # Master Streamlit dashboard entrypoint & navigation
├── requirements.txt            # Unified project dependencies
├── README.md                   # Comprehensive project documentation
├── test_modules.py             # Automated end-to-end integration test suite
│
├── modules/
│   ├── translator.py           # Task 1: Translation engine & TTS
│   ├── chatbot.py              # Task 2: TF-IDF & Cosine Similarity FAQ engine
│   ├── music_gen.py            # Task 3: Algorithmic Generative Music & Synthesizer
│   └── object_tracking.py      # Task 4: YOLOv8 & OpenCV multi-object tracker
│
└── data/
    └── faqs.json               # Structured FAQ knowledge base dataset
```

---

## ⚡ Quick Start & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/karunakaraamaravathi-eng/codealpha_project.git
cd codealpha_project
```

### 2. Set Up Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Automated Tests
```bash
python test_modules.py
```

### 5. Launch the Web Application
```bash
python -m streamlit run app.py
```
Open **`http://localhost:8501`** in your browser to interact with all 4 AI applications.

---

## 🛠️ Technology Stack

| Domain | Libraries & Frameworks |
| :--- | :--- |
| **Web UI & Dashboard** | Streamlit, HTML5, Vanilla CSS (Glassmorphism) |
| **Computer Vision** | Ultralytics YOLOv8, OpenCV (`cv2`), Pillow |
| **Natural Language Processing** | Scikit-Learn, NLTK, TF-IDF Vectorizer, Cosine Similarity |
| **Machine Translation & Speech** | Deep-Translator, Google Translate REST API, gTTS |
| **Generative Audio & Music** | Music21, Wave, NumPy, SciPy |

---

## 👨‍💻 Author & Acknowledgements

- **Author**: Karunakara Amaravathi ([@karunakaraamaravathi-eng](https://github.com/karunakaraamaravathi-eng))
- **Internship Program**: [CodeAlpha](https://www.codealpha.tech/) - Artificial Intelligence Internship
- **Domain**: Artificial Intelligence & Machine Learning

---

<div align="center">
  <sub>Built with ❤️ for the CodeAlpha AI Internship. Star ⭐ this repository if you find it helpful!</sub>
</div>