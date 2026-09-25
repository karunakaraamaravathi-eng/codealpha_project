"""
Task 1: Language Translation Tool
Unified AI Web Application - CodeAlpha AI Internship
"""

import io
import json
import urllib.parse
import urllib.request
import streamlit as st
from deep_translator import GoogleTranslator
from gtts import gTTS

# Supported Languages mapping: Name -> Code
SUPPORTED_LANGUAGES = {
    "English": "en",
    "Hindi": "hi",
    "Telugu": "te",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Japanese": "ja",
    "Tamil": "ta",
    "Bengali": "bn",
    "Russian": "ru",
    "Arabic": "ar",
    "Italian": "it",
    "Portuguese": "pt",
    "Korean": "ko",
    "Chinese (Simplified)": "zh-CN",
    "Dutch": "nl",
    "Greek": "el",
    "Turkish": "tr",
    "Marathi": "mr",
    "Gujarati": "gu",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Punjabi": "pa"
}

SAMPLE_TEXTS = [
    "Artificial intelligence is transforming how we learn, communicate, and solve real-world challenges.",
    "CodeAlpha provides exceptional hands-on internship opportunities for budding AI engineers.",
    "Welcome to our unified AI web application with modern real-time capabilities!",
    "Machine learning models can identify patterns that are impossible for humans to detect manually."
]


def translate_via_direct_api(text: str, src: str, tgt: str) -> str:
    """Translates text using direct Google Translate REST gateway."""
    encoded_text = urllib.parse.quote(text)
    url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl={src}&tl={tgt}&dt=t&q={encoded_text}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    )
    with urllib.request.urlopen(req, timeout=10) as response:
        content = response.read().decode("utf-8")
        data = json.loads(content)
        translated_chunks = [item[0] for item in data[0] if item[0]]
        return "".join(translated_chunks)


def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    """Translates text from source language to target language using resilient multi-tier fallbacks."""
    if not text.strip():
        return ""
    src = "auto" if source_lang == "Auto-Detect" else SUPPORTED_LANGUAGES.get(source_lang, "auto")
    tgt = SUPPORTED_LANGUAGES.get(target_lang, "en")
    
    # Tier 1: Direct High-Speed REST Gateway
    try:
        res = translate_via_direct_api(text, src, tgt)
        if res and res.strip():
            return res
    except Exception:
        pass

    # Tier 2: Deep-Translator Google Wrapper
    try:
        translator = GoogleTranslator(source=src, target=tgt)
        translated = translator.translate(text)
        if translated and translated.strip():
            return translated
    except Exception:
        pass

    return text


def generate_speech(text: str, lang_name: str) -> bytes:
    """Generates speech audio bytes using gTTS."""
    lang_code = SUPPORTED_LANGUAGES.get(lang_name, "en")
    code_for_tts = lang_code.split("-")[0] if "-" in lang_code else lang_code
    
    try:
        tts = gTTS(text=text, lang=code_for_tts, slow=False)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        return audio_buffer.read()
    except Exception:
        # Fallback to english if target language voice is unsupported
        tts = gTTS(text=text, lang="en", slow=False)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        return audio_buffer.read()


def render_translator_ui():
    """Renders the Streamlit UI for the Language Translation Tool."""
    st.markdown("""
        <div style="background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); padding: 1.5rem; border-radius: 12px; color: white; margin-bottom: 1.5rem; box-shadow: 0 4px 12px rgba(37,99,235,0.2);">
            <h2 style="margin:0; font-weight:700; color:white;">🌐 AI Multi-Language Translation Tool</h2>
            <p style="margin:0.5rem 0 0 0; opacity:0.9; font-size: 0.95rem;">Translate text instantly across 20+ languages with automatic detection and Text-to-Speech (TTS) audio synthesis.</p>
        </div>
    """, unsafe_allow_html=True)

    if "translation_history" not in st.session_state:
        st.session_state.translation_history = []

    # Preset Sample Selector
    with st.expander("💡 Quick Sample Presets (Click to insert)"):
        preset_cols = st.columns(len(SAMPLE_TEXTS))
        for i, sample in enumerate(SAMPLE_TEXTS):
            if preset_cols[i].button(f"Preset #{i+1}", key=f"preset_{i}", use_container_width=True):
                st.session_state["input_text_area"] = sample
                st.rerun()

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("#### 📝 Source Input")
        src_opts = ["Auto-Detect"] + list(SUPPORTED_LANGUAGES.keys())
        source_lang = st.selectbox("Source Language", options=src_opts, index=0, key="src_lang")
        
        input_text = st.text_area(
            "Enter text to translate:",
            value=st.session_state.get("input_text_area", "Artificial Intelligence is shaping the future of technology."),
            height=180,
            key="input_text_area"
        )
        
        char_count = len(input_text)
        word_count = len(input_text.split()) if input_text.strip() else 0
        st.caption(f"📊 Stats: **{char_count}** characters | **{word_count}** words")

    with col2:
        st.markdown("#### 🎯 Target Output")
        tgt_opts = list(SUPPORTED_LANGUAGES.keys())
        default_tgt_index = tgt_opts.index("Hindi") if "Hindi" in tgt_opts else 1
        target_lang = st.selectbox("Target Language", options=tgt_opts, index=default_tgt_index, key="tgt_lang")
        
        # Translation Trigger
        translate_btn = st.button("🚀 Translate Now", type="primary", use_container_width=True)

    translated_text = ""
    
    if translate_btn or "last_translation" in st.session_state:
        if translate_btn:
            if not input_text.strip():
                st.warning("⚠️ Please enter some text before translating.")
            else:
                with st.spinner("Translating text..."):
                    try:
                        translated_text = translate_text(input_text, source_lang, target_lang)
                        st.session_state["last_translation"] = {
                            "source_text": input_text,
                            "translated_text": translated_text,
                            "src": source_lang,
                            "tgt": target_lang
                        }
                        # Add to history
                        st.session_state.translation_history.insert(0, {
                            "src": source_lang,
                            "tgt": target_lang,
                            "original": input_text[:60] + ("..." if len(input_text) > 60 else ""),
                            "translated": translated_text[:60] + ("..." if len(translated_text) > 60 else "")
                        })
                    except Exception as err:
                        st.error(f"❌ Translation failed: {str(err)}")
        else:
            cached = st.session_state.get("last_translation", {})
            translated_text = cached.get("translated_text", "")

    if translated_text:
        st.markdown("---")
        res_col1, res_col2 = st.columns([2, 1], gap="medium")
        
        with res_col1:
            st.markdown("### 📋 Translated Result")
            st.markdown(f"""
                <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-left: 4px solid #3b82f6; border-radius: 8px; padding: 1.25rem; font-size: 1.15rem; line-height: 1.6; color: #0f172a; min-height: 100px;">
                    {translated_text}
                </div>
            """, unsafe_allow_html=True)
            
            st.code(translated_text, language="text")
            st.caption("📋 Use the copy icon in the upper right of the box above to copy instantly.")

        with res_col2:
            st.markdown("### 🔊 Text-to-Speech (TTS)")
            if st.button("🎙️ Generate & Play Audio", use_container_width=True):
                with st.spinner("Synthesizing audio voice..."):
                    try:
                        audio_data = generate_speech(translated_text, target_lang)
                        st.audio(audio_data, format="audio/mp3")
                        st.success("Audio playback ready!")
                    except Exception as e:
                        st.error(f"TTS error: {e}")

    # Recent History
    if st.session_state.translation_history:
        st.markdown("---")
        with st.expander("🕒 Recent Translation History"):
            for idx, hist in enumerate(st.session_state.translation_history[:5]):
                st.markdown(f"**#{idx+1} [{hist['src']} ➔ {hist['tgt']}]**")
                st.markdown(f"*Original:* {hist['original']}")
                st.markdown(f"*Translated:* {hist['translated']}")
                st.markdown("---")
