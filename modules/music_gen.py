"""
Task 3: Music Generation with AI
Unified AI Web Application - CodeAlpha AI Internship
"""

import os
import pickle
import random
import io
import math
import struct
import wave
import streamlit as st
import music21 as m21

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
SAMPLE_NOTES_FILE = os.path.join(DATA_DIR, "sample_notes.pkl")

# Musical Scale mappings (Root + Intervals in semitones)
SCALES = {
    "C Major": ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5"],
    "A Minor (Melancholic)": ["A3", "B3", "C4", "D4", "E4", "F4", "G4", "A4"],
    "G Major (Uplifting)": ["G3", "A3", "B3", "C4", "D4", "E4", "F#4", "G4"],
    "D Dorian (Jazz / Ambient)": ["D4", "E4", "F4", "G4", "A4", "B4", "C5", "D5"],
    "Blues Pentatonic": ["C4", "Eb4", "F4", "F#4", "G4", "Bb4", "C5"],
    "Japanese Insen (Oriental)": ["C4", "Db4", "F4", "G4", "Bb4", "C5"]
}

# Seed sequences for Markov generator
SEED_SEQUENCES = [
    ["C4", "E4", "G4", "B4", "C5", "G4", "E4", "C4"],
    ["A3", "C4", "E4", "A4", "G4", "E4", "C4", "B3"],
    ["D4", "F4", "A4", "C5", "B4", "G4", "E4", "D4"],
    ["C4", "G4", "A4", "F4", "E4", "G4", "C5", "B4"],
    ["E4", "G#4", "B4", "E5", "D#5", "B4", "G#4", "E4"],
    ["F4", "A4", "C5", "E5", "D5", "Bb4", "G4", "C4"]
]


def ensure_sample_notes_file():
    """Generates and saves baseline seed training notes to data/sample_notes.pkl if missing."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(SAMPLE_NOTES_FILE):
        corpus_notes = []
        for seq in SEED_SEQUENCES:
            corpus_notes.extend(seq)
        with open(SAMPLE_NOTES_FILE, "wb") as f:
            pickle.dump(corpus_notes, f)


class MusicMarkovGenerator:
    """Lightweight statistical Markov Chain sequence generator for musical note transitions."""
    def __init__(self, order=1):
        self.order = order
        self.transitions = {}
        self._train()

    def _train(self):
        ensure_sample_notes_file()
        try:
            with open(SAMPLE_NOTES_FILE, "rb") as f:
                notes = pickle.load(f)
        except Exception:
            notes = [n for seq in SEED_SEQUENCES for n in seq]

        for i in range(len(notes) - self.order):
            key = tuple(notes[i:i + self.order])
            next_note = notes[i + self.order]
            if key not in self.transitions:
                self.transitions[key] = []
            self.transitions[key].append(next_note)

    def generate(self, length=24, scale_name="C Major", temperature=0.7):
        scale_pool = SCALES.get(scale_name, SCALES["C Major"])
        if not self.transitions:
            return [random.choice(scale_pool) for _ in range(length)]

        # Start from a random existing key or scale note
        keys = list(self.transitions.keys())
        current_key = random.choice(keys)
        result = list(current_key)

        for _ in range(length - len(result)):
            # With some probability depending on temperature, explore scale note
            if random.random() < (1.0 - temperature) and current_key in self.transitions:
                next_note = random.choice(self.transitions[current_key])
            else:
                next_note = random.choice(scale_pool)

            result.append(next_note)
            current_key = tuple(result[-self.order:])

        return result


def note_name_to_freq(note_str: str) -> float:
    """Converts a scientific note name (e.g. 'A4', 'C#4') to its frequency in Hertz."""
    try:
        p = m21.pitch.Pitch(note_str)
        return float(p.frequency)
    except Exception:
        # Fallback table
        base_freqs = {
            "C4": 261.63, "C#4": 277.18, "Db4": 277.18, "D4": 293.66, "Eb4": 311.13,
            "E4": 329.63, "F4": 349.23, "F#4": 369.99, "G4": 392.00, "G#4": 415.30,
            "Ab4": 415.30, "A4": 440.00, "Bb4": 466.16, "B4": 493.88, "C5": 523.25,
            "D5": 587.33, "Eb5": 622.25, "E5": 659.25, "A3": 220.00, "B3": 246.94, "G3": 196.00
        }
        return base_freqs.get(note_str, 440.0)


def synthesize_audio_wav(note_sequence, tempo_bpm=120, timbre="Piano") -> bytes:
    """
    Synthesizes a musical note sequence into standard 44.1kHz 16-bit Mono WAV audio bytes
    using pure Python additive harmonic synthesis with ADSR envelopes and zero external DLLs.
    """
    sample_rate = 44100
    quarter_duration = 60.0 / float(tempo_bpm)
    dur = quarter_duration * 0.95
    num_samples_per_note = int(dur * sample_rate)
    
    total_samples = int((len(note_sequence) * quarter_duration + 0.5) * sample_rate)
    samples = [0.0] * total_samples

    for idx, note_str in enumerate(note_sequence):
        freq = note_name_to_freq(note_str)
        start_idx = int(idx * quarter_duration * sample_rate)

        for s in range(num_samples_per_note):
            t = float(s) / sample_rate
            
            # Harmonics
            if timbre == "Piano":
                val = (
                    0.60 * math.sin(2.0 * math.pi * freq * t) +
                    0.25 * math.sin(2.0 * math.pi * 2.0 * freq * t) +
                    0.10 * math.sin(2.0 * math.pi * 3.0 * freq * t) +
                    0.05 * math.sin(2.0 * math.pi * 4.0 * freq * t)
                )
                # Envelope: Attack & Exponential decay
                env = math.exp(-3.5 * t / dur) if t > 0.02 else (t / 0.02)
            elif timbre == "Warm Synth Lead":
                val = (
                    0.50 * math.sin(2.0 * math.pi * freq * t) +
                    0.30 * math.sin(2.0 * math.pi * 2.0 * freq * t) +
                    0.15 * math.sin(2.0 * math.pi * 3.0 * freq * t)
                )
                env = 1.0 - (0.3 * t / dur) if t > 0.05 else (t / 0.05)
            else:  # Crystal Bell
                val = (
                    0.50 * math.sin(2.0 * math.pi * freq * t) +
                    0.35 * math.sin(2.0 * math.pi * 2.756 * freq * t) +
                    0.15 * math.sin(2.0 * math.pi * 5.404 * freq * t)
                )
                env = math.exp(-5.0 * t / dur) if t > 0.01 else (t / 0.01)

            sample_val = val * env
            pos = start_idx + s
            if pos < total_samples:
                samples[pos] += sample_val

    # Normalize samples
    max_amp = max(abs(x) for x in samples) if samples else 1.0
    if max_amp == 0:
        max_amp = 1.0
    scaling = 28000.0 / max_amp

    # Pack into 16-bit PCM WAV
    wav_io = io.BytesIO()
    with wave.open(wav_io, "wb") as wf:
        wf.setnchannels(1)        # mono
        wf.setsampwidth(2)        # 16-bit
        wf.setframerate(sample_rate)
        
        frames = bytearray()
        for sample in samples:
            int_val = int(max(-32767, min(32767, sample * scaling)))
            frames.extend(struct.pack("<h", int_val))
        wf.writeframes(frames)

    wav_io.seek(0)
    return wav_io.read()


def generate_midi_file(note_sequence, tempo_bpm=120) -> bytes:
    """Creates a standard MIDI file from a note sequence using music21."""
    score = m21.stream.Stream()
    score.append(m21.tempo.MetronomeMark(number=tempo_bpm))
    score.append(m21.instrument.Piano())

    for note_str in note_sequence:
        try:
            n = m21.note.Note(note_str)
            n.quarterLength = 1.0
            score.append(n)
        except Exception:
            continue

    temp_midi_path = os.path.join(DATA_DIR, "temp_generated.mid")
    score.write("midi", fp=temp_midi_path)
    
    with open(temp_midi_path, "rb") as f:
        midi_bytes = f.read()
    
    if os.path.exists(temp_midi_path):
        os.remove(temp_midi_path)
        
    return midi_bytes


def render_music_gen_ui():
    """Renders the Streamlit UI for Task 3 AI Music Generation."""
    st.markdown("""
        <div style="background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%); padding: 1.5rem; border-radius: 12px; color: white; margin-bottom: 1.5rem; box-shadow: 0 4px 12px rgba(168,85,247,0.2);">
            <h2 style="margin:0; font-weight:700; color:white;">🎵 AI Music & Melody Composition Generator</h2>
            <p style="margin:0.5rem 0 0 0; opacity:0.9; font-size: 0.95rem;">Compose original algorithmic melodies using Music21 symbolic encoding, Markov transition models, and pure audio wave synthesis.</p>
        </div>
    """, unsafe_allow_html=True)

    ensure_sample_notes_file()
    generator = MusicMarkovGenerator(order=1)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("#### 🎼 Composition Parameters")
        scale_choice = st.selectbox("Musical Scale / Mood", list(SCALES.keys()), index=0)
        timbre_choice = st.selectbox("Instrument Timbre", ["Piano", "Warm Synth Lead", "Crystal Bell"], index=0)
        seq_length = st.slider("Melody Length (Notes)", min_value=8, max_value=48, value=20, step=4)
        tempo_bpm = st.slider("Tempo (BPM)", min_value=60, max_value=180, value=120, step=5)
        temperature = st.slider("Creativity / Temperature", min_value=0.1, max_value=1.0, value=0.65, step=0.05,
                                help="Higher values introduce more unexpected melodic transitions.")

        generate_btn = st.button("✨ Generate AI Melody", type="primary", use_container_width=True)

    with col2:
        st.markdown("#### 🎹 Generated Melody Preview")
        
        if generate_btn:
            with st.spinner("Composing and synthesizing audio..."):
                notes = generator.generate(length=seq_length, scale_name=scale_choice, temperature=temperature)
                midi_data = generate_midi_file(notes, tempo_bpm=tempo_bpm)
                wav_data = synthesize_audio_wav(notes, tempo_bpm=tempo_bpm, timbre=timbre_choice)

                st.session_state["generated_music"] = {
                    "notes": notes,
                    "midi_data": midi_data,
                    "wav_data": wav_data,
                    "scale": scale_choice,
                    "tempo": tempo_bpm
                }

        if "generated_music" in st.session_state:
            music_obj = st.session_state["generated_music"]
            
            st.success(f"🎶 Melody successfully generated ({len(music_obj['notes'])} notes in {music_obj['scale']})!")
            
            st.markdown("##### 🔊 In-Browser Audio Player")
            st.audio(music_obj["wav_data"], format="audio/wav")

            # Sequence Note representation
            st.markdown("##### 📝 Symbolic Note Sequence")
            notes_display = " ➔ ".join(music_obj["notes"])
            st.markdown(f"""
                <div style="background:#f1f5f9; padding:0.75rem; border-radius:8px; font-family: monospace; font-size: 0.9rem; max-height: 100px; overflow-y: auto; color: #334155; border: 1px solid #cbd5e1;">
                    {notes_display}
                </div>
            """, unsafe_allow_html=True)

            # Download Buttons
            st.markdown("##### 💾 Export Options")
            d_col1, d_col2 = st.columns(2)
            d_col1.download_button(
                label="📥 Download MIDI (.mid)",
                data=music_obj["midi_data"],
                file_name="ai_composition.mid",
                mime="audio/midi",
                use_container_width=True
            )
            d_col2.download_button(
                label="📥 Download Audio (.wav)",
                data=music_obj["wav_data"],
                file_name="ai_composition.wav",
                mime="audio/wav",
                use_container_width=True
            )
        else:
            st.info("👈 Configure your musical parameters and click **'Generate AI Melody'** to create your first track!")
