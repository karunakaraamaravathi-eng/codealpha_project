"""
Automated Integration & Verification Suite
CodeAlpha AI Projects
"""

import sys
import os

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def test_translator():
    print("Testing Task 1: Language Translation & TTS...")
    from modules.translator import translate_text, generate_speech
    
    sample_en = "Hello, welcome to CodeAlpha Artificial Intelligence Internship."
    translated_hi = translate_text(sample_en, "English", "Hindi")
    print(f"  [Translator] English -> Hindi: {translated_hi}")
    assert len(translated_hi) > 0, "Translation returned empty string"

    audio_bytes = generate_speech("Hello world", "English")
    print(f"  [TTS] Generated audio bytes length: {len(audio_bytes)}")
    assert len(audio_bytes) > 0, "TTS audio bytes empty"
    print("✅ Task 1 (Translator & TTS) passed!\n")


def test_chatbot():
    print("Testing Task 2: FAQ Chatbot with TF-IDF & Cosine Similarity...")
    from modules.chatbot import load_faq_database, PureTfidfEngine
    
    faqs = load_faq_database()
    print(f"  [Chatbot] Loaded {len(faqs)} FAQ entries from data/faqs.json")
    assert len(faqs) > 0, "FAQs empty"

    engine = PureTfidfEngine(faqs)
    
    query = "What is artificial intelligence and machine learning?"
    answer, score, matched = engine.find_best_match(query)
    print(f"  [Chatbot] Query: '{query}'")
    print(f"  [Chatbot] Matched Score: {score:.4f}, Category: {matched['category'] if matched else 'None'}")
    print(f"  [Chatbot] Answer: {answer[:100]}...")
    assert score > 0.20, "Similarity score unexpectedly low"
    print("✅ Task 2 (FAQ Chatbot) passed!\n")


def test_music_gen():
    print("Testing Task 3: Music Generation & Pure Audio Synthesis...")
    from modules.music_gen import MusicMarkovGenerator, generate_midi_file, synthesize_audio_wav
    
    gen = MusicMarkovGenerator(order=1)
    notes = gen.generate(length=16, scale_name="C Major")
    print(f"  [Music Gen] Generated {len(notes)} notes: {' '.join(notes)}")
    assert len(notes) == 16, "Generated notes count mismatch"

    midi_bytes = generate_midi_file(notes, tempo_bpm=120)
    print(f"  [Music Gen] MIDI size: {len(midi_bytes)} bytes")
    assert len(midi_bytes) > 0, "MIDI bytes empty"

    wav_bytes = synthesize_audio_wav(notes, tempo_bpm=120, timbre="Piano")
    print(f"  [Music Gen] Synthesized WAV size: {len(wav_bytes)} bytes")
    assert len(wav_bytes) > 0, "WAV bytes empty"
    print("✅ Task 3 (Music Generation) passed!\n")


def test_object_tracking():
    print("Testing Task 4: Vision Object Detection & Tracking Engine...")
    import cv2
    import numpy as np
    from modules.object_tracking import get_detector
    
    det_type, detector = get_detector()
    print(f"  [Vision Engine] Initialized: {det_type.upper()}")

    # Synthetic image test
    dummy_img = np.full((300, 300, 3), 200, dtype=np.uint8)
    cv2.circle(dummy_img, (150, 150), 50, (0, 0, 255), -1)

    if det_type == "yolo":
        results = detector.predict(dummy_img, verbose=False)
        print("  [YOLO] Predict succeeded")
    else:
        annotated, counts, tracked = detector.detect_and_track(dummy_img, conf_threshold=0.2, is_video=False)
        print(f"  [OpenCV Vision] Output shape: {annotated.shape}, Tracked objects: {len(tracked)}")
        assert annotated.shape == dummy_img.shape, "Annotated frame shape mismatch"

    print("✅ Task 4 (Object Tracking) passed!\n")


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Running CodeAlpha AI Suite All-Module Verification")
    print("=" * 60)
    
    test_translator()
    test_chatbot()
    test_music_gen()
    test_object_tracking()

    print("=" * 60)
    print("🎉 ALL 4 TASKS VERIFIED AND PASSING SUCCESSFULLY!")
    print("=" * 60)
