"""
Task 2: FAQ Chatbot with NLP & Cosine Similarity
Unified AI Web Application - CodeAlpha AI Internship
"""

import json
import math
import os
import re
from collections import Counter
import streamlit as st

# Default fallback message
FALLBACK_RESPONSE = (
    "🤖 I'm sorry, but I couldn't find a sufficiently relevant answer in my knowledge base. "
    "Please try rephrasing your question or check the FAQ categories above!"
)

FAQ_FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "faqs.json")

# Standard English stopwords list
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll",
    "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's",
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once",
    "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
    "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there",
    "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll",
    "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's",
    "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't",
    "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"
}


def clean_and_tokenize(text: str):
    """Preprocesses text: lowercasing, stripping punctuation, stop-word removal, and n-grams."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    words = [w for w in text.split() if w and w not in STOPWORDS]
    
    # Generate unigrams + bigrams
    tokens = list(words)
    for i in range(len(words) - 1):
        tokens.append(f"{words[i]}_{words[i+1]}")
    return tokens


@st.cache_data
def load_faq_database():
    """Loads and caches the FAQ knowledge base from JSON."""
    if not os.path.exists(FAQ_FILE_PATH):
        return []
    with open(FAQ_FILE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


class PureTfidfEngine:
    """
    High-performance pure Python TF-IDF and Cosine Similarity engine.
    Computes exact sublinear TF-IDF vectors and cosine similarities without external DLL dependencies.
    """
    def __init__(self, faqs):
        self.faqs = faqs
        self.doc_tokens = []
        self.vocabulary = {}
        self.idf = {}
        self.doc_vectors = []
        self._build_index()

    def _build_index(self):
        if not self.faqs:
            return

        # Tokenize all documents
        for item in self.faqs:
            q = item.get("question", "")
            keywords = " ".join(item.get("keywords", []))
            cat = item.get("category", "")
            combined = f"{q} {keywords} {cat}"
            tokens = clean_and_tokenize(combined)
            self.doc_tokens.append(tokens)

        # Build vocabulary & Document Frequency (DF)
        df = Counter()
        for tokens in self.doc_tokens:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                df[token] += 1

        num_docs = len(self.doc_tokens)
        self.vocabulary = {token: idx for idx, token in enumerate(df.keys())}
        
        # Compute smooth IDF: ln((1 + N) / (1 + df)) + 1
        for token, count in df.items():
            self.idf[token] = math.log((1.0 + num_docs) / (1.0 + count)) + 1.0

        # Compute TF-IDF vectors for documents
        for tokens in self.doc_tokens:
            vec = self._vectorize(tokens)
            self.doc_vectors.append(vec)

    def _vectorize(self, tokens):
        """Converts a token list into a normalized TF-IDF sparse dictionary."""
        tf = Counter(tokens)
        vec = {}
        norm_sq = 0.0

        for token, count in tf.items():
            if token in self.idf:
                # Sublinear TF scaling: 1 + ln(count)
                weight = (1.0 + math.log(count)) * self.idf[token]
                vec[token] = weight
                norm_sq += weight * weight

        # L2 Normalization
        norm = math.sqrt(norm_sq)
        if norm > 0:
            for token in vec:
                vec[token] /= norm
        return vec

    def cosine_similarity(self, vec1, vec2) -> float:
        """Calculates dot product of two L2-normalized sparse vectors."""
        score = 0.0
        # Iterate over smaller dict
        if len(vec1) > len(vec2):
            vec1, vec2 = vec2, vec1
        for token, val in vec1.items():
            if token in vec2:
                score += val * vec2[token]
        return float(score)

    def find_best_match(self, user_query: str, threshold: float = 0.30):
        """Matches user query against FAQ corpus using Cosine Similarity."""
        if not self.faqs or not self.doc_vectors:
            return FALLBACK_RESPONSE, 0.0, None

        q_tokens = clean_and_tokenize(user_query)
        if not q_tokens:
            return "Please type a valid question.", 0.0, None

        q_vec = self._vectorize(q_tokens)
        
        best_score = 0.0
        best_idx = -1

        for idx, doc_vec in enumerate(self.doc_vectors):
            sim = self.cosine_similarity(q_vec, doc_vec)
            if sim > best_score:
                best_score = sim
                best_idx = idx

        if best_idx >= 0 and best_score >= threshold:
            matched_faq = self.faqs[best_idx]
            return matched_faq["answer"], best_score, matched_faq
        else:
            return FALLBACK_RESPONSE, best_score, None


def render_chatbot_ui():
    """Renders the Streamlit UI for the FAQ Chatbot."""
    st.markdown("""
        <div style="background: linear-gradient(135deg, #0f766e 0%, #14b8a6 100%); padding: 1.5rem; border-radius: 12px; color: white; margin-bottom: 1.5rem; box-shadow: 0 4px 12px rgba(20,184,166,0.2);">
            <h2 style="margin:0; font-weight:700; color:white;">💬 NLP FAQ Intelligent Chatbot</h2>
            <p style="margin:0.5rem 0 0 0; opacity:0.9; font-size: 0.95rem;">Powered by TF-IDF tokenization, n-gram text representation, and Cosine Similarity intent matching.</p>
        </div>
    """, unsafe_allow_html=True)

    faqs = load_faq_database()
    engine = PureTfidfEngine(faqs)

    # Sidebar settings for chatbot
    with st.sidebar:
        st.markdown("### ⚙️ Chatbot Settings")
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.10,
            max_value=0.80,
            value=0.25,
            step=0.05,
            help="Minimum cosine similarity score required to accept an FAQ match."
        )
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {
                "role": "assistant",
                "content": "Hello! I am your AI Assistant. Ask me anything about Artificial Intelligence, Machine Learning, Computer Vision, or the CodeAlpha Internship guidelines!",
                "score": 1.0,
                "category": "System"
            }
        ]

    # Quick Suggestion Chips
    st.markdown("##### 💡 Suggested Questions")
    sample_queries = [
        "What is Artificial Intelligence (AI)?",
        "How does YOLO object detection work?",
        "Difference between supervised and unsupervised learning?",
        "How do I submit my CodeAlpha internship project?",
        "What is TF-IDF and Cosine Similarity?"
    ]

    chip_cols = st.columns(len(sample_queries))
    selected_prompt = None
    for i, q in enumerate(sample_queries):
        if chip_cols[i].button(f"📌 {q[:20]}...", key=f"chip_{i}", help=q, use_container_width=True):
            selected_prompt = q

    # Knowledge Base Inspector Modal / Expander
    with st.expander("📚 View FAQ Knowledge Base (Active Dataset)"):
        categories = sorted(list(set(item["category"] for item in faqs)))
        sel_cat = st.selectbox("Filter by Category", ["All"] + categories)
        filtered = faqs if sel_cat == "All" else [f for f in faqs if f["category"] == sel_cat]
        for f in filtered:
            st.markdown(f"**[{f['category']}] {f['question']}**")
            st.markdown(f"_{f['answer']}_")
            st.markdown("---")

    # Render Conversation
    st.markdown("---")
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])
            if msg.get("score") is not None and msg["role"] == "assistant" and msg.get("category") != "System":
                score_pct = int(msg["score"] * 100)
                cat_badge = f" • **Category:** `{msg['category']}`" if msg.get("category") else ""
                st.caption(f"🎯 Match Confidence: **{score_pct}%** (Threshold: {int(confidence_threshold*100)}%){cat_badge}")

    # Chat Input handler
    user_input = st.chat_input("Type your question here (e.g. 'What is a neural network?')...")
    
    if selected_prompt:
        user_input = selected_prompt

    if user_input:
        # Append User Message
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Compute Match
        ans, score, matched_faq = engine.find_best_match(user_input, threshold=confidence_threshold)
        cat = matched_faq["category"] if matched_faq else None
        
        # Append Assistant Response
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": ans,
            "score": score,
            "category": cat
        })
        st.rerun()
