import streamlit as st
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data import load_spam_data
from src.model import SpamDetector

st.set_page_config(page_title="Spam Detector", page_icon="📧", layout="wide")
st.title("📧 Spam Email Detector — Naive Bayes")
st.markdown("Classify messages as **Ham** or **Spam** using Multinomial Naive Bayes with TF-IDF features.")

train_texts, test_texts, y_train, y_test = load_spam_data()

tab1, tab2, tab3 = st.tabs(["Train & Evaluate", "Live Classify", "Top Features"])

with tab1:
    col1, col2, col3 = st.columns(3)
    with col1:
        vec_type = st.selectbox("Vectorizer", ["tfidf", "count"])
    with col2:
        max_feat = st.slider("Max Features", 500, 10000, 5000, 500)
    with col3:
        alpha = st.select_slider("Smoothing (alpha)", [0.1, 0.5, 1.0, 2.0, 5.0], 1.0)

    if st.button("Train Model", type="primary"):
        with st.spinner("Training..."):
            detector = SpamDetector(vectorizer_type=vec_type, max_features=max_feat, alpha=alpha)
            detector.fit(train_texts, y_train)
            results = detector.evaluate(test_texts, y_test)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{results['accuracy']:.2%}")
        c2.metric("Precision", f"{results['precision']:.2%}")
        c3.metric("Recall", f"{results['recall']:.2%}")
        c4.metric("F1 Score", f"{results['f1']:.2%}")

with tab2:
    st.subheader("Classify a Message")
    user_text = st.text_area("Enter message text", height=120, placeholder="Type or paste a message...")
    if st.button("Classify", type="primary") and user_text.strip():
        detector = SpamDetector()
        model_path = Path(__file__).parent / "outputs" / "models" / "spam_detector.joblib"
        if model_path.exists():
            detector.load(str(model_path))
        else:
            detector.fit(train_texts, y_train)
        pred = detector.predict([user_text])[0]
        proba = detector.predict_proba([user_text])[0]
        if pred == 1:
            st.error(f"🚨 **SPAM** (confidence: {proba[1]:.1%})")
        else:
            st.success(f"✅ **HAM** (confidence: {proba[0]:.1%})")
        st.progress(float(proba[1]), text=f"Spam probability: {proba[1]:.1%}")

with tab3:
    st.subheader("Most Discriminative Words")
    if st.button("Show Top Features", type="primary"):
        detector = SpamDetector()
        detector.fit(train_texts, y_train)
        top = detector.get_top_features(20)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Top Spam Words**")
            for word, score in top["top_spam_words"][:15]:
                st.write(f"🔴 `{word}` — {score:.2f}")
        with c2:
            st.markdown("**Top Ham Words**")
            for word, score in top["top_ham_words"][:15]:
                st.write(f"🔵 `{word}` — {score:.2f}")
