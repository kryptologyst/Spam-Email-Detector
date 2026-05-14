import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support,
)
from loguru import logger
from typing import Optional, Literal
import joblib


class SpamDetector:
    def __init__(
        self,
        vectorizer_type: Literal["count", "tfidf"] = "tfidf",
        max_features: int = 5000,
        alpha: float = 1.0,
    ):
        self.vectorizer_type = vectorizer_type
        self.max_features = max_features
        self.alpha = alpha
        self.vectorizer = (
            TfidfVectorizer(max_features=max_features, stop_words="english")
            if vectorizer_type == "tfidf"
            else CountVectorizer(max_features=max_features, stop_words="english")
        )
        self.model = MultinomialNB(alpha=alpha)
        self.classes_ = ["ham", "spam"]

    def fit(
        self, texts: list, labels: np.ndarray,
    ) -> "SpamDetector":
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)
        logger.info(
            f"Naive Bayes fitted: {X.shape[1]} features, "
            f"{len(texts)} samples"
        )
        return self

    def predict(self, texts: list) -> np.ndarray:
        X = self.vectorizer.transform(texts)
        return self.model.predict(X)

    def predict_proba(self, texts: list) -> np.ndarray:
        X = self.vectorizer.transform(texts)
        return self.model.predict_proba(X)

    def evaluate(self, texts: list, labels: np.ndarray) -> dict:
        y_pred = self.predict(texts)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, y_pred, average="binary",
        )
        return {
            "accuracy": float(accuracy_score(labels, y_pred)),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "classification_report": classification_report(
                labels, y_pred, target_names=self.classes_, output_dict=True,
            ),
            "confusion_matrix": confusion_matrix(labels, y_pred).tolist(),
        }

    def get_top_features(self, n: int = 20) -> dict:
        feature_names = self.vectorizer.get_feature_names_out()
        spam_probs = self.model.feature_log_prob_[1]
        ham_probs = self.model.feature_log_prob_[0]
        diff = spam_probs - ham_probs
        top_spam_idx = np.argsort(diff)[-n:][::-1]
        top_ham_idx = np.argsort(diff)[:n]
        return {
            "top_spam_words": [
                (feature_names[i], float(diff[i])) for i in top_spam_idx
            ],
            "top_ham_words": [
                (feature_names[i], float(diff[i])) for i in top_ham_idx
            ],
        }

    def save(self, path: str) -> None:
        joblib.dump(
            {"vectorizer": self.vectorizer, "model": self.model}, path,
        )
        logger.info(f"Model saved to {path}")

    def load(self, path: str) -> None:
        data = joblib.load(path)
        self.vectorizer = data["vectorizer"]
        self.model = data["model"]
        logger.info(f"Model loaded from {path}")
