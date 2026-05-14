import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import SpamDetector
from src.data import load_spam_data


class TestSpamDetector:
    @pytest.fixture
    def data(self):
        return load_spam_data()

    @pytest.fixture
    def fitted_detector(self, data):
        train_texts, _, y_train, _ = data
        detector = SpamDetector()
        detector.fit(train_texts, y_train)
        return detector

    def test_fit(self, fitted_detector, data):
        _, test_texts, _, _ = data
        preds = fitted_detector.predict(test_texts)
        assert len(preds) > 0
        assert set(preds).issubset({0, 1})

    def test_accuracy_above_80(self, fitted_detector, data):
        _, test_texts, _, y_test = data
        results = fitted_detector.evaluate(test_texts, y_test)
        assert results["accuracy"] > 0.80

    def test_proba_sums_to_one(self, fitted_detector, data):
        _, test_texts, _, _ = data
        proba = fitted_detector.predict_proba(test_texts[:5])
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_known_spam(self, fitted_detector):
        spam_texts = [
            "WIN A FREE IPHONE NOW! Click here to claim your prize!",
            "URGENT: Your account has been compromised. Verify now.",
            "Get rich quick! Earn $5000 per week from home.",
        ]
        preds = fitted_detector.predict(spam_texts)
        assert sum(preds) >= 2

    def test_known_ham(self, fitted_detector):
        ham_texts = [
            "Hey, are we still meeting for lunch today?",
            "Can you send me the report when you get a chance?",
            "The project deadline has been extended to Friday.",
        ]
        preds = fitted_detector.predict(ham_texts)
        assert sum(preds) <= 1

    def test_top_features(self, fitted_detector):
        top = fitted_detector.get_top_features(10)
        assert len(top["top_spam_words"]) == 10
        assert len(top["top_ham_words"]) == 10

    def test_save_load(self, fitted_detector, data, tmp_path):
        _, test_texts, _, _ = data
        p = str(tmp_path / "spam_model.joblib")
        fitted_detector.save(p)
        d2 = SpamDetector()
        d2.load(p)
        assert np.array_equal(
            fitted_detector.predict(test_texts[:3]),
            d2.predict(test_texts[:3]),
        )
