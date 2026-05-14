import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional
from loguru import logger


class SpamVisualizer:
    @staticmethod
    def plot_confusion_matrix(
        cm: list, save_path: Optional[Path] = None,
    ) -> None:
        plt.figure(figsize=(5, 4))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Reds",
            xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"],
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix — Spam Detection")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Confusion matrix saved to {save_path}")
        plt.close()

    @staticmethod
    def plot_top_features(
        top_features: dict, save_path: Optional[Path] = None,
    ) -> None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        spam_words, spam_scores = zip(*top_features["top_spam_words"])
        ax1.barh(list(spam_words), list(spam_scores), color="crimson")
        ax1.set_title("Top Spam Indicators")
        ax1.invert_yaxis()
        ham_words, ham_scores = zip(*top_features["top_ham_words"])
        ax2.barh(list(ham_words), list(ham_scores), color="steelblue")
        ax2.set_title("Top Ham Indicators")
        ax2.invert_yaxis()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Feature plot saved to {save_path}")
        plt.close()
