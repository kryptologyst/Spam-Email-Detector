import typer
import sys
from loguru import logger

from .config import settings
from .data import load_spam_data
from .model import SpamDetector
from .visualizer import SpamVisualizer

app = typer.Typer(help="Spam Email Detector CLI")

logger.remove()
logger.add(sys.stderr, level=settings.log_level)


@app.command()
def train(
    vectorizer: str = typer.Option("tfidf", help="Vectorizer: count or tfidf"),
    max_features: int = typer.Option(5000, help="Max features"),
    visualize: bool = typer.Option(True, help="Generate plots"),
):
    logger.info("Training Naive Bayes spam detector...")
    train_texts, test_texts, y_train, y_test = load_spam_data()
    detector = SpamDetector(
        vectorizer_type=vectorizer, max_features=max_features,
    )
    detector.fit(train_texts, y_train)
    results = detector.evaluate(test_texts, y_test)
    logger.info(
        f"Accuracy: {results['accuracy']:.2%} | "
        f"Precision: {results['precision']:.2%} | "
        f"Recall: {results['recall']:.2%} | "
        f"F1: {results['f1']:.2%}"
    )
    top = detector.get_top_features(15)
    logger.info("\nTop spam words: " + ", ".join(w for w, _ in top["top_spam_words"][:10]))
    if visualize:
        vis = SpamVisualizer()
        vis.plot_confusion_matrix(
            results["confusion_matrix"],
            save_path=settings.plots_dir / "confusion_matrix.png",
        )
        vis.plot_top_features(
            top, save_path=settings.plots_dir / "top_features.png",
        )
    detector.save(str(settings.models_dir / "spam_detector.joblib"))
    logger.success("Training complete!")


@app.command()
def predict(text: str = typer.Option(..., help="Text to classify")):
    detector = SpamDetector()
    model_path = settings.models_dir / "spam_detector.joblib"
    if model_path.exists():
        detector.load(str(model_path))
    else:
        logger.error("No saved model. Train first.")
        raise typer.Exit(1)
    pred = detector.predict([text])[0]
    proba = detector.predict_proba([text])[0]
    label = "SPAM" if pred == 1 else "HAM"
    logger.success(f"Classification: {label}")
    logger.info(f"P(ham)={proba[0]:.3f}, P(spam)={proba[1]:.3f}")


if __name__ == "__main__":
    app()
