# Spam Email Detector

Multinomial **Naive Bayes** classifier for SMS/email spam detection using TF-IDF features.

## Overview

- Downloads the real **SMS Spam Collection** dataset (5,574 messages) with fallback to built-in samples
- TF-IDF or Count vectorization with configurable max features
- Reports accuracy, precision, recall, F1, and confusion matrix
- Extracts most discriminative spam/ham words
- **Streamlit dashboard** for training, live classification, and feature inspection

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
# CLI: python -m src.main train
pytest tests/ -v
```

## Docker

```bash
docker compose up --build
```

## Project Structure

```
0006 Spam Email Detector/
├── app.py              # Streamlit dashboard
├── Dockerfile / docker-compose.yml
├── src/
│   ├── model.py        # MultinomialNB wrapper with TF-IDF
│   ├── data.py         # SMS Spam Collection loader
│   ├── visualizer.py   # Confusion matrix, feature importance
│   └── main.py         # Typer CLI
└── tests/test_model.py # 7 tests
```

## License

MIT
# Spam-Email-Detector
