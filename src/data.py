import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from loguru import logger
from typing import Tuple
import io
import urllib.request
import zipfile


SMS_SPAM_URL = (
    "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
)


def _fetch_sms_spam() -> pd.DataFrame:
    logger.info("Downloading SMS Spam Collection...")
    try:
        with urllib.request.urlopen(SMS_SPAM_URL, timeout=15) as resp:
            with zipfile.ZipFile(io.BytesIO(resp.read())) as zf:
                for name in zf.namelist():
                    if name.endswith(".csv") or "SMSSpamCollection" in name:
                        with zf.open(name) as f:
                            return pd.read_csv(
                                io.TextIOWrapper(f),
                                sep="\t",
                                header=None,
                                names=["label", "text"],
                            )
        raise FileNotFoundError("CSV not found in zip")
    except Exception:
        logger.warning("Download failed, using built-in sample data")
        return _sample_data()


def _sample_data() -> pd.DataFrame:
    samples = [
        ("ham", "Hey, are we still meeting for lunch today?"),
        ("ham", "Don't forget to bring the documents tomorrow."),
        ("ham", "The project deadline has been extended to Friday."),
        ("ham", "Can you send me the report when you get a chance?"),
        ("ham", "Happy birthday! Hope you have a great day."),
        ("spam", "URGENT! You have won a $1000 gift card! Claim now at http://scam.com"),
        ("spam", "FREE entry in weekly competition to win an iPad. Text WIN to 80123"),
        ("spam", "CONGRATULATIONS! You've been selected for a free cruise. Call now!"),
        ("spam", "Get rich quick! Earn $5000/week working from home. Limited offer."),
        ("spam", "Your account has been compromised. Verify now: http://phish.net"),
        ("ham", "Meeting rescheduled to 3pm. Please confirm."),
        ("ham", "Thanks for your help with the presentation yesterday."),
        ("ham", "I'll be working from home tomorrow."),
        ("ham", "The server maintenance is scheduled for Saturday night."),
        ("ham", "Please review the attached contract and let me know your thoughts."),
        ("spam", "HOT STOCK ALERT: This penny stock will explode 500% by Friday!"),
        ("spam", "You have 1 new voicemail. Call 0906-xxx-xxxx to listen."),
        ("spam", "Double your income in 30 days! Proven system. Reply YES for info."),
        ("spam", "We refinance homes at 2% interest. Bad credit OK. Apply now!"),
        ("spam", "Lose 20lbs in 2 weeks with this miracle pill! Order now."),
        ("ham", "Lunch tomorrow at the usual place?"),
        ("ham", "The build passed all tests. Ready for deployment."),
        ("ham", "Can we move the 1-on-1 to Thursday?"),
        ("ham", "Invoice #4521 has been paid. Confirmation attached."),
        ("ham", "Great work on the Q4 review, team!"),
        ("spam", "CLAIM YOUR FREE IPHONE 15! Limited time offer. Click here."),
        ("spam", "Dear winner, your email has won £500,000 in our lottery."),
        ("spam", "Meet singles in your area tonight! 100% free to join."),
        ("spam", "Your PayPal account has been limited. Login to resolve."),
        ("spam", "Make $200/day posting links on Google. No experience needed."),
    ]
    return pd.DataFrame(samples, columns=["label", "text"])


def load_spam_data(
    test_size: float = 0.2, random_state: int = 42,
) -> Tuple[list, list, np.ndarray, np.ndarray]:
    df = _fetch_sms_spam()
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    texts = df["text"].tolist()
    labels = df["label"].values
    train_texts, test_texts, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state,
        stratify=labels,
    )
    logger.info(
        f"Data loaded: {len(train_texts)} train, {len(test_texts)} test, "
        f"spam ratio={labels.mean():.1%}"
    )
    return train_texts, test_texts, y_train, y_test
