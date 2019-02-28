import re
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from string import punctuation

# Download NLTK data if not already present
try:
    stopwords.words("english")
except LookupError:
    import nltk
    nltk.download("punkt")
    nltk.download("stopwords")

def clean_text(text):
    """
    Cleans the input text by removing special characters, extra spaces, and converting to lowercase.
    """
    text = re.sub(r"[^a-zA-Z0-9\s."]", "", text) # Keep alphanumeric, spaces, periods, and double quotes
    text = re.sub(r"\s+", " ", text).strip() # Remove extra spaces
    text = text.lower()
    return text

def tokenize_sentences(text):
    """
    Tokenizes the text into sentences.
    """
    return sent_tokenize(text)

def tokenize_words(text):
    """
    Tokenizes the text into words and removes stopwords and punctuation.
    """
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    return filtered_words

if __name__ == "__main__":
    sample_text = "Hello, world! This is a sample text for NLP summarization. It contains some punctuation and common words."
    print(f"Original text: {sample_text}")

    cleaned_text = clean_text(sample_text)
    print(f"Cleaned text: {cleaned_text}")

    sentences = tokenize_sentences(sample_text)
    print(f"Sentences: {sentences}")

    words = tokenize_words(sample_text)
    print(f"Words (filtered): {words}")
