import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from rouge_score import rouge_scorer

def perform_summarization(text, model_name="sshleifer/distilbart-cnn-12-6", max_length=150, min_length=30):
    """
    Performs abstractive text summarization using a pre-trained Hugging Face model.

    Args:
        text (str): The input text to be summarized.
        model_name (str): The name of the pre-trained model to use.
        max_length (int): The maximum length of the generated summary.
        min_length (int): The minimum length of the generated summary.

    Returns:
        str: The generated summary.
    """
    print(f"Loading summarization model: {model_name}")
    summarizer = pipeline("summarization", model=model_name, tokenizer=model_name)

    print("Generating summary...")
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]["summary_text"]

def evaluate_summarization(reference_summary, generated_summary):
    """
    Evaluates the generated summary against a reference summary using ROUGE scores.

    Args:
        reference_summary (str): The human-written reference summary.
        generated_summary (str): The machine-generated summary.

    Returns:
        dict: A dictionary containing ROUGE scores.
    """
    print("Evaluating summary using ROUGE scores...")
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference_summary, generated_summary)
    return {k: v.fmeasure for k, v in scores.items()}

if __name__ == "__main__":
    sample_text = (
        "Artificial intelligence (AI) is intelligence—perceiving, synthesizing, and inferring information—demonstrated by machines, "
        "as opposed to intelligence displayed by animals or humans. Leading AI textbooks define the field as the study of \"intelligent agents\": "
        "any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals. "
        "Colloquially, the term \"artificial intelligence\" is often used to describe machines that mimic \"cognitive\" functions that humans "
        "associate with the human mind, such as \"learning\" and \"problem-solving\"."
    )
    reference_summary = (
        "Artificial intelligence (AI) is machine intelligence, in contrast to natural intelligence. "
        "AI textbooks define it as the study of intelligent agents that perceive their environment and maximize goal achievement. "
        "The term is often used for machines mimicking human cognitive functions like learning and problem-solving."
    )

    print("--- Sample Text Summarization ---")
    generated_summary = perform_summarization(sample_text)
    print(f"\nOriginal Text:\n{sample_text}")
    print(f"\nGenerated Summary:\n{generated_summary}")

    rouge_scores = evaluate_summarization(reference_summary, generated_summary)
    print(f"\nROUGE Scores:\n{rouge_scores}")

    print("\n--- Extractive Summarization Example (NLTK) ---")
    from nltk.corpus import stopwords
    from nltk.tokenize import sent_tokenize, word_tokenize
    from collections import defaultdict
    from heapq import nlargest

    def extractive_summarize(text, num_sentences=3):
        stop_words = set(stopwords.words("english"))
        words = word_tokenize(text.lower())
        freq = defaultdict(int)
        for word in words:
            if word.isalnum() and word not in stop_words:
                freq[word] += 1

        sentences = sent_tokenize(text)
        sentence_scores = {}
        for i, sentence in enumerate(sentences):
            for word in word_tokenize(sentence.lower()):
                if word in freq:
                    if i not in sentence_scores:
                        sentence_scores[i] = freq[word]
                    else:
                        sentence_scores[i] += freq[word]

        summarized_sentences = nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
        final_summary = [sentences[j] for j in sorted(summarized_sentences)]
        return " ".join(final_summary)

    try:
        import nltk
        nltk.data.find("tokenizers/punkt")
        nltk.data.find("corpora/stopwords")
    except LookupError:
        print("Downloading NLTK data...")
        nltk.download("punkt")
        nltk.download("stopwords")

    extractive_summary = extractive_summarize(sample_text, num_sentences=2)
    print(f"\nExtractive Summary:\n{extractive_summary}")

