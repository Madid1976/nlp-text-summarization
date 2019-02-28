from collections import defaultdict
from heapq import nlargest
from nlp_summarizer.text_processor import clean_text, tokenize_sentences, tokenize_words

def calculate_word_frequencies(words):
    """
    Calculates the frequency of each word in a list of words.
    """
    word_frequencies = defaultdict(int)
    for word in words:
        word_frequencies[word] += 1
    return word_frequencies

def calculate_sentence_scores(sentences, word_frequencies):
    """
    Calculates a score for each sentence based on the frequency of its words.
    """
    sentence_scores = defaultdict(int)
    for i, sentence in enumerate(sentences):
        for word in tokenize_words(sentence):
            if word in word_frequencies:
                sentence_scores[i] += word_frequencies[word]
    return sentence_scores

def generate_summary(text, num_sentences=3):
    """
    Generates a summary of the given text using a simple frequency-based approach.
    """
    cleaned_text = clean_text(text)
    sentences = tokenize_sentences(cleaned_text)
    words = tokenize_words(cleaned_text)

    if not sentences or not words:
        return ""

    word_frequencies = calculate_word_frequencies(words)
    sentence_scores = calculate_sentence_scores(sentences, word_frequencies)

    # Select the top N sentences for the summary
    summary_sentences = nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    summary_sentences = sorted(summary_sentences) # Maintain original sentence order

    final_summary = [sentences[i] for i in summary_sentences]
    return ". ".join(final_summary)

if __name__ == "__main__":
    long_text = """
    Artificial intelligence (AI) is intelligence—perceiving, synthesizing, and inferring information—demonstrated by machines, as opposed to intelligence displayed by animals or humans. Example tasks in which AI is used include speech recognition, computer vision, translation between natural languages, and other mappings of inputs. AI applications include advanced web search engines (e.g., Google Search), recommendation systems (used by YouTube, Amazon, and Netflix), understanding human speech (such as Siri and Alexa), self-driving cars (e.g., Waymo), generative or creative tools (ChatGPT and AI art), and competing at the highest level in strategic game systems (such as chess and Go). John McCarthy, who coined the term "artificial intelligence" in 1955, defined it as "the science and engineering of making intelligent machines." The field of AI research was founded at a workshop at Dartmouth College in 1956. The attendees became the leaders of AI research for decades. They predicted that a machine as intelligent as a human being would exist in no more than 20 years. Their optimism was based on the success of early AI programs, such as Allen Newell and Herbert Simon's Logic Theorist and Arthur Samuel's checkers program. The field then went through several cycles of optimism followed by disappointment and loss of funding, known as "AI winters". Funding and interest were revived in the 21st century due to new approaches, the increase in computing power, and the availability of large datasets. AI techniques are now an essential part of the technology industry, providing the heavy lifting for many of the most challenging problems in computer science.
    """
    print(f"Original text length: {len(long_text)} characters")

    summary = generate_summary(long_text, num_sentences=3)
    print("\nGenerated Summary:")
    print(summary)
    print(f"Summary length: {len(summary)} characters")

    # Test with a shorter text
    short_text = "The quick brown fox jumps over the lazy dog. This is a simple sentence. Another one for testing."
    short_summary = generate_summary(short_text, num_sentences=2)
    print("\nGenerated Short Summary:")
    print(short_summary)

    # Test with empty text
    empty_summary = generate_summary("")
    print("\nGenerated Empty Summary:", empty_summary)
    assert empty_summary == ""

    # Test with text having less sentences than num_sentences
    less_sentences_text = "This is sentence one. This is sentence two."
    less_sentences_summary = generate_summary(less_sentences_text, num_sentences=5)
    print("\nGenerated Less Sentences Summary:", less_sentences_summary)
    assert len(tokenize_sentences(less_sentences_summary)) <= 5
