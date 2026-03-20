import string
from typing import Any

def simple_clean(text: str) -> str:
    table = str.maketrans("", "", string.punctuation)
    return text.translate(table).lower()

def remove_stopwords(text: str, stopwords: list[str]) -> str:
    return " ".join([word for word in text.split() if word not in stopwords])

def tokenize_based_word(text: str) -> list[str]:
    return text.split()

def compare_list_tokens(list1: list[str], list2: list[str]) -> bool:
    set1, set2 = set(list1), set(list2)
    return not set(set1).isdisjoint(set2)

def preprocess(text: str, stopwords: set[str], stemmer: Any) -> list[str]:
    """
    Preprocess text by cleaning, removing stopwords, and stemming.
    
    Args:
        text (str): The text to preprocess.
        stopwords (set[str]): The set of stopwords to remove.
        stemmer (Any): The stemmer to use.
    
    Returns:
        list[str]: The preprocessed text as a list of tokens.
    """
    clean_text = simple_clean(text)

    tokens = [
        stemmer.stem(word)
        for word in clean_text.split()
        if word not in stopwords
    ]

    return tokens
