import string

def simple_clean(text: str) -> str:
    table = str.maketrans("", "", string.punctuation)
    return text.translate(table).lower()

def remove_stopwords(text: str, stopwords: list[str]) -> str:
    return " ".join([word for word in text.split() if word not in stopwords])

def tokenize_based_word(text: str) -> list[str]:
    return text.split()

def compare_list_tokens(list1: list[str], list2: list[str]) -> bool:
    set1, set2 = set(list1), set(list2)
    return set1.intersection(set2)