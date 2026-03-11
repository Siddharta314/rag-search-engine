import string

def simple_clean(text: str) -> str:
    table = str.maketrans("", "", string.punctuation)
    return text.translate(table).lower()