import math
from pickle import dump, load
from nltk.stem import PorterStemmer
from load_files import load_stopwords, load_movies
from text_processing import preprocess
from .constants import BM25_K1, BM25_B, DEFAULT_BM25_LIMIT

class InvertedIndex():
    def __init__(self):
        self.index = {}
        self.docmap = {}
        self.doc_length = {}
        self.term_frequencies = {}
        self.stemm = PorterStemmer()
        self.stopwords = set(load_stopwords())
        self.movies = load_movies()
        self.__index_file = "cache/index.pkl"
        self.__docmap_file = "cache/docmap.pkl"
        self.__tf_file = "cache/term_frequencies.pkl"
        self.__doc_lengths_path = "cache/doc_lengths.pkl"

    def __add_document(self, doc_id: int, text: str) -> None:
        """
        Adds a doc_id to the index.
        
        Args:
            doc_id (int): The document ID.
            text (str): The text of the document.
        """
        tokenized_text : list[str] = preprocess(text, self.stopwords, self.stemm)
        if doc_id not in self.term_frequencies:
            self.term_frequencies[doc_id] = {}
        for token in tokenized_text:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)
            current_tf = self.term_frequencies[doc_id].get(token, 0)
            self.term_frequencies[doc_id][token] = current_tf + 1
        self.doc_length[doc_id] = len(tokenized_text)

    def get_document(self, term: str) -> list[int]:
        """
        Returns the set of document IDs that contain the given term in ascending order.
        """
        if term in self.stopwords:
            return []
        clean_term = self.stemm.stem(term.lower())
        docs_id = self.index.get(clean_term, set())
        return sorted(docs_id)

    def get_tf(self, doc_id: int, term: str) -> int:
        """
        Returns the term frequency for a given document and term.
        """
        tokens = term.split()
        if len(tokens) != 1:
            raise ValueError(f"Expected a single token, but got: '{term}'")

        clean_term = self.stemm.stem(tokens[0].lower())
        return self.term_frequencies.get(doc_id, {}).get(clean_term, 0)

    def get_idf(self, term: str) -> float:
        """
        Returns the inverse document frequency for a given term.
        """
        docs = self.get_document(term)
        return math.log((len(self.movies) + 1) / (len(docs) + 1))

    def get_bm25_idf(self, term: str) -> float:
        """
        Returns the BM25 inverse document frequency for a given term.
        """
        tokens = preprocess(term, self.stopwords, self.stemm)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        token = tokens[0]
        docs = self.get_document(token)
        return math.log((len(self.movies) - len(docs) + 0.5) / (len(docs) + 0.5) + 1)

    def get_bm25_tf(self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
        """
        Returns the BM25 term frequency for a given document and term.
        """
        tf = self.get_tf(doc_id, term)
        avg_doc_length = self.__get_avg_doc_length()
        length_norm = 1 - b + b * (self.doc_length[doc_id] / avg_doc_length)
        return tf / (tf + k1 *length_norm) * (k1 + 1)

    def bm25(self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
        """
        Returns the BM25 score for a given document and term.
        """
        return self.get_bm25_tf(doc_id, term, k1, b) * self.get_bm25_idf(term)

    def bm25_search(self, query: str, limit: int = DEFAULT_BM25_LIMIT) -> list[tuple[int, float]]:
        """
        Returns the top k documents for a given query using BM25.
        """
        tokens =  preprocess(query, self.stopwords, self.stemm)
        scores = {} # doc_id : bm25_score
        for doc in self.docmap.keys():
            scores[doc] = 0.0
            for token in tokens:
                scores[doc] += self.bm25(doc, token)
        result = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]
        return [{ "id": id, "title": self.docmap[id]["title"], "score": score } for id, score in result]

    def __get_avg_doc_length(self) -> float:
        """
        Returns the average document length.
        """
        if not self.doc_length:
            return 0.0
        return sum(self.doc_length.values()) / len(self.doc_length)

    def build(self) -> None:
        for movie in self.movies:
            self.__add_document(movie["id"], f"{movie['title']} {movie['description']}")
            self.docmap[movie["id"]] = movie

    def save(self) -> None:
        import os
        # create cache directory
        os.makedirs("cache", exist_ok=True)

        # save index
        with open(self.__index_file, "wb") as f:
            dump(self.index, f)

        # save docmap
        with open(self.__docmap_file, "wb") as f:
            dump(self.docmap, f)

        # save term frequencies
        with open(self.__tf_file, "wb") as f:
            dump(self.term_frequencies, f)

        # save document lengths
        with open(self.__doc_lengths_path, "wb") as f:
            dump(self.doc_length, f)

    def load(self) -> None:
        # Raise an error if files don't exist
        import os
        if not os.path.exists(self.__index_file) or not os.path.exists(self.__docmap_file):
            raise FileNotFoundError("Index or docmap file not found")
        # load index
        with open(self.__index_file, "rb") as f:
            self.index = load(f)

        # load docmap
        with open(self.__docmap_file, "rb") as f:
            self.docmap = load(f)

        # load term frequencies
        with open(self.__tf_file, "rb") as f:
            self.term_frequencies = load(f)

        # load document lengths
        with open(self.__doc_lengths_path, "rb") as f:
            self.doc_length = load(f)
