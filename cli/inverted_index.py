from pickle import dump, load
from nltk.stem import PorterStemmer
from load_files import load_stopwords, load_movies
from text_processing import preprocess

class InvertedIndex():
    def __init__(self):
        self.index = {}
        self.docmap = {}
        self.term_frequencies = {}
        self.stemm = PorterStemmer()
        self.stopwords = set(load_stopwords())
        self.movies = load_movies()
        self.__index_file = "cache/index.pkl"
        self.__docmap_file = "cache/docmap.pkl"
        self.__tf_file = "cache/term_frequencies.pkl"

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

