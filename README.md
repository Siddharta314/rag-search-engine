```
uv run cli/keyword_search_cli.py search "your search query" 
uv run cli/keyword_search_cli.py
```
1. will execute the "search" case and populate "your search query" into the args.query variable.
2. with no (or invalid) arguments will print the parser's help message.


## TF-IDF

- TF (Term Frequency): How often a term appears in a document
- IDF (Inverse Document Frequency): How rare a term is across all documents


## BM25

BM25 addresses three key problems with basic TF-IDF:

- Better IDF calculation: More stable scoring for rare/common terms
- Term frequency saturation: Prevents terms from dominating by appearing too often
- Document length normalization: Accounts for longer vs shorter documents

$$IDF = \log\left(\frac{N - df + 0.5}{df + 0.5} + 1\right)$$

Numerator (N - df + 0.5): Count of documents WITHOUT the term (plus laplace smoothing)
Denominator (df + 0.5): Count of documents WITH the term (plus laplace smoothing)
+1 ensure always positive