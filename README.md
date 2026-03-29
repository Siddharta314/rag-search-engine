```
uv run cli/keyword_search_cli.py search "your search query" 
uv run cli/keyword_search_cli.py
```
1. will execute the "search" case and populate "your search query" into the args.query variable.
2. with no (or invalid) arguments will print the parser's help message.


# TF-IDF

- TF (Term Frequency): How often a term appears in a document
- IDF (Inverse Document Frequency): How rare a term is across all documents


# BM25

BM25 addresses three key problems with basic TF-IDF:

- Better IDF calculation: More stable scoring for rare/common terms
- Term frequency saturation: Prevents terms from dominating by appearing too often
- Document length normalization: Accounts for longer vs shorter documents

$$IDF = \log\left(\frac{N - df + 0.5}{df + 0.5} + 1\right)$$

Numerator (N - df + 0.5): Count of documents WITHOUT the term (plus laplace smoothing)
Denominator (df + 0.5): Count of documents WITH the term (plus laplace smoothing)
+1 ensure always positive

## Term Frequency Saturation. 
This prevents any single term from dominating search results just because it appears many many times.

## Document Length Normalization
This adjusts for documents that are longer or shorter than average, ensuring fair comparison.
$$\text{The Core Ratio} = \frac{\text{doc\_length}}{\text{avg\_doc\_length}}$$


**b** is a tunable parameter that controls how much we care about document length.

    If b=0 then length norm is always 1.
    If b=1 then full normalization is applied.

The key insight is:

* Long documents get higher length_norm and are penalized (lower scores)
* Short documents get lower length_norm and are boosted (higher scores)

## Chunking
Problems:
* Semantic dilution: The embedding tries to capture ALL topics at once
* Token limits: Models have a limit to what they can fit in one embedding effectively
* Poor precision: Specific concepts get "averaged out"
* Irrelevant matches: Parts of the document may match queries poorly

