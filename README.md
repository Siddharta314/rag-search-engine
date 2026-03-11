```
uv run cli/keyword_search_cli.py search "your search query" 
uv run cli/keyword_search_cli.py
```
1. will execute the "search" case and populate "your search query" into the args.query variable.
2. with no (or invalid) arguments will print the parser's help message.
