"""Concrete `EmbeddingProvider` implementations.

`sentence_transformers_provider.py` is the only module in this application
permitted to import `sentence_transformers` or `huggingface_hub`.

No concrete `ImageEmbeddingProvider` is implemented here -- see
`interfaces/embedding_provider.py` for why that's a deliberate scope
boundary, not an oversight.
"""
