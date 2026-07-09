"""Knowledge representation engine: converts a parsed `Paper` into a graph
of retrievable knowledge units and the relationships between them.

Named `chunking` for continuity with the project's pre-established module
map, but its actual responsibility is not naive text splitting -- see
`builder/knowledge_builder.py` for the design. Consumes only
`backend.domain.Paper`; never depends on PDFs, Docling, OCR, or parser
internals.
"""
