"""Scientific document parsing engine: converts an ingested PDF into the
domain `Paper`.

Docling is an implementation detail confined to `providers/docling_parser.py`.
Nothing outside that module imports `docling` or `docling_core`, and nothing
leaves this package except `backend.domain` types.
"""
