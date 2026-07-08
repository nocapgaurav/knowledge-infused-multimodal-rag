"""Document ingestion pipeline: accept, validate, identify, and persist
uploaded documents.

This package does not parse PDFs, read document text, or construct a
`Paper` -- it only gets a document safely into a workspace and tracks the
lifecycle of doing so. Module 4 (the parser) is the only consumer allowed
to turn an ingested document into a `Paper`.
"""
