"""Identifiers specific to the ingestion pipeline.

A document's id is `backend.domain.PaperId` -- reused, not duplicated,
since it is the same identifier the parser will later assign to the
`Paper` it produces from this document.
"""

from typing import NewType
from uuid import UUID

UploadJobId = NewType("UploadJobId", UUID)
