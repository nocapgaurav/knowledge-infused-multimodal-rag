"""Maps application exceptions to HTTP responses.

Centralized here so route handlers never translate exceptions into status
codes themselves -- they raise a meaningful exception and let it propagate.
"""

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from backend.chunking.exceptions import (
    PaperNotParsedError,
    RepresentationStorageError,
    RepresentationValidationError,
)
from backend.embeddings.exceptions import (
    EmbeddingStorageError,
    EmbeddingValidationError,
    NoEmbeddingsProducedError,
    RepresentationNotFoundError,
)
from backend.graph.exceptions import (
    GraphStorageError,
    GraphStoreError,
    GraphValidationError,
    KnowledgeRepresentationNotFoundError,
)
from backend.ingestion.exceptions import (
    EmptyFileError,
    FileTooLargeError,
    IngestionStorageError,
    InvalidPdfContentError,
    UnsupportedFileTypeError,
    UploadJobNotFoundError,
)
from backend.parser.exceptions import (
    DocumentNotIngestedError,
    DocumentValidationError,
    ParserStorageError,
    UnreadablePdfError,
)
from backend.search.exceptions import (
    EmbeddingArtifactsNotFoundError,
    IndexStorageError,
    IndexValidationError,
    MultiCollectionIndexingNotSupportedError,
    NoVectorsIndexedError,
    VectorStoreError,
)


def register_exception_handlers(app: FastAPI) -> None:
    """Register application exception handlers on the FastAPI app.

    Args:
        app: Application instance to register handlers on.
    """

    @app.exception_handler(UnsupportedFileTypeError)
    async def _handle_unsupported_file_type(
        request: Request, exc: UnsupportedFileTypeError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, content={"detail": str(exc)}
        )

    @app.exception_handler(EmptyFileError)
    async def _handle_empty_file(request: Request, exc: EmptyFileError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, content={"detail": str(exc)}
        )

    @app.exception_handler(InvalidPdfContentError)
    async def _handle_invalid_pdf_content(
        request: Request, exc: InvalidPdfContentError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, content={"detail": str(exc)}
        )

    @app.exception_handler(FileTooLargeError)
    async def _handle_file_too_large(request: Request, exc: FileTooLargeError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, content={"detail": str(exc)}
        )

    @app.exception_handler(UploadJobNotFoundError)
    async def _handle_not_found(request: Request, exc: UploadJobNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"detail": str(exc)})

    @app.exception_handler(IngestionStorageError)
    async def _handle_storage_error(request: Request, exc: IngestionStorageError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "document ingestion failed"},
        )

    @app.exception_handler(DocumentNotIngestedError)
    async def _handle_document_not_ingested(
        request: Request, exc: DocumentNotIngestedError
    ) -> JSONResponse:
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"detail": str(exc)})

    @app.exception_handler(UnreadablePdfError)
    async def _handle_unreadable_pdf(request: Request, exc: UnreadablePdfError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, content={"detail": str(exc)}
        )

    @app.exception_handler(DocumentValidationError)
    async def _handle_document_validation_error(
        request: Request, exc: DocumentValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, content={"detail": str(exc)}
        )

    @app.exception_handler(ParserStorageError)
    async def _handle_parser_storage_error(
        request: Request, exc: ParserStorageError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "document parsing failed"},
        )

    @app.exception_handler(PaperNotParsedError)
    async def _handle_paper_not_parsed(request: Request, exc: PaperNotParsedError) -> JSONResponse:
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"detail": str(exc)})

    @app.exception_handler(RepresentationValidationError)
    async def _handle_representation_validation_error(
        request: Request, exc: RepresentationValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, content={"detail": str(exc)}
        )

    @app.exception_handler(RepresentationStorageError)
    async def _handle_representation_storage_error(
        request: Request, exc: RepresentationStorageError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "knowledge representation failed"},
        )

    @app.exception_handler(RepresentationNotFoundError)
    async def _handle_representation_not_found(
        request: Request, exc: RepresentationNotFoundError
    ) -> JSONResponse:
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"detail": str(exc)})

    @app.exception_handler(EmbeddingValidationError)
    async def _handle_embedding_validation_error(
        request: Request, exc: EmbeddingValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, content={"detail": str(exc)}
        )

    @app.exception_handler(NoEmbeddingsProducedError)
    async def _handle_no_embeddings_produced(
        request: Request, exc: NoEmbeddingsProducedError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"detail": str(exc)}
        )

    @app.exception_handler(EmbeddingStorageError)
    async def _handle_embedding_storage_error(
        request: Request, exc: EmbeddingStorageError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "embedding generation failed"},
        )

    @app.exception_handler(EmbeddingArtifactsNotFoundError)
    async def _handle_embedding_artifacts_not_found(
        request: Request, exc: EmbeddingArtifactsNotFoundError
    ) -> JSONResponse:
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"detail": str(exc)})

    @app.exception_handler(MultiCollectionIndexingNotSupportedError)
    async def _handle_multi_collection_not_supported(
        request: Request, exc: MultiCollectionIndexingNotSupportedError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_501_NOT_IMPLEMENTED, content={"detail": str(exc)}
        )

    @app.exception_handler(NoVectorsIndexedError)
    async def _handle_no_vectors_indexed(
        request: Request, exc: NoVectorsIndexedError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"detail": str(exc)}
        )

    @app.exception_handler(IndexValidationError)
    async def _handle_index_validation_error(
        request: Request, exc: IndexValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"detail": str(exc)}
        )

    @app.exception_handler(VectorStoreError)
    async def _handle_vector_store_error(request: Request, exc: VectorStoreError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "vector store operation failed"},
        )

    @app.exception_handler(IndexStorageError)
    async def _handle_index_storage_error(request: Request, exc: IndexStorageError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "document indexing failed"},
        )

    @app.exception_handler(KnowledgeRepresentationNotFoundError)
    async def _handle_knowledge_representation_not_found(
        request: Request, exc: KnowledgeRepresentationNotFoundError
    ) -> JSONResponse:
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"detail": str(exc)})

    @app.exception_handler(GraphValidationError)
    async def _handle_graph_validation_error(
        request: Request, exc: GraphValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"detail": str(exc)}
        )

    @app.exception_handler(GraphStoreError)
    async def _handle_graph_store_error(request: Request, exc: GraphStoreError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "graph store operation failed"},
        )

    @app.exception_handler(GraphStorageError)
    async def _handle_graph_storage_error(request: Request, exc: GraphStorageError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "document graph construction failed"},
        )
