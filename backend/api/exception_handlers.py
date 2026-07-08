"""Maps application exceptions to HTTP responses.

Centralized here so route handlers never translate exceptions into status
codes themselves -- they raise a meaningful exception and let it propagate.
"""

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

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
