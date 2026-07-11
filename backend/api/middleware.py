"""Transport-level middleware for the API application.

`UnhandledErrorResponseMiddleware` exists because of where Starlette
places its own error handling: unhandled exceptions escape all user
middleware and are answered by `ServerErrorMiddleware`, the outermost
layer of the stack -- *outside* `CORSMiddleware`. That 500 therefore
leaves the server without an `Access-Control-Allow-Origin` header, the
browser refuses to expose it to JavaScript, and a browser-based client
misreports a real server error as a network failure ("could not reach
the server"). Observed live in Module 12 verification: a parsing 500
surfaced in the frontend as a connectivity error.

Converting unhandled exceptions into a regular JSON 500 *inside* the
middleware stack means the response flows back out through
`CORSMiddleware` and reaches the browser like any other error response.
No endpoint contract changes: anything with a registered exception
handler is answered exactly as before; this only catches what would
otherwise have been an anonymous, CORS-less 500.
"""

import logging

from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger(__name__)


class UnhandledErrorResponseMiddleware:
    """Answers otherwise-unhandled exceptions with a plain JSON 500."""

    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        response_started = False

        async def sending(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self._app(scope, receive, sending)
        except Exception:
            logger.exception(
                "unhandled error while handling %s %s", scope.get("method"), scope.get("path")
            )
            if response_started:
                # Too late to produce a replacement response; let the
                # server terminate the connection as it would have anyway.
                raise
            response = JSONResponse(status_code=500, content={"detail": "internal server error"})
            await response(scope, receive, send)
