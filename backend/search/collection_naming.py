"""Deterministic collection naming.

Shared, not owned by any one subpackage: the planner uses this to decide
where to route each batch, the repository/validator use it to know what to
check, and Module 9 will need the identical function to know what to
query. Kept at the top level of `search/` rather than inside `planner/`,
since burying it there would misrepresent it as planner-private logic.
"""

import re

_SANITIZE_PATTERN = re.compile(r"[^a-zA-Z0-9]+")
_REVISION_PREFIX_LENGTH = 8


def build_collection_name(prefix: str, model_name: str, model_version: str, target: str) -> str:
    """Compute the deterministic collection name for a given model and target.

    The same `(prefix, model_name, model_version, target)` always produces
    the same name -- this is what makes it safe for multiple modules to
    independently compute the same collection identity without coordination.

    Args:
        prefix: Namespace prefix (e.g. "kimrag"), for operational safety if
            a vector database instance is ever shared across deployments.
        model_name: Name of the embedding model (e.g. "BAAI/bge-m3").
        model_version: Resolved revision of the embedding model.
        target: The embedding target (e.g. "text", "image").

    Returns:
        A collection name safe for Qdrant's naming rules (and, in
        practice, any other vector database's).
    """
    sanitized_model = _SANITIZE_PATTERN.sub("_", model_name).strip("_")
    short_version = model_version[:_REVISION_PREFIX_LENGTH]
    return f"{prefix}_{sanitized_model}_{short_version}_{target}".lower()
