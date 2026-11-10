"""Server-side storage for chat transcripts.

The session cookie only ever carries a chat id. The transcript itself lives
here, in a local file cache, so the cookie stays small and the browser never
receives the conversation contents.
"""

import uuid

from cachelib.file import FileSystemCache

from mainfile.config.config import (
    CHAT_STORE_DIR,
    CHAT_STORE_THRESHOLD,
    CHAT_TTL_SECONDS,
    MAX_HISTORY_MESSAGES,
)
from mainfile.common.logger import get_logger

logger = get_logger(__name__)

_cache = FileSystemCache(
    CHAT_STORE_DIR,
    threshold=CHAT_STORE_THRESHOLD,
    default_timeout=CHAT_TTL_SECONDS,
)


def _key(chat_id):
    return f"chat:{chat_id}"


def new_chat_id():
    """Opaque id; this is the only chat state the cookie carries."""
    return uuid.uuid4().hex


def get_messages(chat_id):
    if not chat_id:
        return []
    return _cache.get(_key(chat_id)) or []


def append_message(chat_id, role, content, figures=None):
    """Append one turn and return the trimmed transcript.

    `figures` carries the retrieved image metadata for assistant turns; only
    the fields the template needs are kept so stored transcripts stay small.
    """
    messages = get_messages(chat_id)
    entry = {"role": role, "content": content}
    if figures:
        entry["figures"] = [
            {
                "id": f.get("id"),
                "caption": f.get("caption", ""),
                "page": f.get("page"),
                "explanation": f.get("explanation", ""),
            }
            for f in figures
        ]
    messages.append(entry)

    # Keep only the most recent turns so a long chat cannot grow without bound.
    if len(messages) > MAX_HISTORY_MESSAGES:
        messages = messages[-MAX_HISTORY_MESSAGES:]

    _cache.set(_key(chat_id), messages, timeout=CHAT_TTL_SECONDS)
    return messages


def clear_messages(chat_id):
    if chat_id:
        _cache.delete(_key(chat_id))
