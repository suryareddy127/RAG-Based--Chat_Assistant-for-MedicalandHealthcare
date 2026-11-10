from flask import (Flask, abort, render_template, request, session, redirect,
                   send_from_directory, url_for)
from flask_session import Session
from cachelib.file import FileSystemCache

from mainfile.components.retreive import answer_question
from mainfile.common.chat_store import (
    append_message,
    clear_messages,
    get_messages,
    new_chat_id,
)
from mainfile.common.logger import get_logger
from mainfile.components.document_loader import load_manifest
from mainfile.config.config import (
    CHAT_STORE_THRESHOLD,
    FIGURES_DIR,
    CHAT_TTL_SECONDS,
    SECRET_KEY,
    SESSION_STORE_DIR,
)
from dotenv import load_dotenv
import os

logger = get_logger(__name__)

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")

app = Flask(__name__)

# A fresh random key each boot would invalidate every session cookie on every
# restart, so prefer a stable key from the environment.
app.secret_key = SECRET_KEY or os.urandom(24)
if not SECRET_KEY:
    logger.warning(
        "SECRET_KEY is not set; using a random key. Sessions will be dropped "
        "on restart. Set SECRET_KEY in .env for stable sessions."
    )

# Server-side sessions: the browser cookie holds only a signed session id,
# never the conversation itself.
app.config.update(
    SESSION_TYPE="cachelib",
    SESSION_SERIALIZATION_FORMAT="json",
    SESSION_CACHELIB=FileSystemCache(
        cache_dir=SESSION_STORE_DIR,
        threshold=CHAT_STORE_THRESHOLD,
        default_timeout=CHAT_TTL_SECONDS,
    ),
    SESSION_PERMANENT=False,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
)
Session(app)

# Flask only auto-reloads templates when debug is on, so with debug=False a
# running server keeps serving the index.html it cached at startup. Enable it
# explicitly so front-end edits show up without a restart.
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.jinja_env.auto_reload = True

import bleach
import markdown as md
from markupsafe import Markup, escape

# Tags the assistant is allowed to emit. Anything else the model produces is
# stripped, and user text is never treated as markup at all.
_ALLOWED_TAGS = [
    "p", "br", "strong", "em", "b", "i", "u", "code", "pre", "blockquote",
    "ul", "ol", "li", "h1", "h2", "h3", "h4", "h5", "h6", "hr",
    "table", "thead", "tbody", "tr", "th", "td",
]


def render_markdown(value):
    """Render assistant markdown to sanitised HTML for the chat bubble."""
    html = md.markdown(str(value), extensions=["extra", "sane_lists", "nl2br"])
    return Markup(bleach.clean(html, tags=_ALLOWED_TAGS, attributes={}, strip=True))


def nl2br(value):
    """Plain-text rendering for user messages - escape first, then keep newlines."""
    return Markup(str(escape(value)).replace(chr(10), "<br>" + chr(10)))


app.jinja_env.filters["markdown"] = render_markdown
app.jinja_env.filters["nl2br"] = nl2br

def _current_chat_id():
    """The only chat state held in the session is this id."""
    chat_id = session.get("chat_id")
    if not chat_id:
        chat_id = new_chat_id()
        session["chat_id"] = chat_id
    return chat_id


@app.route("/", methods=["GET", "POST"])
def index():
    chat_id = _current_chat_id()

    if request.method == "POST":
        user_input = request.form.get("prompt")

        if user_input:
            history = get_messages(chat_id)
            append_message(chat_id, "user", user_input)

            try:
                # History is read before the new turn is appended, so the
                # condenser sees the prior conversation, not the question twice.
                result = answer_question(user_input, history)
                append_message(chat_id, "assistant", result["answer"],
                               figures=result.get("figures"))

            except Exception as e:
                logger.error(f"Failed to answer question: {e}")
                append_message(chat_id, "assistant", f"Error: {e}")

        return redirect(url_for("index"))

    return render_template("index.html", messages=get_messages(chat_id))


@app.route("/figure/<figure_id>")
def figure(figure_id):
    """Serve an extracted figure.

    The id is matched against the manifest rather than used as a path, so a
    crafted id cannot escape the figures directory.
    """
    entry = next((f for f in load_manifest() if f["id"] == figure_id), None)
    if entry is None:
        abort(404)
    return send_from_directory(os.path.abspath(FIGURES_DIR), entry["file"])


@app.route("/clear")
def clear():
    clear_messages(session.get("chat_id"))
    session.pop("chat_id", None)
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)