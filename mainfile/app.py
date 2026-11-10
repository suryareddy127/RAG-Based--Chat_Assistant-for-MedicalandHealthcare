from flask import Flask, render_template, request, session, redirect, url_for
from mainfile.components.retreive import create_qa_chain  # Your LCEL chain
from dotenv import load_dotenv
import os

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")

app = Flask(__name__)
app.secret_key = os.urandom(24)

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

@app.route("/", methods=["GET", "POST"])
def index():
    if "messages" not in session:
        session["messages"] = []

    if request.method == "POST":
        user_input = request.form.get("prompt")

        if user_input:
            messages = session["messages"]
            messages.append({"role": "user", "content": user_input})
            session["messages"] = messages

            try:
                qa_chain = create_qa_chain()
                if qa_chain is None:
                    raise Exception("QA chain could not be created (LLM or VectorStore issue)")
                
                # ✅ FIXED #1: Pass string directly
                response = qa_chain.invoke(user_input)  # str input, not dict
                
                # ✅ FIXED #2: LCEL returns str directly
                result = str(response)  # No .get("result")

                messages.append({"role": "assistant", "content": result})
                session["messages"] = messages

            except Exception as e:
                error_msg = f"Error: {str(e)}"
                messages.append({"role": "assistant", "content": error_msg})
                session["messages"] = messages
            
        return redirect(url_for("index"))
    
    return render_template("index.html", messages=session.get("messages", []))

@app.route("/clear")
def clear():
    session.pop("messages", None)
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)