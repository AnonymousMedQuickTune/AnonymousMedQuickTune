import os
from flask import Flask, send_from_directory, render_template, abort

app = Flask(
    __name__,
    static_folder="static",
    template_folder="templates",
    static_url_path=""
)

@app.get("/")
def home():
    # serve the static landing page
    return send_from_directory(app.static_folder, "index.html")

# simple download route (for files under static/datasets/)
@app.get("/download/<path:filename>")
def download(filename):
    safe = os.path.normpath(filename)
    if ".." in safe:
        abort(400)
    return send_from_directory(
        os.path.join(app.static_folder, "datasets"),
        safe,
        as_attachment=True
    )

# Quickstart (later render a Markdown file or link to it)
@app.get("/quickstart")
def quickstart():
    return render_template("quickstart.html")  # create a template later
