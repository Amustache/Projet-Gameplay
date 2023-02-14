from logging import FileHandler, Formatter
import io
import logging
import os
import shutil
import ML.model

from datascience import *
from flask import flash, Flask, redirect, render_template, request, Response, url_for
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {"mkv", "mp4", "webm", "avi"}
DEMO_TRUTH = "truth.csv"
DEMO_FNAME = "video.webm"

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join("static", "inputs")
app.config["DEMO_FOLDER"]   = os.path.join("static", "inputs", "demo")
app.config["MAX_CONTENT_PATH"] = 200_000_000
app.config.from_object("config")

def root_path_join(*args):
    return os.path.join(app.root_path, *args)

def clean_upload_folder():
    for f in os.listdir(app.config["UPLOAD_FOLDER"]):
        full_path = root_path_join(app.config["UPLOAD_FOLDER"], f)
        if os.path.isfile(full_path) and f != ".gitkeep":
            os.remove(full_path)

def flash_and_redirect(dest, msg):
    flash(msg)
    return redirect(url_for(dest))

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/uploader", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return flash_and_redirect("experience", "Pas de fichier téléversé")

        file = request.files["file"]
        if file.filename == "":
            return flash_and_redirect("experience", "Pas de fichier sélectionné")

        if not (file and allowed_file(file.filename)):
            return flash_and_redirect("experience", "Merci d'utiliser une vidéo valide")
            
        clean_upload_folder()
        fname = secure_filename(file.filename)
        full_path = root_path_join(app.config["UPLOAD_FOLDER"], fname)

        # Save the new file
        file.save(full_path)

        # Process the video and extract the data
        ML.model.predict(full_path)

        # Show the result
        return redirect(url_for("experience_show", filename=fname))


@app.route("/example")
def example():
    clean_upload_folder()

    # Copy example data
    for f in os.listdir(app.config["DEMO_FOLDER"]):
        if f != ".gitignore":
            shutil.copyfile(
                root_path_join(app.config["DEMO_FOLDER"], f),
                root_path_join(app.config["UPLOAD_FOLDER"], f)
            )
    return redirect(url_for("experience_show", filename=DEMO_FNAME))


@app.route("/experience")
def experience():
    return render_template("pages/experience.html")

"""
@app.route("/barcode.png")
def barcode():
    file = root_path_join(app.config["UPLOAD_FOLDER"], DEMO_TRUTH)
    df_trim = extract_keys_from_file(file)
    data = extract_timeline_from_df(df_trim)

    fig = fig_generate_barcode(data)
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype="image/png")


@app.route("/tenpatterns.png")
def tenpatterns():
    file = root_path_join(app.config["UPLOAD_FOLDER"], DEMO_TRUTH)
    df_trim = extract_keys_from_file(file)
    all_keys = extract_pattern_from_df(df_trim)
    all_patterns = find_all_patterns(all_keys)

    fig = fig_10_patterns(all_patterns)
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype="image/png")
"""

@app.route("/markov.png")
def markov():
    file = os.path.join(app.config["UPLOAD_FOLDER"], DEMO_TRUTH)
    df_trim = extract_keys_from_file(file)
    data = extract_timeline_from_df(df_trim)

    fig = create_figure()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype="image/png")


@app.route("/experience-show")
def experience_show():
    if not request.args.get("filename"):
        return flash_and_redirect("experience", "Il faut d'abord téléverser une vidéo !")
        
    filename = os.path.join(app.config["UPLOAD_FOLDER"], request.args.get("filename"))
    if not os.path.isfile(filename):
        return flash_and_redirect("experience", "Le fichier sélectionné semble invalide")

    return render_template("pages/experience-show.html", filename=filename)

@app.route("/")
def home():
    return render_template("pages/home.html")

@app.route("/rapport")
def rapport():
    return render_template("pages/rapport.html")

@app.route("/resultats")
def resultats():
    return render_template("pages/resultats.html")

# Error handlers.

@app.errorhandler(500)
def internal_error(error):
    return render_template("errors/500.html"), 500

@app.errorhandler(404)
def not_found_error(error):
    return render_template("errors/404.html"), 404

if not app.debug:
    file_handler = FileHandler("error.log")
    file_handler.setFormatter(
        Formatter("%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]")
    )
    app.logger.setLevel(logging.INFO)
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.info("errors")

if __name__ == "__main__":
    if app.config["DEBUG"]:
        app.run()
    else:
        app.run(port=5000, host="0.0.0.0")
