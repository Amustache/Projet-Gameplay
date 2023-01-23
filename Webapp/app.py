from logging import FileHandler, Formatter
import io
import logging
import os
import shutil


from datascience import *
from flask import flash, Flask, redirect, render_template, request, Response, url_for
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from werkzeug.utils import secure_filename


ALLOWED_EXTENSIONS = {"mkv", "mp4", "webm", "avi"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join("static", "inputs")
app.config["MAX_CONTENT_PATH"] = 200_000_000
app.config.from_object("config")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/uploader", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            flash("Pas de fichier téléversé")
            return redirect(url_for("experience"))
        file = request.files["file"]
        if file.filename == "":
            flash("Pas de fichier sélectionné")
            return redirect(url_for("experience"))
        if file and allowed_file(file.filename):
            # Delete all files in the directory
            for f in os.listdir(os.path.join(app.root_path, app.config["UPLOAD_FOLDER"])):
                if f != ".gitkeep":
                    os.remove(os.path.join(app.root_path, app.config["UPLOAD_FOLDER"], f))

            # Save the new file
            fname = secure_filename(file.filename)
            file.save(os.path.join(app.root_path, app.config["UPLOAD_FOLDER"], fname))

            # Process the video and extract the data
            print("TODO")  # TODO

            # Show the result
            return redirect(url_for("experience_show", filename=fname))
        flash("Merci d'utiliser une vidéo valide")
        return redirect(url_for("experience"))


@app.route("/example")
def example():
    # Delete all files in the directory
    for f in os.listdir(os.path.join(app.root_path, app.config["UPLOAD_FOLDER"])):
        if f != ".gitkeep":
            os.remove(os.path.join(app.root_path, app.config["UPLOAD_FOLDER"], f))

    # Copy example data
    for f in os.listdir(os.path.join(app.root_path, "demo_inputs")):
        shutil.copyfile(
            os.path.join(app.root_path, "demo_inputs", f),
            os.path.join(app.root_path, app.config["UPLOAD_FOLDER"], f),
        )

    fname = "demo_video.webm"

    return redirect(url_for("experience_show", filename=fname))


@app.route("/experience")
def experience():
    return render_template("pages/experience.html")


@app.route("/barcode.png")
def barcode():
    file = os.path.join(app.root_path, app.config["UPLOAD_FOLDER"], "keylog.csv")
    df_trim = extract_keys_from_file(file)
    data = extract_timeline_from_df(df_trim)

    fig = fig_generate_barcode(data)
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype="image/png")


@app.route("/tenpatterns.png")
def tenpatterns():
    file = os.path.join(app.root_path, app.config["UPLOAD_FOLDER"], "keylog.csv")
    df_trim = extract_keys_from_file(file)
    all_keys = extract_pattern_from_df(df_trim)
    all_patterns = find_all_patterns(all_keys)

    fig = fig_10_patterns(all_patterns)
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype="image/png")


@app.route("/markov.png")
def markov():
    file = os.path.join(app.root_path, app.config["UPLOAD_FOLDER"], "keylog.csv")
    df_trim = extract_keys_from_file(file)
    data = extract_timeline_from_df(df_trim)

    fig = create_figure()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype="image/png")


@app.route("/experience-show")
def experience_show():
    if not request.args.get("filename"):
        flash("Il faut d'abord téléverser une vidéo !")
        return redirect(url_for("experience"))
    filename = os.path.join(app.config["UPLOAD_FOLDER"], request.args.get("filename"))
    if not os.path.isfile(os.path.join(app.root_path, filename)):
        flash("Le ficiher sélectionné semble invalide")
        return redirect(url_for("experience"))

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
    app.run()
