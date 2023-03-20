import io
import logging
import os
import shutil
import threading
import ML.model

from logging import FileHandler, Formatter
from datascience import *
from flask import flash, Flask, redirect, render_template, request, Response, url_for, g
from flask_babel import Babel, gettext
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {"mkv", "mp4", "webm", "avi"}
DEMO_TRUTH = "truth.csv"
DEMO_FNAME = "video.webm"
YOUTUBE_PATTERN = r"^((?:https?:)?\/\/)?((?:www|m)\.)?((?:youtube(-nocookie)?\.com|youtu.be))(\/(?:[\w\-]+\?v=|embed\/|v\/)?)([\w\-]+)(\S+)?$"


def get_locale():
    print("GET ", request.cookies.get('lang'))
    if request.args.get('lang'):
        return request.args.get('lang')
    if request.cookies.get('lang'):
        return request.cookies.get('lang')
    else:
        return request.accept_languages.best_match(['fr', 'en'])


app = Flask(__name__)
babel = Babel(app, locale_selector=get_locale)
app.config["UPLOAD_FOLDER"] = os.path.join("static", "inputs")
app.config["DEMO_FOLDER"] = os.path.join("static", "inputs", "demo")
app.config["MAX_CONTENT_PATH"] = 200_000_000
app.config.from_object("config")

current_ML_thread = None
current_ML_progress = 0


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


def run_ML(path):
    ML.model.predict(path, callback=run_ML_callback)


def run_ML_callback(progress):
    global current_ML_progress
    print(f"Progress {str(round(progress * 100, 2))}%...")
    current_ML_progress = progress


@app.after_request
def after_request(response):
    if request.args.get('lang'):
        response.set_cookie('lang', request.args.get('lang'))
    return response


@app.route("/youtube", methods=["GET", "POST"])
def youtube_link():
    return "oui"


@app.route("/uploader", methods=["GET", "POST"])
def upload_file():
    global current_ML_thread
    if request.method == "POST":
        if "file" not in request.files:
            return flash_and_redirect("experience", gettext("No file to uploaded"))

        file = request.files["file"]
        if file.filename == "":
            return flash_and_redirect("experience", gettext("No file selected"))

        if not (file and allowed_file(file.filename)):
            return flash_and_redirect("experience", gettext("Please, use a video in a valid format"))

        if not current_ML_thread:
            clean_upload_folder()
            fname = secure_filename(file.filename)
            full_path = root_path_join(app.config["UPLOAD_FOLDER"], fname)

            # Save the new file
            file.save(full_path)
            current_ML_thread = threading.Thread(target=run_ML, args=(full_path,))
            current_ML_thread.start()
            return redirect(url_for("waiting", filename=fname))
        else:
            return redirect(url_for("overloaded"))


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


@app.route("/")
def home():
    return render_template("pages/home.html")


@app.route("/recording")
def recording():
    return render_template("pages/recording.html")


@app.route("/report")
def report():
    return render_template("pages/report.html")


@app.route("/results")
def results():
    return render_template("pages/results.html")


@app.route("/overloaded")
def overloaded():
    return render_template("pages/overloaded.html")


@app.route("/waiting")
def waiting():
    return render_template("pages/waiting.html")


@app.route("/progression")
def progression():
    global current_ML_progress
    return {"progression": current_ML_progress}


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
        return flash_and_redirect("experience", gettext("Please, upload a video first"))

    filename = os.path.join(app.config["UPLOAD_FOLDER"], request.args.get("filename"))
    if not os.path.isfile(filename):
        return flash_and_redirect("experience", gettext("The file seems to be invalid"))

    return render_template("pages/experience-show.html", filename=filename)


# Error handlers.

@app.errorhandler(500)
def internal_error(error):
    return render_template("errors/500.html"), 500


@app.errorhandler(404)
def not_found_error(error):
    return render_template("errors/404.html"), 404


if not app.debug:
    file_handler = FileHandler("error.log")
    file_handler.setFormatter(Formatter("%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]"))
    app.logger.setLevel(logging.INFO)
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.info("errors")

if __name__ == "__main__":
    if app.config["DEBUG"]:
        app.run()
    else:
        app.run(port=5000, host="0.0.0.0")
