import io
import logging
import os
import re
import shutil
import threading
import random
import string

from logging import FileHandler, Formatter
from flask import flash, Flask, g, redirect, render_template, request, Response, url_for
from flask_babel import Babel, gettext
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from pytube import YouTube
import ML.model

ALLOWED_EXTENSIONS = {"mkv", "mp4", "webm", "avi"}
DEMO_TRUTH = "truth.csv"
DEMO_FNAME = "video.webm"
YOUTUBE_PATTERN = re.compile(
    r"^((?:https?:)?\/\/)?((?:www|m)\.)?((?:youtube(-nocookie)?\.com|youtu.be))(\/(?:[\w\-]+\?v=|embed\/|v\/)?)([\w\-]+)(\S+)?$"
)
ML_THREADS = {}
MAX_ML_THREADS = 10
FILENAME_LENGTH = 7
current_YT_progress = 0


def get_locale():
    print("GET ", request.cookies.get("lang"))
    if request.args.get("lang"):
        return request.args.get("lang")
    if request.cookies.get("lang"):
        return request.cookies.get("lang")
    else:
        return request.accept_languages.best_match(["fr", "en"])


def root_path_join(*args):
    return os.path.join(app.root_path, *args)


def clean_upload_folder():
    path = root_path_join(app.config["UPLOAD_FOLDER"])
    for f in os.listdir(path):
        full_path = os.path.join(path, f)
        if os.path.isfile(full_path) and f != ".gitkeep":
            os.remove(full_path)


def flash_and_redirect(dest, msg):
    flash(msg)
    return redirect(url_for(dest))


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def run_ML(path, thread_name):
    global ML_THREADS
    ML.model.predict(path, callback=run_ML_callback, name=thread_name)
    del ML_THREADS[thread_name]['thread']


def run_ML_callback(progress, name):
    global ML_THREADS
    print(f"Progress of {name} " + str(round(progress * 100, 2)) + "% ...")
    ML_THREADS[name]['progress'] = progress


def progress_function(stream, chunk, bytes_remaining):
    global current_YT_progress
    current_YT_progress = 1 - float(bytes_remaining) / float(stream.filesize)
    print(f"YouTube processing: {current_YT_progress}")


def get_video_random_name():
    return ''.join(random.choice(string.ascii_letters) for i in range(FILENAME_LENGTH))


app = Flask(__name__)
babel = Babel(app, locale_selector=get_locale)
app.config["UPLOAD_FOLDER"] = os.path.join("static", "inputs")
app.config["DEMO_FOLDER"] = os.path.join("static", "inputs", "demo")
app.config["MAX_CONTENT_PATH"] = 200_000_000
app.config.from_object("config")


@app.after_request
def after_request(response):
    if request.args.get("lang"):
        response.set_cookie("lang", request.args.get("lang"))
    return response


@app.route("/youtube", methods=["GET", "POST"])
def youtube_link():
    global ML_THREADS
    if request.method == "POST":
        form_data = request.form
        if "url" in form_data:
            yt_link = form_data["url"]
            if not YOUTUBE_PATTERN.match(yt_link):
                return flash_and_redirect("experience", gettext("Please use a valid YouTube link."))

            if len(ML_THREADS) <= MAX_ML_THREADS:
                # clean_upload_folder()

                fname = get_video_random_name()
                path = root_path_join(app.config["UPLOAD_FOLDER"])
                full_path = os.path.join(path, fname)
                print(f"Processing: {yt_link}")
                YouTube(
                    yt_link, on_progress_callback=progress_function, on_complete_callback=None
                ).streams.filter(progressive=True, file_extension="mp4").order_by(
                    "resolution"
                ).desc().first().download(
                    output_path=path, filename=fname
                )

                ML_THREADS[fname] = {'thread': None, 'progress': 0}
                ML_THREADS[fname]['thread'] = threading.Thread(target=run_ML, args=(full_path, fname))
                ML_THREADS[fname]['thread'].start()

                return redirect(url_for("experience_waiting", filename=fname))
            else:
                return redirect(url_for("overloaded"))


@app.route("/youtube_download", methods=["GET", "POST"])
def youtube_progress():
    global current_YT_progress
    return {"progression": current_YT_progress}


@app.route("/uploader", methods=["GET", "POST"])
def upload_file():
    global ML_THREADS
    if request.method == "POST":
        if "file" not in request.files:
            return flash_and_redirect("experience", gettext("No file uploaded."))

        file = request.files["file"]
        if file.filename == "":
            return flash_and_redirect("experience", gettext("No file selected."))

        if not (file and allowed_file(file.filename)):
            return flash_and_redirect(
                "experience", gettext("Please, use a video in a valid format.")
            )

        if "truthFile" in request.files:
            truth = request.files["truthFile"]
            if truth.filename != "" and not (truth and truth.filename.endswith(".csv")):
                return flash_and_redirect("experience", gettext("Please, use a truth file in a csv format."))
        else:
            truth = None

        if len(ML_THREADS) <= MAX_ML_THREADS:
            # clean_upload_folder()
            fname = get_video_random_name()
            path = root_path_join(app.config["UPLOAD_FOLDER"])
            full_path = os.path.join(path, fname)

            # Save the new file
            file.save(full_path)

            if truth:
                truth.save(os.path.join(path, "truth.csv"))

            ML_THREADS[fname] = {'thread': None, 'progress': 0}
            ML_THREADS[fname]['thread'] = threading.Thread(target=run_ML, args=(full_path, fname))
            ML_THREADS[fname]['thread'].start()

            return redirect(url_for("experience_waiting", filename=fname))
        else:
            return redirect(url_for("overloaded"))


@app.route("/example")
def example():
    # clean_upload_folder()

    # Copy example data
    for f in os.listdir(app.config["DEMO_FOLDER"]):
        if f != ".gitignore":
            shutil.copyfile(
                root_path_join(app.config["DEMO_FOLDER"], f),
                root_path_join(app.config["UPLOAD_FOLDER"], f),
            )
    return redirect(url_for("experience_show", filename=DEMO_FNAME))


@app.route("/progression")
def progression():
    global ML_THREADS
    if request.args.get('filename') != None:
        print("VALUE : ", ML_THREADS[request.args.get('filename')]['progress'])
        return {"progression": ML_THREADS[request.args.get('filename')]['progress']}
    else:
        return {"progression": "Invalid file"}


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


@app.route("/overloaded")
def overloaded():
    return render_template("pages/overloaded.html")


@app.route("/experience-waiting")
def experience_waiting():
    return render_template("pages/waiting.html")


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
