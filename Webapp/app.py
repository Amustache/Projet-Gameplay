from logging import FileHandler, Formatter
import io
import logging


from datascience import *
from flask import Flask, render_template, request, Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


app = Flask(__name__)
app.config.from_object("config")


# Results
@app.route("/barcode.png")
def barcode_png():
    data = get_data()
    fig = fig_generate_barcode(data)

    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype="image/png")


@app.route("/patterns.png")
def patterns_png():
    df_trim = get_df_trim()
    all_keys = extract_pattern_from_df(df_trim)
    all_patterns = find_all_patterns(all_keys)
    fig = fig_10_patterns(all_patterns)

    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype="image/png")


@app.route("/")
def home():
    return render_template("pages/home.html")


@app.route("/rapport")
def rapport():
    return render_template("pages/rapport.html")


@app.route("/resultats")
def resultats():
    return render_template("pages/resultats.html")


@app.route("/experience")
def experience():
    return render_template("pages/experience.html")


@app.route("/experience-show")
def experience_show():
    return render_template("pages/experience-show.html")


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
