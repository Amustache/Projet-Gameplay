# ----------------------------------------------------------------------------#
# Imports
# ----------------------------------------------------------------------------#

from logging import FileHandler, Formatter
import io

# from flask.ext.sqlalchemy import SQLAlchemy
import logging
import os
import random


from datascience import *
from flask import Flask, render_template, request, Response
from forms import *
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


# ----------------------------------------------------------------------------#
# App Config.
# ----------------------------------------------------------------------------#

app = Flask(__name__)
app.config.from_object("config")
# db = SQLAlchemy(app)

# Automatically tear down SQLAlchemy.
"""
@app.teardown_request
def shutdown_session(exception=None):
    db_session.remove()
"""

# Login required decorator.
"""
def login_required(test):
    @wraps(test)
    def wrap(*args, **kwargs):
        if 'logged_in' in session:
            return test(*args, **kwargs)
        else:
            flash('You need to login first.')
            return redirect(url_for('login'))
    return wrap
"""


# ----------------------------------------------------------------------------#
# Controllers.
# ----------------------------------------------------------------------------#
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


@app.route("/login")
def login():
    form = LoginForm(request.form)
    return render_template("forms/login.html", form=form)


@app.route("/register")
def register():
    form = RegisterForm(request.form)
    return render_template("forms/register.html", form=form)


@app.route("/forgot")
def forgot():
    form = ForgotForm(request.form)
    return render_template("forms/forgot.html", form=form)


# Error handlers.


@app.errorhandler(500)
def internal_error(error):
    # db_session.rollback()
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

# ----------------------------------------------------------------------------#
# Launch.
# ----------------------------------------------------------------------------#

# Default port:
if __name__ == "__main__":
    app.run()

# Or specify port manually:
"""
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
"""
