from flask import Flask
import fastai
from fastai.vision import *

application = Flask(__name__)

@application.route("/")
def index():

    fai = fastai.__version__

    html = "<h3>FASTAI LOADED:  {fai}</h3>"

    return html.format(fai=fai)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8888)
