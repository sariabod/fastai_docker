from flask import Flask, request, jsonify
import fastai
from fastai.vision import *

application = Flask(__name__)

def get_label(x):
    return Path('data/label/0.png')

src = (SegmentationItemList.from_folder(path='data', convert_mode='L')
       .split_none()
       .label_from_func(get_label, classes=[0,1])
      )

data = src.databunch(no_check=True).normalize(imagenet_stats)
learn = unet_learner(data, models.resnet34).to_fp16()
learn.load('model')


@application.route("/")
def index():

    fai = fastai.__version__
    html = "<h3>FASTAI LOADED:  {fai}</h3>"
    return html.format(fai=fai)


@application.route("/predict", methods=['POST'])
def predict():
    inputs = request.get_json()





















    return jsonify(learn.model)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8888)
