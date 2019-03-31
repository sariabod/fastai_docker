from flask import Flask, request, jsonify
from helper import *

application = Flask(__name__)

wave = get_segmentation_model('wave',[0,1])
signal = get_segmentation_model('signal',[0,1])


@application.route("/")
def index():

    html = "<h3>These are not the droids you are looking for....</h3>"
    return html.format(fai=fai)


@application.route("/predict", methods=['POST'])
def predict():
    inputs = request.get_json()
    


    img = torch.tensor(np.ones([1,100,3600]), dtype=torch.float)
    pred = wave.predict(img)
    pred = signal.predict(img)





    return jsonify(['hey'])


if __name__ == "__main__":
    application.run(host='0.0.0.0', port=8000)
