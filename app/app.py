from flask import Flask, request, jsonify
from helper import *

application = Flask(__name__)

wave = get_segmentation_model('wave',[0,1])
signal = get_segmentation_model('signal',[0,1])


@application.route("/")
def index():

    html = "<h3>These are not the droids you are looking for....</h3>"
    return html.format()


@application.route("/predict", methods=['POST'])
def predict():
    inputs = request.get_json()
    data = inputs['data']
    timestamp = np.array(list(zip(*data))[0])
    psi = list(zip(*data))[1]
    diff = list(zip(*data))[2]

    pred_signal = signal.predict(build_input(diff, 1000))
    pred_wave = wave.predict(build_input(psi, 6000))
    wave_group = pred_details(pred_wave, timestamp)
    signal_group = pred_details(pred_signal, timestamp)
    final_output = {"waves":wave_group, "signals":signal_group}

    return jsonify(final_output)


if __name__ == "__main__":
    application.run(host='0.0.0.0', port=8000)
