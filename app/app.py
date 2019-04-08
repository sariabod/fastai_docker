from flask import Flask, request, jsonify, abort, Response
from helper import *
import os
os.environ['TORCH_MODEL_ZOO'] = '/app/data/models'

application = Flask(__name__)

wave = get_segmentation_model('wave',[0,1])
signal = get_segmentation_model('signal',[0,1])

@application.route("/")
def index():

    html = """
    <h3>JSON Payload</h3>
    <p>Array of Arrays ordered by timestamp.</p>
    <pre>
    {
        "data": 
        [
            [
                TIMESTAMP,
                PRESSURE,
                PRESSURE DIFF
            ]
        ]
    }
    <pre>

    <h3>Example Request</h3>
    <pre>
    {
        "data": 
        [
            [
                "2/1/2019 4:52:56",
                18.67,
                512.81
            ],
            [
                "2/1/2019 4:52:57",
                47.41,
                528.74
            ],
            [
               "2/1/2019 4:52:58",
               88.72,
               541.31
            ]
        ]
    }
    </pre>
    <h3>Example Response</h3>
    <p>Signals and Waves, row id, time in seconds, start of downtime, end of downtime.</p>
    <pre>
    {
        "signals": 
        [
            [
                0,
                254,
                "2/1/2019 4:57:26",
                "2/1/2019 5:01:39"
            ],
            [
                1,
                194,
                "2/1/2019 5:03:45",
                "2/1/2019 5:06:58"
            ]
        ],
        "waves": 
        [
            [    
                0,
                148,
                "2/1/2019 4:57:33",
                "2/1/2019 5:00:00"
            ],
            [
                1,
                73,
                "2/1/2019 5:00:08",
                "2/1/2019 5:01:20"
            ],
       ]
    }
    </pre>
    

    """

    return html


@application.route("/predict", methods=['POST'])
def predict():
    inputs = request.get_json()

    try:
        data = inputs['data']
    except KeyError as e:
        return Response('Malformed JSON : Missing data Key', 400)

    try:
        preds = inputs['preds']
    except KeyError as e:
        preds = 0

    if len(data) > 3600:
        return Response('JSON Size Error : Too many records, MAX 3600', 400)

    try:
        timestamp = np.array(list(zip(*data))[0])
        psi = list(zip(*data))[1]
        diff = list(zip(*data))[2]
    except Exception as e:
        return Response('JSON Array Error : Mismatched Columns', 400)

    try:
        sig_input = build_input(diff, 1000)
        sig_pred = signal.predict(torch.tensor(sig_input, dtype=torch.float)/255)
        sig_group = pred_details(sig_pred, timestamp, 50, 29)
        wave_input = build_input(psi, 5000)
        wave_pred = wave.predict(torch.tensor(wave_input, dtype=torch.float)/255)
        wave_group = pred_details(wave_pred, timestamp, 50, 29)
    except Exception as e:
        return Response(e, 400)


    if preds:
        final_output = {"waves":wave_group, "signals":sig_group, "wave_pred":wave_pred[1].tolist(), "wave_input":wave_input.tolist(), "signal_pred":sig_pred[1].tolist(), "signal_input":sig_input.tolist()}
    else:
        final_output = {"waves":wave_group, "signals":sig_group}


    return jsonify(final_output)


if __name__ == "__main__":
    application.run(host='0.0.0.0', port=8000, debug=True)
