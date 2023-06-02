import config as cfg
import json
import numpy as np
import pickle
import sklearn

from flask import Flask, request, jsonify

app = Flask(__name__)
app.__name__ = 'DSML Salary Predicion API' 
app.__version__ = '0.0.1' 

model = pickle.load(open(cfg.MODEL_DIR, "rb"))

@app.route("/", methods=['GET'])
def home():
    return '{} v{}'.format(app.__name__, app.__version__)

@app.route("/about", methods=['GET'])
def about():
    model_info = dict()
    model_info['name'] = app.__name__
    model_info['version'] = app.__version__
    model_info['type'] = str(model)
    return jsonify(model_info) 

@app.route("/predict", methods=['POST'])
def predict():
    input_data = request.get_data()
    input_data = input_data.decode('UTF-8')
    input_data = json.loads(input_data)
    prediction = model.predict(np.array([[input_data['yoe']]]))[0][0]
    return jsonify({"expected": prediction})

if __name__ == '__main__':
     app.run(debug=False)
