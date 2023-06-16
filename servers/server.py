import config as cfg
import json
import numpy as np
import pickle
import sklearn
import pandas as pd

from flask import Flask, request, jsonify

app = Flask(__name__)
app.__name__ = 'DSML Salary Predicion API' 
app.__version__ = '0.0.1' 

# Load the models
model_exp = pickle.load(open(cfg.MODEL_EXP_DIR, "rb"))
model_q25 = pickle.load(open(cfg.MODEL_Q25_DIR, "rb"))
model_q75 = pickle.load(open(cfg.MODEL_Q75_DIR, "rb"))

# Load the COLI lookup
df_coli2022 = pd.read_csv('./data/cost-of-living-index-2022.csv')
df_coli2022.drop(
    [
        'Rank',
        'Rent Index',
        'Cost of Living Plus Rent Index',
        'Groceries Index',
        'Restaurant Price Index',
        'Local Purchasing Power Index'
    ],
    axis=1,
    inplace=True
)
df_coli2022.rename(
    columns={
        "Country": "location",
        "Cost of Living Index": "coli"
    },
    inplace=True
);
df_colibs2022 = pd.read_csv('./data/cost-of-living-index-by-state-2022.csv')
df_colibs2022.drop(
    [
        'groceryCost',
        'housingCost',
        'utilitiesCost',
        'transportationCost',
        'miscCost'
    ],
    axis=1,
    inplace=True
);
df_colibs2022.rename(
    columns={
        "state": "location",
        "costIndex": "coli"
    },
    inplace=True
);
df_colibs2022['coli'] = df_colibs2022['coli'] - df_colibs2022['coli'].mean() + df_coli2022.loc[131, 'coli']
df_coli2022 = df_coli2022._append(df_colibs2022, ignore_index=True)


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
    coli = df_coli2022[df_coli2022['location'] == input_data['location']]['coli'].values[0]
    input_ = np.array(
            [
                [input_data['yoe'],
                 input_data['yac'],
                 int(input_data['female']),
                 coli]
                ]
            )
    prediction_exp = np.e**model_exp.predict(input_)[0]
    prediction_q25 = np.e**model_q25.predict(input_)[0]
    prediction_q75 = np.e**model_q75.predict(input_)[0]
    return jsonify(
            {"expected": prediction_exp,
             "quantile_low": prediction_q25,
             "quantile_high": prediction_q75})

if __name__ == '__main__':
     app.run(debug=False)
