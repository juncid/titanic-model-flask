import pandas as pd
# import requests
from flask import Flask, jsonify, request
import pickle

#load model
model = pickle.load(open('model_titanic.pkl', 'rb'))

#app
app = Flask(__name__)

#routes
@app.route('/', methods=['POST'])
def predict():
    # get data
    data = request.get_json(force=True)

    # convert data in dataframe
    data.update((x, [y]) for x, y in data.items())
    data_df = pd.DataFrame.from_dict(data)

    # predictions
    result = model.predict(data_df)

    if int(result[0]) == 0:
        status_passenger = 'Not survived'
    elif int(result[0] == 1):
        status_passenger = 'Survived'

    # send back to browser
    output = {'results': status_passenger}

    # return data
    return jsonify(output)


if __name__ == '__main__':
    app.run(port=5000, debug=True)
