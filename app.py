import pandas as pd
from flask import Flask, request, jsonify
import numpy as np
import pickle

model = pickle.load(open('iris_model.pkl', 'rb'))
app = Flask(__name__)


@app.route('/')
def index():
    return "Hello world"


# @app.route('/predict', methods=['POST'])
# def predict():
#     s_l = request.form.get('sl')
#     s_w = request.form.get('sw')
#     p_l = request.form.get('pl')
#     p_w = request.form.get('pw')
#     input_arr = np.array([[s_l, s_w, p_l, p_w]]).astype('float32')
#     # input_arr = np.array([[1, 1, 1, 1]]).astype('float32')
#
#     classes = ['Iris Setosa', 'Iris Versicolor', 'Iris Virginica']
#     result = classes[model.predict(input_arr).argmax()]
#     # print(result)
#
#     # content = request.json         #parsing Json directly
#     # df = pd.DataFrame(content)
#     # df = df.expand_dims().cast('float32')
#     # result = classes[model.predict(df).argmax()]
#     # print(content['sl'])
#
#     # s_l = content['sl']
#     # s_w = content['sw']
#     # p_l = content['pl']
#     # p_w = content['pw']
#     # input_arr = np.array([[s_l, s_w, p_l, p_w]]).astype('float32')
#     # input_arr = np.array([[1, 1, 1, 1]]).astype('float32')    #testing
#     # result = classes[model.predict(input_arr).argmax()]
#     return jsonify({'Flower type': result})


if __name__ == '__main__':
    app.run(debug=True)
