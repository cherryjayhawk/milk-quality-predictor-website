from flask import Flask, render_template, request, jsonify
# from flask_cors import CORS
import joblib
# import pandas as pd
import numpy as np

app = Flask(__name__)
# CORS(app)



@app.route("/")
def home():
    return render_template('index.html')


@app.route('/prediksi', methods=["POST"])
def prediksi():

    algoritma = int(request.form['algoritma'])

    file = ""

    if (algoritma == 1):
        file = "knn_model.jlb"
    elif (algoritma == 2):
        file = "svm_model.jlb"
    else :
        file = "dt_model.jlb"

    model = joblib.load(open(file, "rb"))

    data1 = float(request.form['keasaman'])
    data2 = float(request.form['temperatur']) 
    data3 = float(request.form['rasa'])
    data4 = float(request.form['bau'])
    data5 = float(request.form['lemak'])
    data6 = float(request.form['kekentalan'])
    data7 = float(request.form['warna'])
    
    arr = np.array([[data1, data2, data3, data4, data5, data6, data7]])
    pred = model.predict(arr)
        
    return render_template('index.html', prediction = "{}".format(pred[0]))
    

if __name__ == "__main__":
    app.run(debug=True)
