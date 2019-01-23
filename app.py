from flask import Flask,render_template,url_for,request
from flask_material import Material

# EDA PKg
import pandas as pd 
import numpy as np 

# ML Pkg
from sklearn.externals import joblib


app = Flask(__name__)
Material(app)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/preview')
def preview():
    df = pd.read_csv("data/Placed.csv")
    return render_template("preview.html",df_view = df)
	 
@app.route('/name')
def name():
    JsonData = open('data/data_avg1.json','r+').read()
    return render_template('content.html', text=JsonData)

@app.route('/name',methods=["POST"])
def name1():
	JsonData = open('data/data_avg1.json','r+').read()
	if request.method == 'POST':
		Quatitative  = request.form['quants']
		Communication = request.form['comms']
		Logical_Reasoning = request.form['logical']
		# sepal_width = request.form['sepal_width']
		model_choice = request.form['model_choice']

		# Clean the data by convert from unicode to float 
		sample_data = [Quatitative,Communication,Logical_Reasoning]
		clean_data = [int(i) for i in sample_data]
		data_c=data[0]

		# Reshape the Data as a Sample not Individual Features
		ex1 = np.array(clean_data).reshape(1,-1)
		logit_model = joblib.load('data/data_avg.pkl')

		# ex1 = np.array([6.2,3.4,5.4,2.3]).reshape(1,-1)

		# Reloading the Model
		if model_choice == 'logitmodel':
		    logit_model = joblib.load('data/logit_model_placed.pkl')
		    result_prediction = logit_model.predict(ex1)
		elif model_choice == 'knnmodel':
			knn_model = joblib.load('data/knn_model_placed.pkl')
			result_prediction = knn_model.predict(ex1)
		elif model_choice == 'svmmodel':
			knn_model = joblib.load('data/svm_model_iris.pkl')
			result_prediction = knn_model.predict(ex1)

	return render_template('content.html',
		text=JsonData,
		Quantative=Quatitative,
		comms=Communication,
		logical=Logical_Reasoning,
		clean_data=clean_data,
		result_prediction=result_prediction,
		model_selected=model_choice,
		)


@app.route('/',methods=["POST"])
def analyze():
	if request.method == 'POST':
		Quatitative  = request.form['Quantative']
		Communication = request.form['comms']
		Logical_Reasoning = request.form['logical']
		# sepal_width = request.form['sepal_width']
		model_choice = request.form['model_choice']

		# Clean the data by convert from unicode to float 
		sample_data = [Quatitative,Communication,Logical_Reasoning]
		clean_data = [int(i) for i in sample_data]

		# Reshape the Data as a Sample not Individual Features
		ex1 = np.array(clean_data).reshape(1,-1)

		# ex1 = np.array([6.2,3.4,5.4,2.3]).reshape(1,-1)

		# Reloading the Model
		if model_choice == 'logitmodel':
		    logit_model = joblib.load('data/logit_model_placed.pkl')
		    result_prediction = logit_model.predict(ex1)
		elif model_choice == 'knnmodel':
			knn_model = joblib.load('data/knn_model_placed.pkl')
			result_prediction = knn_model.predict(ex1)
		elif model_choice == 'svmmodel':
			knn_model = joblib.load('data/svm_model_iris.pkl')
			result_prediction = knn_model.predict(ex1)

	return render_template('index.html',
		Quantative=Quatitative,
		comms=Communication,
		logical=Logical_Reasoning,
		clean_data=clean_data,
		result_prediction=result_prediction,
		model_selected=model_choice,
		)



if __name__ == '__main__':
	app.run(debug=True)