import pickle
import pandas as pd
from flask import Flask, request

with open("knn_heart_disease.pkl", 'rb') as file:
	classifier = pickle.load(file)

app = Flask(__name__)

@app.route("/KNN-heart-disease", methods=["POST"])
def knn_heart_disease():
	data = request.get_json(force=True)
	df = pd.DataFrame(data, index=[0])
	prediction = classifier.predict(df)

	if prediction[0] == 0:
		return "No presence of heart disease"
	else:
		return "Presence detected"

if __name__== "__main__":
	app.run(port=9000)