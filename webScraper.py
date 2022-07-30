from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import cross_validate
import pickle


PATH = "C:\Program Files (x86)\chromedriver.exe"
ops = Options();
ops.add_argument("--headless")
driver = webdriver.Chrome(executable_path = PATH, options = ops)

driver.get("https://b2gdevs.github.io/MLIntro/heart-disease.html")

age_elements = driver.find_elements(By.CLASS_NAME, "patient-age")
sex_elements = driver.find_elements(By.CLASS_NAME, "patient-sex")
cp_elements = driver.find_elements(By.CLASS_NAME, "patient-cp")
trestbps_elements = driver.find_elements(By.CLASS_NAME, "patient-trestbps")
chol_elements = driver.find_elements(By.CLASS_NAME, "patient-chol")
fbs_elements = driver.find_elements(By.CLASS_NAME, "patient-fbs")
restecg_elements = driver.find_elements(By.CLASS_NAME, "patient-restecg")
thalach_elements = driver.find_elements(By.CLASS_NAME, "patient-thalach")
exang_elements = driver.find_elements(By.CLASS_NAME, "patient-exang")
oldpeak_elements = driver.find_elements(By.CLASS_NAME, "patient-oldpeak")
slope_elements = driver.find_elements(By.CLASS_NAME, "patient-slope")
ca_elements = driver.find_elements(By.CLASS_NAME, "patient-ca")
thal_elements = driver.find_elements(By.CLASS_NAME, "patient-thal")
target_elements = driver.find_elements(By.CLASS_NAME, "patient-target")


table = []

for i in range(len(age_elements)-1):
	row =[]
	row.append(int(age_elements[i].text))
	row.append(int(sex_elements[i].text))
	row.append(int(cp_elements[i].text))
	row.append(int(trestbps_elements[i].text))
	row.append(int(chol_elements[i].text))
	row.append(int(fbs_elements[i].text))
	row.append(int(restecg_elements[i].text))
	row.append(int(thalach_elements[i].text))
	row.append(int(exang_elements[i].text))
	row.append(float(oldpeak_elements[i].text))
	row.append(int(slope_elements[i].text))
	row.append(int(ca_elements[i].text))
	row.append(int(thal_elements[i].text))
	row.append(int(target_elements[i].text))

	table.append(row)

table_headers = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","target"]
df = pd.DataFrame(table, columns=table_headers)

y = df["target"]
df = df.drop(["target"], axis=1)
x = df

classifier = KNN(n_neighbors=3)
classifier.fit(x, y)

with open("knn_heart_disease.pkl", 'wb') as file:
	pickle.dump(classifier, file)



driver.quit()