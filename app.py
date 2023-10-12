import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
model = joblib.load(open(r'C:\Users\purvi\Downloads\Crop-Recommender\app\final_model.joblib', 'rb'))
sc = joblib.load(open(r'C:\Users\purvi\Downloads\Crop-Recommender\app\final_scaler.joblib','rb'))

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == "POST":
        features = [float(x) for x in request.form.values()]  
        input_data = np.array(features).reshape(1, -1)  
        scaled_features = sc.transform(input_data)
        prediction = model.predict(scaled_features)  
        output = prediction[0]  

        return render_template('result.html', prediction_text='The Recommended Crop to grow is {}'.format(output))

    else:
        return render_template('index.html')

    
if __name__ == "__main__":
    app.run(debug=True)

 
