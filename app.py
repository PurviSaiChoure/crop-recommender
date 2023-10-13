import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
model = None
scaler = None

try:
    model = joblib.load('final_model.joblib')
    scaler = joblib.load('final_scaler.joblib')
except Exception as e:
    print("Error loading model or scaler:", str(e))

@app.route('/', methods=['GET'])
def home():
    return render_template('index2.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == "POST":
        try:
            features = [float(x) for x in request.form.values()]
            input_data = np.array(features).reshape(1, -1)
            if scaler is not None:
                scaled_features = scaler.transform(input_data)
                prediction = model.predict(scaled_features)
                output = prediction[0]
                return render_template('result2.html', prediction_text=output)
            else:
                return render_template('error.html', error_message='Scaler not loaded.')
        except Exception as e:
            return render_template('error.html', error_message=str(e))
    else:
        return render_template('index2.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = request.get_json(force=True)
        if model is not None and scaler is not None:
            prediction = model.predict([np.array(list(data.values()))])
            output = prediction[0]
            return jsonify({'prediction': output})
        else:
            return jsonify({'error': 'Model or scaler not loaded.'})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
