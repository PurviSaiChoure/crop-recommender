import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
model = None
scaler = None


try:
    model = joblib.load('model.joblib')
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
            prediction = model.predict(input_data)
            output = prediction[0]
            return render_template('result2.html', prediction_text=output)
        except Exception as e:
            return render_template('error.html', error_message=str(e))
    else:
        return render_template('index2.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = request.get_json(force=True)
        input_data = np.array(list(data.values())).reshape(1, -1)
        prediction = model.predict(input_data)
        output = prediction[0]
        return jsonify({'prediction': output})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
