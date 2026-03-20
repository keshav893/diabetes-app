from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [
            float(request.form['preg']),
            float(request.form['glucose']),
            float(request.form['bp']),
            float(request.form['skin']),
            float(request.form['insulin']),
            float(request.form['bmi']),
            float(request.form['dpf']),
            float(request.form['age']),
            float(request.form['hypertension']),
            float(request.form['heart']),
            float(request.form['smoking'])
        ]

        input_data = np.array([data])

        prediction = model.predict(input_data)[0]

        result = "Diabetes ⚠️" if prediction == 1 else "No Diabetes ✅"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)