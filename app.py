from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load the trained model and scaler
with open('heartpredictor.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    data = list(map(float, data.values()))
    data = np.array(data).reshape(1, -1)
    
    prediction = model.predict(data)
    
    result = 'Survived' if prediction[0] == 0 else 'Not Survived'
    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
