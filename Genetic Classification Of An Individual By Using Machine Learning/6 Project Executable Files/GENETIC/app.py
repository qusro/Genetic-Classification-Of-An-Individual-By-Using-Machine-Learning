from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load your machine learning model
model = pickle.load(open(r"C:\Users\Linga murthy\Desktop\GENETIC\Genetic-classification.pkl", 'rb'))

# Route for index page
@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

# Route for About page
@app.route('/about')
def about():
    return render_template('index.html')

# Route for Prediction page (GET method)
@app.route('/Prediction', methods=["GET"])
def prediction_form():
    return render_template('predict.html')

# Route for Prediction form submission (POST method)
@app.route('/Prediction', methods=["POST"])
def make_prediction():
    # Retrieve data from the form
    x = [int(request.form.get('X1')),
         int(request.form.get('X2')),
         int(request.form.get('X3')),
         int(request.form.get('X4')),
         int(request.form.get('X5'))]
    
    x = np.array(x).reshape(1, -1)  # Reshape the input to 2D array with one row

    # Inspect the number of features the model expects
    print(f"Number of features the model expects: {model.n_features_in_}")
    print(f"Input data shape: {x.shape}")

    # Make prediction using your model
    prediction = model.predict(x)
    
    # Determine prediction result
    if prediction == 0:
        predict = "Clinical variant Classification"
    else:
        predict = "Conflicting variant classification"
    
    return render_template("result.html", predict=predict)

if __name__ == "__main__":
    app.run(debug=True, port=1234)
