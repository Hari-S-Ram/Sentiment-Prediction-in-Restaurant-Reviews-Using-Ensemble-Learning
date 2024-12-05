from flask import Flask, render_template, request, jsonify, send_from_directory
import joblib
import os

app = Flask(__name__)

model = joblib.load('model.pkl')

@app.route('/background.jpg')
def send_background_image():
    return send_from_directory(os.getcwd(), 'background.jpg')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    review = request.form['review']

    if "good" in review.lower() and review.lower().count("good") > 1:
        sentiment = "Liked" 
    else:
        prediction = model.predict([review])[0]  
        sentiment = "Liked" if prediction == 1 else "Not Liked" 

    return render_template('index.html', prediction_text=f'Sentiment: {sentiment}')

if __name__ == '__main__':
    app.run(debug=True)
