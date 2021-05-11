
import numpy as np
from flask import Flask, request, jsonify, render_template

from textblob import TextBlob
import nltk
from newspaper import Article

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')
    
@app.route('/movie')
def movie():
    return render_template('movie.html')


@app.route('/predict',methods=['POST'])
def predict():
    input = request.form.get("review")
    
    
    object = TextBlob(input)

    sentiment = object.sentiment.polarity
    x = round(sentiment, 2)
    if sentiment ==0:
        predict = "Neutral with a score of {} out of 1".format(x)
    elif sentiment > 0:
        predict = "Positive with a score of {} out of 1".format(x)
    else:
        predict = "Negative with a score of {} out of 1".format(x)

    return render_template('movie.html', prediction_text = 'Your review was rated as {}'.format(predict))

if __name__ == "__main__":
        app.run(debug=True)



