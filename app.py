from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')

# Load the data
df = pd.read_csv(r'C:\Users\Disyawns\Downloads\reviews_data_dump\reviews_badminton\data.csv')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Define a function to clean the text
def clean_text(text):
    # Convert the text to lowercase
    text = text.lower()
    
    # Remove stop words
    words = [word for word in text.split() if word not in stopwords.words('english')]
    
    # Lemmatize the words
    words = [lemmatizer.lemmatize(word) for word in words]
    
    # Join the words back into a string
    cleaned_text = ' '.join(words)
    
    return cleaned_text

# Apply the function to the 'Review text' column, handling NaN values
df['cleaned_text'] = df['Review text'].apply(lambda x: clean_text(str(x)) if not pd.isnull(x) else "")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['cleaned_text'], df['Ratings'], test_size=0.2, random_state=42)

# Initialize the TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit the TF-IDF vectorizer on training data
X_train_transformed = vectorizer.fit_transform(X_train)

# Define mapping dictionary
rating_mapping = {1: 'negative', 2: 'negative', 3: 'normal', 4: 'positive', 5: 'positive'}

# Define the home page route
@app.route('/')
def home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Sentiment Analysis</title>
    </head>
    <body>
        <h1>Sentiment Analysis</h1>
        <form action="/predict" method="post">
            <label for="text">Enter your text:</label><br>
            <textarea id="text" name="text" rows="4" cols="50"></textarea><br>
            <input type="submit" value="Submit">
        </form>
    </body>
    </html>
    """

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the text from the form
    text = request.form['text']
    
    # Preprocess the text
    cleaned_text = clean_text(text)
    
    # Vectorize the text
    text_vectorized = vectorizer.transform([cleaned_text])
    
    # Make prediction
    prediction = model.predict(text_vectorized)[0]
    
    # Map numerical prediction to label
    sentiment_label = rating_mapping[prediction]
    
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Sentiment Analysis Result</title>
    </head>
    <body>
        <h1>Sentiment Analysis Result</h1>
        <p>The predicted sentiment is: {}</p>
        <a href="/">Go back to input page</a>
    </body>
    </html>
    """.format(sentiment_label)

if __name__ == '__main__':
    app.run(debug=True)
