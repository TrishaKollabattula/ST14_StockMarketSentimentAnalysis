from flask import Flask, request, render_template
import pickle
from preprocess import transform_text  # Make sure this is defined properly

# Initialize the Flask app
app = Flask(__name__)

# Load the model and vectorizer
with open("model.pickle", "rb") as m_file:
    model = pickle.load(m_file)

with open("vectorizer.pickle", "rb") as v_file:
    vectorizer = pickle.load(v_file)

# Define the routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    
    # Apply preprocessing to the message (e.g., tokenization, cleaning)
    transformed_text = transform_text(message)
    
    # Ensure transformed_text is a string (if it's a list, join it into a string)
    if isinstance(transformed_text, list):
        transformed_text = ' '.join(transformed_text)

    # Vectorize the input text
    vectorized_text = vectorizer.transform([transformed_text])

    # Predict sentiment (will now return 'positive' or 'negative')
    sentiment = model.predict(vectorized_text)[0]

    # Define sentiment responses
    sentiment_responses = {
        'positive': "The sentiment is positive! Market trends seem favorable.",
        'negative': "The sentiment is negative. Consider analyzing further."
    }

    # Get the response based on the sentiment
    response_text = sentiment_responses.get(sentiment, "Sentiment unclear.")
    
    return render_template('result.html', sentiment=sentiment, response=response_text)

if __name__ == '__main__':
    app.run(debug=True)

