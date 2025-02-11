import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data if you haven't already
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize Lemmatizer
lemmatizer = WordNetLemmatizer()

def transform_text(text):
    """
    Preprocess the input text for sentiment analysis:
    - Lowercasing
    - Removing special characters
    - Tokenizing
    - Removing stopwords
    - Lemmatizing words
    """
    # Step 1: Lowercase the text
    text = text.lower()

    # Step 2: Remove non-alphabetical characters (keeping only letters and spaces)
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Step 3: Tokenize the text into words
    words = word_tokenize(text)

    # Step 4: Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Step 5: Lemmatization - Convert words to their base form
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]

    # Return the processed text as a space-separated string
    return ' '.join(lemmatized_words)
