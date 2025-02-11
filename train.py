import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load your dataset
data = pd.read_csv('stock_data.csv')  # Replace with your actual file path

# Assuming 'text' is your message column and 'sentiment' is your target column
X = data['Text']
y = data['Sentiment']

# Map 1 -> 'positive' and -1 -> 'negative'
y = y.map({1: 'positive', -1: 'negative'})

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text using TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Initialize and train the model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))

# Save the model and vectorizer
with open('model.pickle', 'wb') as m_file:
    pickle.dump(model, m_file)

with open('vectorizer.pickle', 'wb') as v_file:
    pickle.dump(vectorizer, v_file)
