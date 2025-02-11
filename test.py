import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

class TextToNum:

    def __init__(self,text):
        self.text=text

    def cleaner(self):
        cleaned_text = re.sub(r'[^\w\s]', '', self.text)  # Removing all characters except words and spaces
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Replacing multiple spaces with a single space
        cleaned_data = cleaned_text.strip()  # Stripping leading and trailing whitespace
        self.cleaned=cleaned_data
    
    def token(self):
        self.tkn= word_tokenize(self.cleaned)

    def removestop(self):
        stop = stopwords.words('english') 
        self.cl=[i for i in self.tkn if i not in stop]
    
    def stemme(self):
        ps=PorterStemmer()
        self.st=[ps.stem(word) for word in self.cl]
        return self.st

    