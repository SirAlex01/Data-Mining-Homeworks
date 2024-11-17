import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, PorterStemmer
import re

# download the needed resources
# nltk.download('stopwords')
# nltk.download('punkt')

class StringPreprocessor:
    def __init__(self, lang="english"):
        # define stemmer and stopwards, based on lang parameter
        if lang == "english":
            self.stemmer = PorterStemmer()
            self.stopwords = stopwords.words()
        elif lang == "italian":
            self.stemmer = SnowballStemmer("italian")  
            self.stopwords = stopwords.words("italian")
    
    def normalize(self, text): 
        # convert text to lowercase      
        normalized_text = text.lower()
        # remove punctuation and non alfanumeric symbols, but keep dot between digits
        normalized_text = re.sub(r'[^a-z0-9.]|(?<!\d)\.(?!\d)', ' ', normalized_text)

        # remove consecutive spaces
        normalized_text = re.sub(r' +', ' ', normalized_text)
        
        return normalized_text
    
    def remove_stopwords(self, text):
        words = text.split()
        # remove the stopwards
        filtered_words = [word for word in words if word not in self.stopwords]

        return " ".join(filtered_words)
    
    def stemming(self, text):
        words = text.split()
        # apply stemming
        stemmed_words = [self.stemmer.stem(word) for word in words]
        return " ".join(stemmed_words)
    
    def preprocess(self, text):
        #apply all operations in cascade
        normalized_text = self.normalize(text)
        no_stopwords_text = self.remove_stopwords(normalized_text)
        stemmed_text = self.stemming(no_stopwords_text)

        return stemmed_text
