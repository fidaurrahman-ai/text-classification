import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag

# Download NLTK resources
import nltk
nltk.download('popular')
nltk.download('punkt_tab')
#nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Function to map POS tag to wordnet format for lemmatization
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize the text
    words = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # POS tagging and lemmatization
    lemmatizer = WordNetLemmatizer()
    pos_tagged = pos_tag(words)
    words = [lemmatizer.lemmatize(word, get_wordnet_pos(tag) or wordnet.NOUN) for word, tag in pos_tagged]
    
    # Join words back into a single string
    return ' '.join(words)

# Example usage
sample_text = "This is a great movie! I loved the acting and the plot was thrilling."
processed_text = preprocess_text(sample_text)
print(processed_text)
