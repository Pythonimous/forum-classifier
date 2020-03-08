import string
import nltk
#nltk.download('stopwords'), nltk.download('punkt')
import re
from rnnmorph.predictor import RNNMorphPredictor
from tqdm import tqdm

russian_stopwords = nltk.corpus.stopwords.words("russian")
predictor = RNNMorphPredictor(language="ru")

def preprocess_sentence(sen):
    sen = sen.rstrip().lower().replace('-то',' -то').replace('-нибудь',' -нибудь').replace('ё','е')
    for c in list(string.punctuation + '1234567890'):
        sen = sen.replace(c, ' ')
    sen = sen.replace('»', ' ').replace('«', ' ').replace('—', '').replace('–', '')
    sen = re.sub(u'[A-z]', u'', sen)
    sen = [i for i in sen.split() if i]

    tokens = [token for token in sen if token not in russian_stopwords]
    text = ' '.join([a.normal_form for a in predictor.predict(tokens)])
    return text