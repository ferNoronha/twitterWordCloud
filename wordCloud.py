from wordcloud import WordCloud, STOPWORDS
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import re, string, unicodedata
import nltk
import contractions
import inflect
#from bs4 import BeautifulSoup
from nltk import word_tokenize, sent_tokenize
#from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
import os
import numpy as np

def transform_format(val):
    if val == 0:
        return 255
    else:
        return val


def denoise_text(linha):
    linha = linha.rstrip()
    linha = re.sub(r'http\S+', '', linha)
    translator = str.maketrans({key: None for key in string.punctuation})
    linha = linha.translate(translator)
    return linha

def replace_contractions(text):
    return contractions.fix(text)

def remove_non_ascii(words):
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def replace_numbers(words):
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words
def remove_stopwords(words):
    new_words = []
    stopwords_pt = nltk.corpus.stopwords.words('portuguese')
    stopwords_pt.append('rt')
    stopwords_pt.append('bolsonaro')
    for word in words:
        if word not in stopwords_pt:
            new_words.append(word)
    return new_words

def stem_words(words):
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems

def lemmatize_verbs(words):
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas

def normalize(words):
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = replace_numbers(words)
    words = remove_stopwords(words)
    return words


def get_tweet_text(linha):
    tweet_lower = ''
    linha = re.sub(r'http\S+', '', linha)
    tweet = linha.rstrip()
    print(tweet)
    translator = str.maketrans({key: None for key in string.punctuation})
    tweet = tweet.translate(translator)
    tweet_lower += tweet.lower()
    print(tweet_lower)
    return(tweet_lower)





tweets = pd.read_csv('tweets.csv',index_col=None)

bag_words = []
print(type(tweets))
for i in tweets.index:
    #print(tweets.loc[i,'tweets'])
    print(i)
    sample = denoise_text(tweets.loc[i,'tweets'])
    sample = replace_contractions(sample)
    words = nltk.word_tokenize(sample)
    bag_words = normalize(words) + bag_words
    #print(bag_words)
    #os.sys.exit()



#print(tweets.head(5))


#print(tweets.iloc[0,1])

brasil = np.array(Image.open("mapa-brasil.png"))


transformed_brasil_mask = np.ndarray((brasil.shape[0],brasil.shape[1]), np.int32)

for i in range(len(brasil)):
    transformed_brasil_mask[i] = list(map(transform_format, brasil[i]))

print(len(bag_words))
# width=600, height = 600
wordcloud = WordCloud(background_color = 'white', mask = transformed_brasil_mask,max_words=1000).generate((" ").join(bag_words))

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
