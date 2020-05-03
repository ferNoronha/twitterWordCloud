from wordcloud import WordCloud, STOPWORDS
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import nltk
import string
import re

#nltk.download('stopwords')

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



stopwords_pt = nltk.corpus.stopwords.words('portuguese')

tweets = pd.read_csv('tweets.csv',index_col=None)


#print(tweets.head(5))


#print(tweets.iloc[0,1])


wordcloud = WordCloud(stopwords=stopwords_pt).generate(get_tweet_text(tweets.iloc[0,1]))

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
