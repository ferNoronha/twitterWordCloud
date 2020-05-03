from wordcloud import wordcloud
from requests_oauthlib import OAuth1Session
from operator import add
import requests_oauthlib
import requests
import string
import ast
import pandas as pd
import pprint
import json




consumer_key = ""
consumer_secret = ""
access_token = ""
access_token_secret = ""

search_term = 'Bolsonaro'
#long_lat = '-69.64,-13.6,-69.64,-13.6'
#sample_url = 'https://stream.twitter.com/1.1/statuses/sample.json'
filter_url = 'https://stream.twitter.com/1.1/statuses/filter.json?track='+search_term
#filter_location = 'https://stream.twitter.com/1.1/statuses/filter.json?locations='+long_lat

auth = requests_oauthlib.OAuth1(consumer_key, consumer_secret, access_token, access_token_secret)

tweets = []
location = []
response = requests.get(filter_url, auth = auth, stream = True)
print( response)
count = 0
for line in response.iter_lines():
    try:
        if count > 1000:
            break
        post = json.loads(line.decode('utf-8'))
        print(count)
        contents = [post['text']]
        count += 1
        tweets.append(str(contents))
        location.append([post['user']['location']])
    except:
        result = False



tweets_series = pd.Series(tweets,name='tweets')
tweets_series.to_csv('tweets.csv')

locations = pd.Series(location,name='location')
locations.to_csv('locations.csv')

print('saved')

