
# PRE-PRE-PROCESSING

import pickle
import json 
import re
from const_module import NUM_TRAIN, NUM_TWEETS

file = open('tweets.json', 'r')
results = json.load(file)
file.close()
all_tweets = []
for i in range(len(results)):
    all_tweets.extend(results[i]['statuses'])

training_set = all_tweets[0:NUM_TRAIN]
test_set = all_tweets[NUM_TRAIN:]

# TRAINING_TWEETS IS A LIST OF TWEET STRINGS
training_tweets = []
for tweet in training_set:
    if 'retweeted_status' in tweet:
        t = tweet['retweeted_status']['full_text']
    else:
        t = tweet['full_text']
    x = re.split("https", t)
    x = x[0].replace('\n', '')
    Tweet = re.sub('@[^\s]+','',x)
    training_tweets.append(Tweet)

# TEST_TWEETS IS A LIST OF TWEET STRINGS
test_tweets = []
for tweet in test_set:
    if 'retweeted_status' in tweet:
        t = tweet['retweeted_status']['full_text']
    else:
        t = tweet['full_text']
    x = re.split("https", t)
    x = x[0].replace('\n', '')
    Tweet = re.sub('@[^\s]+','',x)
    test_tweets.append(Tweet)

# CREATE A DICT OF THE TRAINING_TWEETS AND TEST_TWEETS
pickle_dict = {
    "train": training_tweets,
    "test": test_tweets
}

open("tweets.txt","w").close()      # clear tweets.txt
fp = open("tweets.txt", "wb")  # pickle pickle_dict list
pickle.dump(pickle_dict, fp)