
# TWEETS SENTIMENT ANALYSIS WITH VADER

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pickle
from nltk.corpus import stopwords
import nltk
from nltk.tokenize import word_tokenize 
import string

# IMPORT ALL TWEETS FROM TWEET.TXT USING PICKLE - VERIFIED PICKLE WORKS
fp = open("tweets.txt", "rb")
PICKLE_DICT = pickle.load(fp)
TRAIN = PICKLE_DICT["train"]
TEST = PICKLE_DICT["test"]
TRAIN.extend(TEST)
ALL_TWEETS = TRAIN
fp.close()

# PERFORM SENTIMENT ANALYSIS ON ALL TWEETS AND SEPARATE TWEETS INTO POSITIVE AND NEGATIVE LIST
POS_TWEETS = []
NEG_TWEETS = []
sid_obj = SentimentIntensityAnalyzer()
compound_sentiment = 0
for tweet in ALL_TWEETS:
    sentiment_dict = sid_obj.polarity_scores(tweet)
    compound_sentiment += sentiment_dict['compound']
    if sentiment_dict['compound'] <= 0.05:     # NEGATIVE SENTIMENT
        NEG_TWEETS.append(tweet)
    elif sentiment_dict['compound'] >= 0.05:    # POSITIVE SENTIMENT
        POS_TWEETS.append(tweet)
compound_sentiment /= len(ALL_TWEETS)
print("AV COMPOUND SENTIMENT: " , compound_sentiment)

# USE NLTK TO REMOVE STOPWORDS
stop_words = set(stopwords.words('english')) 
POS_TOKENS = []
NEG_TOKENS = []
for tweet in POS_TWEETS:
    POS_TOKENS.extend(word_tokenize(tweet))
for tweet in NEG_TWEETS:
    NEG_TOKENS.extend(word_tokenize(tweet))
POS_TOKENS = [word.lower() for word in POS_TOKENS]
NEG_TOKENS = [word.lower() for word in NEG_TOKENS]
FILTERED_POS_TOKENS = [w for w in POS_TOKENS if not w in stop_words if w.isalnum()]
FILTERED_NEG_TOKENS = [w for w in NEG_TOKENS if not w in stop_words if w.isalnum()]

# FIND TOP 10 WORDS IN NEGATIVE TWEETS AND TOP 10 WORDS IN POSITIVE TWEETS
N = 10
fdist = nltk.FreqDist()
fdist = nltk.FreqDist(FILTERED_POS_TOKENS)
POS_MOST_COMMON = [a_tuple[0] for a_tuple in fdist.most_common(N)]
print("10 MOST COMMON WORDS FOR POSITIVE TWEETS:\n", POS_MOST_COMMON )
fdist = nltk.FreqDist(FILTERED_NEG_TOKENS)
NEG_MOST_COMMON = [a_tuple[0] for a_tuple in fdist.most_common(N)]
print("10 MOST COMMON WORDS FOR NEGATIVE TWEETS:\n", NEG_MOST_COMMON)