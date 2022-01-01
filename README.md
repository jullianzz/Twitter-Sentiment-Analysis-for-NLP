# Twitter Sentiment Analysis for NLP

Tweet generation and sentiment analysis using n-grams language model in Python. 

* Used the Twitter Developer API to extract 10,000 most recent tweets in the English Language from Twitter with the keyword 'covid', and trained unigram, bigram, and trigram language models with Kneser-Ney smoothing
* Utilized the NLTK library to preprocess and clean the scraped tweets, i.e. sentence segmenting, tokenizing, lower casing, and padding and computed compound sentiment analysis (using VADER) to understand Twitter user sentiment towards COVID-19
* Built an algorithm that generated tweets for each learned language model 


### Download and Run

NOTE TO THE USER
--------------------------------------------------------------
To run this version from step 0, create your own Twitter API account and fetch your own auth key. 
Otherwise, use the scrapped Tweets made available in tweets.json.zip file. and begin at step 1. 


TWITTER SCRAPING
---------------------------------------------------------------
ONLY complete these steps if json needs to be updated:

0. scraping.py -> tweets.json
0. prepreprocesing.py -> removes https and @username-> tweets.txt, test_tweets.txt

---------------------------------------------------------------
INSTRUCTIONS TO RUN CODE:

1. lm.py -> preprocess, language models, perplexity av for uni, bi, and tri grams
2. sa.py -> performs sentiment analysis


### Please Read
***ALL code written in this repository is under the authorship of Julia Zeng (@jullianzz), who belongs to the Electrical & Computer Engineering Department at Boston University. All code is written strictly for educational purposes and not authorized for redistribution or re-purposing in any domain or by any individual or enterprise.***
