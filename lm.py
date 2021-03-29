
# PRE-PROCESSING AND LANGUAGE MODELING

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.lm.preprocessing import pad_both_ends, padded_everygram_pipeline
from nltk.util import flatten, bigrams, trigrams
from nltk.lm.models import KneserNeyInterpolated, MLE
import nltk
import json 
import re
import pickle
from const_module import NUM_TRAIN, NUM_TWEETS
import random

def preprocess(original_text):
    text = original_text.lower()
    sent_tokens = sent_tokenize(text)
    s = []
    for sent in sent_tokens:
        v = list(pad_both_ends(word_tokenize(sent), n=2))
        s.extend(v)
    return s    # returns a list of word tokens with padding

# UNPICKLE TO GET PICKLE_DICT
fp = open("tweets.txt", "rb")
PICKLE_DICT = pickle.load(fp)
TRAIN = PICKLE_DICT["train"]
TEST = PICKLE_DICT["test"]
fp.close()

# GENERATE UNIGRAMS
ugms = []
for tweet in TRAIN:
    s = preprocess(tweet)
    ugms.extend(s)
ugms = [(x,) for x in ugms]

# GENERATE BIGRAMS
bgms = []
for tweet in TRAIN:
    s = preprocess(tweet)
    b = bigrams(s)
    bgms.extend(list(b))

# GENERATE TRIGRAMS
trgms = []
for tweet in TRAIN:
    s = preprocess(tweet)
    t = trigrams(s)
    trgms.extend(list(t))

# GENERATE VOCABULARY
VOCAB = []
for tweet in TRAIN:
    s = preprocess(tweet)
    VOCAB.extend(s)
VOCAB = set(VOCAB)

# # TRAIN A UNIGRAMS LANGUAGE MODEL, TEST, FIND AVERAGE PERPLEXITY
UGMS_LM = KneserNeyInterpolated(1)
UGMS_LM.fit([ugms],vocabulary_text=VOCAB)
UGMS_PERPLEXITY = 0
for tweet in TEST:
    s = preprocess(tweet)
    s = [(x,) for x in s]
    try:
        UGMS_PERPLEXITY += UGMS_LM.perplexity(s)
    except ZeroDivisionError:
        pass
UGMS_PERPLEXITY /= len(TEST)
print("UGMS_PERPLEXITY:",UGMS_PERPLEXITY)

# TRAIN A BIGRAMS LANGUAGE MODEL, TEST, FIND AVERAGE PERPLEXITY
BGMS_LM = KneserNeyInterpolated(2)
BGMS_LM.fit([bgms],vocabulary_text=VOCAB)
BGMS_PERPLEXITY = 0
for tweet in TEST:
    s = preprocess(tweet)
    b = bigrams(s)
    try:
        BGMS_PERPLEXITY += BGMS_LM.perplexity(b)
    except ZeroDivisionError:
        pass
BGMS_PERPLEXITY /= len(TEST)
print("BGMS_PERPLEXITY:",BGMS_PERPLEXITY)

# TRAIN A TRIGRAMS LANGUAGE MODEL, TEST, FIND AVERAGE PERPLEXITY
TRGMS_LM = KneserNeyInterpolated(2)
TRGMS_LM.fit([trgms],vocabulary_text=VOCAB)
TRGMS_PERPLEXITY = 0
for tweet in TEST:
    s = preprocess(tweet)
    b = trigrams(s)
    try:
        TRGMS_PERPLEXITY += TRGMS_LM.perplexity(b)
    except ZeroDivisionError:
        pass
TRGMS_PERPLEXITY /= len(TEST)
print("TRGMS_PERPLEXITY:",TRGMS_PERPLEXITY)

# GENERATE TWEETS USING UNIGRAMS MLE LM
UGMS_MLE = MLE(1)
UGMS_MLE.fit([ugms],vocabulary_text=VOCAB)  # train Bigrams MLE LM
print("UNIGRAMS GENERATED TWEETS USING MLE LM:")
for i in range(10):
    print("\nSENTENCE", i+1,": ")
    UGMS_GENERATE = '<s> ' 
    while True:
        l = UGMS_MLE.generate(5) # l is a list of strings
        if '</s>' in l:
            s = " ".join(l)
            s = re.sub("</s>.*",'',s)
            UGMS_GENERATE = UGMS_GENERATE + s + " </s>"
            break
        UGMS_GENERATE = UGMS_GENERATE + " ".join(l) + " "
    print(UGMS_GENERATE)

# GENERATE TWEETS USING BIGRAMS MLE LM
BGMS_MLE = MLE(2)
BGMS_MLE.fit([bgms],vocabulary_text=VOCAB)  # train Bigrams MLE LM
print("BIGRAMS GENERATED TWEETS USING MLE LM:")
for i in range(10):
    print("\nSENTENCE", i+1,": ")
    BGMS_GENERATE = '<s> '  #BGMS_GENERATE IS A STRING
    seed = ['<s>']
    while True:
        l = BGMS_MLE.generate(5,text_seed=seed) # l is a list of strings
        if '</s>' in l:
            s = " ".join(l)
            s = re.sub("</s>.*",'',s)
            BGMS_GENERATE = BGMS_GENERATE + s + " </s>"
            break
        BGMS_GENERATE = BGMS_GENERATE + " ".join(l) + " "
        seed = [l[-1]]
    print(BGMS_GENERATE)


# GENERATE TWEETS USING TRIGRAMS MLE LM
TRGMS_MLE = MLE(3)
TRGMS_MLE.fit([trgms],vocabulary_text=VOCAB)  # train Bigrams MLE LM
print("TRIGRAMS GENERATED TWEETS USING MLE LM:")
for i in range(10):
    print("\nSENTENCE", i+1,": ")
    seed = list(random.choice(trgms))
    TRGMS_GENERATE = "<s> " + " ".join(seed) + ' '
    while True:
        try:
            l = TRGMS_MLE.generate(5,text_seed=seed) # l is a list of strings
        except ValueError:
            seed = list(random.choice(trgms))   # if context has zero count, reinitialize seed
        else:
            if '</s>' in l:
                s = " ".join(l)
                s = re.sub("</s>.*",'',s)
                TRGMS_GENERATE = TRGMS_GENERATE + s + " </s>"
                break
            TRGMS_GENERATE = TRGMS_GENERATE + " ".join(l) + " "
            seed = [l[-2:]]
    print(TRGMS_GENERATE)