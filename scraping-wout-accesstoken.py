from twython import Twython, TwythonError
import json
from const_module import NUM_TRAIN, NUM_TWEETS

# ACCESS TWITTER
APP_KEY = 'INSERT APP KEY HERE'
APP_SECRET = 'INSERT APP SECRET HERE'
twitter = Twython(APP_KEY, APP_SECRET, oauth_version=2)
ACCESS_TOKEN = twitter.obtain_access_token()
twitter = Twython(APP_KEY, access_token=ACCESS_TOKEN)

# SCRAPE 10,000 TWEETS WITH 'COVID'
open('tweets.json', 'w').close()    # clear the json file
file_id = open('tweets.json','w')
results = []
for i in range(round(NUM_TWEETS/100)):
    r = twitter.search(q='covid', count=100, result_type='recent',
        tweet_mode="extended",wait_on_rate_limit=True)
    results.append(r)
# print(len(results))
# results is a list of dicts

file_id.write(json.dumps(results, indent=4))

file_id.close()