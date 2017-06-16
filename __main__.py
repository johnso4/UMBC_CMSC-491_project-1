import sys
from functools import partial

from twitter import OAuth, Twitter

from credentials import (consumer_key, consumer_secret,
                         oauth_token, oauth_secret)


def main():
    coke = search(q='Coke')
    pepsi = search(q='Pepsi')

    for status in coke['statuses']:
        tweet = status['text']
        print(tweet)


def log_in():
    auth = OAuth(oauth_token, oauth_secret, consumer_key, consumer_secret)
    return Twitter(auth=auth)


if __name__ == '__main__':
    twitter = log_in()
    search = partial(twitter.search.tweets, count=25, lang='en')
    sys.exit(main())
