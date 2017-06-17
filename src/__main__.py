from __future__ import absolute_import, division, print_function

import sys
from functools import partial

from nltk.tokenize import TweetTokenizer
from twitter import OAuth, Twitter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from credentials import (consumer_key, consumer_secret,
                         oauth_token, oauth_secret)


def main():
    coke = search(q='Coke')
    pepsi = search(q='Pepsi')

    coke_tweets = [status['text'] for status in coke['statuses']]
    pepsi_tweets = [status['text'] for status in pepsi['statuses']]

    for tweet in coke_tweets + pepsi_tweets:
        diversity = lexical_diversity(tweet)
        sentiment = get_sentiment(tweet)
        print('=' * 79)
        print(u'\n{}\n\nLexical Diversity: {}\nSentiment: {}\n'.format(
            tweet, diversity, sentiment))


def lexical_diversity(text):
    """Return the lexical diversity of the given text."""
    tokenizer = TweetTokenizer(preserve_case=False)
    words = tokenizer.tokenize(text)
    return len(set(words)) / len(words)


def get_sentiment(text):
    """Return the sentiment of the given text."""
    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(text)['compound']
    if score >= 0.5:
        return 'positive'
    if score <= -0.5:
        return 'negative'
    return 'neutral'


def log_in():
    auth = OAuth(oauth_token, oauth_secret, consumer_key, consumer_secret)
    return Twitter(auth=auth)


if __name__ == '__main__':
    twitter = log_in()
    search = partial(twitter.search.tweets, count=25, lang='en')
    sys.exit(main())
