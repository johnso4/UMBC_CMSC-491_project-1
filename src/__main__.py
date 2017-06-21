from __future__ import absolute_import, division, print_function

import sys
import re
from itertools import chain
from functools import partial
from collections import Counter

from nltk.tokenize import TweetTokenizer
from twitter import OAuth, Twitter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from prettytable import PrettyTable

from credentials import (consumer_key, consumer_secret,
                         oauth_token, oauth_secret)


def main():
    coke_tweets, pepsi_tweets, coke_likes, coke_retweets, pepsi_likes, pepsi_retweets = get_tweets()
    coke_lists, pepsi_lists = vectorize(coke_tweets), vectorize(pepsi_tweets)

    tweets = coke_tweets + pepsi_tweets
    word_lists = coke_lists + pepsi_lists
    likes = coke_likes + pepsi_likes
    retweets = coke_retweets + pepsi_retweets

    for (tweet, words, like, retweet) in zip(tweets, word_lists, likes, retweets):
        diversity = lexical_diversity(words)
        sentiment = get_sentiment(tweet)
        print('=' * 79)
        print(u'\n{}\n\nLexical Diversity: {}\nSentiment: {}'.format(
            removeUnicode(tweet), diversity, sentiment))
        print("number of likes = " + like)
        print("number of retweets = " + retweet)
        print("=" * 79)

    print('\nCoke Term Document Matrix:')
    display_tdm(build_tdm(coke_lists))
    print('\nPepsi Term Document Matrix:')
    display_tdm(build_tdm(pepsi_lists))
    print('\nCoke Frequency Analysis:')
    displayWordCount(coke_tweets)
    print('\nPepsi Frequency Analysis:')
    displayWordCount(pepsi_tweets)


def build_tdm(corpus):
    """Return a term document matrix based on the given tweets."""
    tfs = compute_tfs(corpus)
    unique_words = set(flatten(corpus))
    return {word: [tf.get(word, 0) for tf in tfs] for word in unique_words}


def display_tdm(tdm):
    """Print the given term document matrix."""
    fields = ['Word'] + [str(x) for x in range(1, 26)]
    table = PrettyTable(field_names=fields)
    rnd = partial(round, ndigits=2)
    for (word, frequencies) in tdm.items():
        table.add_row([word] + map(rnd, frequencies))
    print(table)


def compute_tfs(corpus):
    """
    Return the term frequencies for the given word lists.

    The return value is a list of dictionaries, one for each tweet,
    each of which has all unique words from its respective tweet as
    its keys and the frequencies of those words as its values.

    Return value format:
    [
        {'So': 0.25, 'long': 0.25, 'and': 0.25, 'thanks': 0.25},  # for tweet 1
        {'for': 0.25, 'all': 0.25, 'the': 0.25, 'fish': 0.25},    # for tweet 2
    ]
    """
    word_counts = (Counter(word_list) for word_list in corpus)
    totals = (len(word_list) for word_list in corpus)
    return [{word: word_count[word] / total_words for word in word_count}
            for (word_count, total_words) in zip(word_counts, totals)]


def lexical_diversity(words):
    """Return the lexical diversity of the given text."""
    return len(set(words)) / len(words)


def get_sentiment(text):
    """Return the sentiment of the given text."""
    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(text)['compound']
    description = ('positive' if score >= 0.5 else
                   ('negative' if score <= -0.5 else 'neutral'))
    return '{} ({})'.format(score, description)


def get_tweets():
    """Return a corpus of tweets."""
    search = partial(twitter.search.tweets, count=25, lang='en')

    coke_handle = search(q='coke')['statuses']
    coke_tweets = [status['text'] for status in coke_handle]
    coke_likes = [str(status['favorite_count']) for status in coke_handle]
    coke_retweets = [str(status["retweet_count"]) for status in coke_handle]

    pepsi_handle = search(q='pepsi')['statuses']
    pepsi_tweets = [status['text'] for status in pepsi_handle]
    pepsi_likes = [str(status['favorite_count']) for status in pepsi_handle]
    pepsi_retweets = [str(status["retweet_count"]) for status in pepsi_handle]

    return (coke_tweets, pepsi_tweets, coke_likes, coke_retweets, pepsi_likes, pepsi_retweets)


def vectorize(tweets):
    """
    Return a list of word lists based on the given tweets.

    Each list of words represents the sequence of words from a single tweet.

    Return value format:
    [
        ['So', 'long', 'and', 'thanks'],  # words from tweet 1
        ['for', 'all', 'the', 'fish'],    # words from tweet 2
    ]
    """
    tokenizer = TweetTokenizer(preserve_case=False)
    vectors = (tokenizer.tokenize(tweet) for tweet in tweets)
    regex = re.compile(r'\w')
    return [[word for word in vector if regex.search(word)]
            for vector in vectors]


def flatten(corpus):
    """Combine all word lists in the corpus into a single list."""
    return chain.from_iterable(corpus)


def log_in():
    """Log in to Twitter."""
    auth = OAuth(oauth_token, oauth_secret, consumer_key, consumer_secret)
    return Twitter(auth=auth)

def displayWordCount(tweets):
    words=[]
    for tweet in tweets:
        for w in tweet.split():
            words.append(w)
    
    cnt = Counter(words)
    
    pt = PrettyTable(field_names=['Word','Count'])
    srtCnt = sorted(cnt.items(), key=lambda pair: pair[1], reverse=True)
    for kv in srtCnt:
        pt.add_row(kv)
    print(pt)

    print("============")
    print("Lexical Diversity")
    print(1.0*len(set(words))/len(words))

def removeUnicode(text):
    asciiText = ""
    for char in text:
        if(ord(char) < 128):
            asciiText = asciiText + char

    return asciiText

if __name__ == '__main__':
    twitter = log_in()
    sys.exit(main())
