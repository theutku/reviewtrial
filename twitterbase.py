from analyzer import Analyzer
import json


class TwitterBase:

    def __init__(self):
        self.analyzer = Analyzer()
        self.analyzer.init_analyzer()

    def connect(self):
        return ''

    def analyze_tweet(self, text):
        feats = self.analyzer.word_processor.find_features(text)
        for classifier in self.analyzer.classifier_base._classifiers:
            sentiment = classifier['classifier'].classify(feats)
            conf = classifier['classifier'].confidence(feats)
            print('Classification: {} with Confidence: {} %'.format(
                sentiment, conf * 100))
            return sentiment, conf


from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
# import sentiment_mod as s


# from twitterapistuff import *
from twitterbase import TwitterBase


class TwitterListener(StreamListener):

    def init_listener(self):
        self.analyzer = Analyzer()
        self.analyzer.init_analyzer()
        consumer_key = 'SmsP3MaBZwEsoD7SeajQVfbj5'
        consumer_secret = 'rexAFZLxD0Z0zNVMr2dlnC283o0a6A97DGtndx38u5G68PS0Vm'
        access_token = '859723615880851456-1EnuyWtSPcoqvX8WZVn7EuaH6WHqfF6'
        access_token_secret = 'AiqYTDDnRJkoOdbNmPE3RP5MmzaO7Je1CZnnAFYu0DIds'

        self.auth = OAuthHandler(consumer_key, consumer_secret)
        self.auth.set_access_token(access_token, access_token_secret)

        # self.twitterStream = Stream(self.auth, self)
        # self.twitterStream.filter(track=['happy'])

    def on_data(self, data):

        all_data = json.loads(data)

        # TO DO
        tweet = all_data["text"]
        sentiment_value, confidence = self.analyze_tweet(tweet)
        print(tweet, sentiment_value, confidence)

        if confidence * 100 >= 80:
            output = open("twitter-out.txt", "a")
            output.write(sentiment_value)
            output.write('\n')
            output.close()

        return True

    def on_error(self, status):
        print(status)

    def analyze_tweet(self, text):
        feats = self.analyzer.word_processor.find_features(text)
        for classifier in self.analyzer.classifier_base._classifiers:
            sentiment = classifier['classifier'].classify(feats)
            conf = classifier['classifier'].confidence(feats)
            print('Classification: {} with Confidence: {} %'.format(
                sentiment, conf * 100))
            return sentiment, conf
