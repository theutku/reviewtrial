import nltk
from nltk.tokenize import word_tokenize
from documentprocessorbase import DocumentProcessorBase
from classifierbase import ClassifierBase


class WordProcessorBase(DocumentProcessorBase):

    def frequency_distribution(self):
        word_freq = nltk.FreqDist(self.all_words)
        self.word_freq = word_freq

    def identify_features(self, count):
        word_features = [word for (word, word_count)
                         in self.word_freq.most_common(count)]
        self.word_features = word_features

    def find_features(self, document):
        words = document
        if type(document) == str:
            words = word_tokenize(document)
        word_set = set(words)
        features = {}
        for word in self.word_features:
            features[word] = (word in word_set)

        return features

    def form_feature_sets(self):
        feature_sets = [(self.find_features(rev), category)
                        for (rev, category) in self.documents]

        self.feature_sets = feature_sets

    def set_training_data(self, start, end):
        training_set = self.feature_sets[start: end]
        return training_set

    def set_testing_data(self, start, end=None):
        if end is None:
            testing_set = self.feature_sets[start:]
        else:
            testing_set = self.feature_sets[start: end]
        return testing_set
