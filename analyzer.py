from processor import WordProcessorBase
from classifierbase import ClassifierBase
from votedclassifier import VoteClassifier


class Analyzer:

    def __init__(self, test_amount):
        self.word_processor = WordProcessorBase()
        self.classifier_base = ClassifierBase()
        self.init_analyzer(test_amount)

    def prepare_documents(self):
        print('Preparing Raw Sample Documents...')
        self.word_processor.open_files()
        self.word_processor.process_files()
        print('Document preparation completed.')

    def process_words(self):
        print('Processing Training Data...')
        self.word_processor.frequency_distribution()
        self.word_processor.identify_features(3000)
        self.word_processor.form_feature_sets()
        self.training_set = self.word_processor.set_training_data(10, 1900)
        self.testing_set = self.word_processor.set_testing_data(1900)
        print('Training Data processing completed.')

    def init_classifiers(self):
        print('Initializing Classifiers...')
        original_classifier = self.classifier_base.init_default_classifier(
            self.training_set, self.testing_set)
        print('Most Distinctive Keywords: ')
        original_classifier['classifier'].show_most_informative_features(15)

        self.classifier_base.init_sklearn_classifiers()
        self.classifier_base.train_classifiers(
            self.training_set, self.testing_set)

    def init_voted_classifier(self, feature_count):
        features = self.word_processor.feature_sets[:feature_count]
        self.classifier_base.init_voted_classifier(features, self.testing_set)

    def init_analyzer(self, test_amount):
        self.prepare_documents()
        self.process_words()
        self.init_classifiers()
        self.init_voted_classifier(test_amount)

    def analyze_tweet(self, text):
        feats = self.word_processor.find_features(text)
        sentiment = self.classifier_base.voted_classifier['classifier'].classify(
            feats)
        conf = self.classifier_base.voted_classifier['classifier'].confidence(
            feats)
        # if type(conf) != str:
        #     conf = conf * 100
        #     print('Voted Classification: {} with Confidence: {} %'.format(
        #         sentiment, conf))
        return sentiment, conf
