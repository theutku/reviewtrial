import random
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import movie_reviews


class DocumentProcessorBase:

    def __init__(self):
        self.documents = []
        self.all_words = []

    def open_files(self):

        for category in movie_reviews.categories():
            for fileid in movie_reviews.fileids(category):
                self.documents.append(
                    (list(movie_reviews.words(fileid)), category))

        random.shuffle(self.documents)

    def process_files(self):

        for word in movie_reviews.words():
            self.all_words.append(word.lower())
