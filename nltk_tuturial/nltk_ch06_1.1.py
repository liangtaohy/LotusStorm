# coding=utf-8


def gender_features(word):
    """
    feature extractor
    :param word:
    :return:
    """
    return {'last_letter' : word[-1]}

import nltk
from nltk.corpus import names
import random


labeled_names = ([(name, 'male') for name in names.words('male.txt')]) + [(name, 'female') for name in names.words('female.txt')]

random.shuffle(labeled_names)
featuresets = [(gender_features(n), gender) for (n, gender) in labeled_names]
train_set, test_set = featuresets[500:], featuresets[:500]
classifier = nltk.NaiveBayesClassifier.train(train_set)

print(classifier.classify(gender_features('Neo')))
print(classifier.classify(gender_features('Trinity')))
print(nltk.classify.accuracy(classifier, test_set))

classifier.show_most_informative_features(5)