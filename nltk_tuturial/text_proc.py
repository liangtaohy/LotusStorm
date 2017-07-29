# coding=utf-8
from __future__ import division
from nltk.corpus import genesis


def lexical_diversity(text):
    """
    lexical diversity

    :param text:
    :return:
    """
    return len(text) / len(set(text))


def plural(word):
    """
    plural of word

    :param word:
    :return:
    """
    if word.endswith('y'):
        return word[:-1] + 'ies'
    elif word[-1] in 'sx' or word[-2:] in ['sh', 'ch']:
        return word + 'es'
    elif word.endswith('an'):
        return word[:-2] + 'en'
    else:
        return word + 's'