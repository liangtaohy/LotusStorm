#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from nltk.tokenize.api import TokenizerI
import jieba
jieba.load_userdict(os.path.dirname(os.path.realpath(__file__)) + '/../framework/user_dict.txt')
import jieba.analyse
import jieba.posseg as pseg
from law import GovDict


class JiebaTokenizer(TokenizerI):
    r"""
        Interface to the Jieba Tokenizer
        """

    def __init__(self, stop_words_path='', just_last_token=False):
        self.name = 'JiebaTokenizer'
        if stop_words_path.strip() != '':
            self.stopList = [line.strip() for line in open(stop_words_path, encoding='utf-8')]
            self.stopList = self.stopList + [' ', '\n', '\t']
            jieba.analyse.set_stop_words(stop_words_path)
        else:
            self.stopList = []
        self.just_last_token = just_last_token
        self.user_dict_file = ''

        for k, v in GovDict.GovDeparments.items():
            jieba.add_word(k, 10000, 'nt')
            if v:
                jieba.add_word(v, 10000, 'nt')

        for k, v in GovDict.GovNationalOffices.items():
            jieba.add_word(k, 10000, 'nt')
            if v:
                jieba.add_word(v, 10000, 'nt')

        for k, v in GovDict.InstuitionDerictyUnderGov.items():
            jieba.add_word(k, 10000, 'nt')
            if v:
                jieba.add_word(v, 10000, 'nt')

        for k, v in GovDict.InstuitionGov.items():
            jieba.add_word(k, 10000, 'nt')
            if v:
                jieba.add_word(v, 10000, 'nt')

        for k, v in GovDict.OrgDirectlyUnderCouncil.items():
            jieba.add_word(k, 10000, 'nt')
            if v:
                jieba.add_word(v, 10000, 'nt')

        for k, v in GovDict.OrgSpecailUnderCouncil.items():
            jieba.add_word(k, 10000, 'nt')
            if v:
                jieba.add_word(v, 10000, 'nt')

        for k, v in GovDict.Gov.items():
            jieba.add_word(k, 10000, 'nt')
            if v:
                jieba.add_word(v, 10000, 'nt')

    def set_user_dict(self, user_dict_file):
        self.user_dict_file = user_dict_file
        jieba.load_userdict(self.user_dict_file)

    """
    tokenize
    @:param s, string
    """
    def tokenize(self, s):
        segs = []
        if s:
            segs_list = pseg.cut(s)
            if len(self.stopList):
                segs = [(word, flag) for (word, flag) in list(segs_list) if word not in self.stopList]
            else:
                segs = list(segs_list)
            if self.just_last_token:
                segs = [segs[-1]]
        return segs

    """
    tokenize_sents
    @:param sents
    """
    def tokenize_sents(self, sents):
        return [self.tokenize(s) for s in sents]