import re
import os
from framework import JiebaTokenizer
from . import DocType
from . import GovDict


class DocumentParser:
    def __init__(self, document, with_stop_words=False):
        self.entity = {
            'doc_type': '',
            'author': document['author'],
            'publish_time': '',
            'valid_time': '',
            'address': '',
            'level': '',  # 效力级别
            'id': '',
            'rels': {},
        }
        if 'id' in document:
            self.entity['id'] = document['id']
        if 'publish_time' in document and len(document['publish_time']):
            self.entity['publish_time'] = document['publish_time']
        if 'valid_time' in document and len(document['valid_time']):
            self.entity['valid_time'] = document['valid_time']

        self.title = document['title']
        self.tokens = []
        if with_stop_words:
            self.tokenizer = JiebaTokenizer.JiebaTokenizer(os.path.dirname(os.path.realpath(__file__)) +
                                                       '/../framework/stop_words_jieba.utf8.txt')
        else:
            self.tokenizer = JiebaTokenizer.JiebaTokenizer()

    def set_user_dict(self, user_dict_file):
        self.tokenizer.set_user_dict(user_dict_file)

    """
    parse text into token array with element like {word, flag}
    @:param text
    """
    def parse_title(self, title=''):
        if not title.strip():
            title = self.title
        else:
            self.title = title

        self.tokens = self.tokenizer.tokenize(title)

        for w, flag in self.tokens:
            if w in DocType.DocType:
                self.entity['doc_type'] = w

        if self.entity['author'] in GovDict.GovDeparments:
            print('author in GovDeparments ' + self.entity['author'])
        else:
            print('author ' + self.entity['author'])

        if self.entity['author'] == '全国人民代表大会常务委员会' or self.entity['author'] == '全国人民代表大会':
            if self.entity['doc_type'] in ['法', '通则', '条例']:
                self.entity['level'] = '法律'
        elif self.entity['author'] in GovDict.Gov:
            if self.entity['doc_type'] in ['条例', '办法', '决定', '细则', '规定', '命令', '令']:
                self.entity['level'] = '行政法规'
            else:
                self.entity['level'] = '规范性文件'
        elif self.entity['doc_type'] in ['复函', '函', '答复']:
            self.entity['level'] = '行政公文'
        elif self.entity['doc_type'] in ['条例'] and (self.entity['author'].find('自治州') or self.entity['author'].find('自治县') or self.entity['author'].find('自治区')):
            self.entity['level'] = '单行条例'
        elif self.entity['author'] in GovDict.GovDeparments or self.entity['author'] in GovDict.OrgSpecailUnderCouncil or self.entity['author'] in GovDict.OrgDirectlyUnderCouncil or self.entity['author'] in GovDict.InstuitionDerictyUnderGov or self.entity['author'] in GovDict.GovNationalOffices:
            if self.entity['doc_type'] in ['命令', '令', '指示', '规定', '办法']:
                self.entity['level'] = '部门规章'
            else:
                self.entity['level'] = '规范性文件'

        print(type(self.tokens))
        self.entity['text'] = self.title
        for w, flag in self.tokens:
            if flag == 'nt':
                if not self.entity['author']:
                    self.entity['author'] = w
            elif flag == 'ns':
                self.entity['address'] += w

    """
    parse relations from content
    """
    def parse_relations(self, content=''):
        if len(content) == 0:
            content = self.content

        if len(content) == 0:
            return None

        s = 0
        e = -1
        m = re.search("《.*?》", content[s:e])

        t,d = m.span()
        print(content[t:d])
        while m:
            s1, e1 = m.span()
            rel_value = content[s + s1:s + e1]
            print(rel_value)
            s += s1
            rel = content[s-2:s]
            s += e1
            m = re.search("《.*?》", content[s:e])
            if rel in GovDict.Rels:
                self.entity['rels'][rel_value] = rel
            else:
                self.entity['rels'][rel_value] = ''
