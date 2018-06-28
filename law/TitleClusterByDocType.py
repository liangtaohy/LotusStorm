#coding=utf-8

import re
import os
import math
import pymysql
from law import Settings as settings
from framework.JiebaTokenizer import JiebaTokenizer
from law.DocType import DocType

class TitleClusterByDocType:
    def __init__(self, with_stop_words=False):
        self.db = pymysql.connect(
            host=settings.MYSQL_HOST,
            user=settings.MYSQL_USER,
            password=settings.MYSQL_PASS,
            db=settings.MYSQL_DB,
            charset=settings.CHARSET,
            cursorclass=pymysql.cursors.DictCursor
        )

        self.table = 'gov'

        self.cursor = self.db.cursor()

        # self.client = Config().get_client('dev')

        if with_stop_words:
            self.tokenizer = JiebaTokenizer(os.path.dirname(os.path.realpath(__file__)) +
                                                           '/../framework/stop_words_jieba.utf8.txt')
        else:
            self.tokenizer = JiebaTokenizer()

    def process_all_titles(self):
        pagesize = 1000

        self.cursor.execute("SELECT COUNT(*) AS total FROM gov")
        row = self.cursor.fetchone()
        total = row['total']
        pages = math.ceil(total / pagesize)

        titles = []
        for page in range(pages):
            self.cursor.execute("SELECT id, title FROM gov WHERE 1 LIMIT %d, %d" % (page * pagesize, pagesize))
            rows = self.cursor.fetchall()
            for row in rows:
                if row['title'] is None:
                    continue
                titles.append(row['title'].strip())

        t = []
        for title in titles:
            result, number = re.subn(r'\(.*\)|（.*）', '', title)
            t.append(result)

        titles = list(set(t))  #remove repeated element

        doc_types = []
        for title in titles:
            tokens = self.tokenizer.tokenize(title)
            if len(tokens) == 0:
                continue

            token = tokens[-1]
            doc_types.append(token)

        doc_types = list(set(doc_types))

        for doc_type in doc_types:
            if doc_type not in DocType:
                print(doc_type)

    def test(self):
        result, number = re.subn(r'\(.*\)|（.*）', '', '中华人民共和国宪法')
        tokens = self.tokenizer.tokenize(result)
        print(tokens)
        print(result)
        print(number)

if __name__ == '__main__':
    doc = TitleClusterByDocType()
    doc.test()
    #doc.process_all_titles()