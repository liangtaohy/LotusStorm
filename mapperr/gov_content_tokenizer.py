#coding=utf-8
import os
import pymysql
import math
import time
import hashlib
from hdfs import Config
from law import JiebaTokenizer
from law import Settings as settings
from law import InvalidEntity as StopEntity
from law import GovDict
from law.DocType import DocType
from law.DocumentContentParser import DocContentParser


class GovContentTokenizer:
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

        #self.client = Config().get_client('dev')

        if with_stop_words:
            self.tokenizer = JiebaTokenizer.JiebaTokenizer(os.path.dirname(os.path.realpath(__file__)) +
                                                       '/../framework/stop_words_jieba.utf8.txt')
        else:
            self.tokenizer = JiebaTokenizer.JiebaTokenizer()

    def total(self, id=0):
        self.cursor.execute("SELECT COUNT(*) AS total FROM %s WHERE id>=%d" % (self.table, id))
        data = self.cursor.fetchone()
        return data['total']

    def guess_doc_type_from_title(self, title):
        tokens = self.tokenizer.tokenize(title)

        doc_type = []
        for w, flag in tokens:
            if w in DocType:
                doc_type.append(w)

        if len(doc_type):
            return doc_type.pop()
        return ''

    def date_normalization(self, str):
        return str

    def fetch_first_lines(self, content=''):
        lines = content.split('\n')
        f = open("first_lines.txt", "w+")
        size = len(lines)
        max = 40
        if size < 40:
            max = size

        #print(lines[:max])
        for line in lines[:max]:
            f.write(line + "\n")
        f.write("==========\n")
        f.close()

    def processing(self):
        total = self.total()
        pagesize = 100
        pages = math.ceil(total / pagesize)

        f = open("law_entity.sql", "a")
        total = 0

        statics = {'publish': 0, 'valid': 0, 'total': 0}

        for page in range(pages):
            self.cursor.execute("SELECT * FROM gov WHERE 1 LIMIT %d,%d" % (page * pagesize, pagesize))
            rows = self.cursor.fetchall()
            for row in rows:
                doc_type = self.guess_doc_type_from_title(row['title'])
                if doc_type in ['法', '条例', '管理办法', '管理条例', '令']:
                    self.fetch_first_lines(row['content'])
                    parser = DocContentParser(row['content'])
                    t = parser.parse_time()
                    author = parser.parse_author().strip()
                    if len(author) == 0:
                        author = t['author']
                    if len(author) == 0:
                        author = row['author']

                    id = int(row['id'])
                    entity_name = row['title'].strip()
                    md5 = hashlib.md5()
                    md5.update(entity_name.encode('UTF-8'))
                    entity_id = md5.hexdigest()
                    publish_time = t['publish_time']
                    valid_time = t['valid_time']
                    if publish_time != '':
                        statics['publish'] += 1
                    if valid_time != '':
                        statics['valid'] += 1
                    statics['total'] += 1

                    invalid_time = ''
                    ctime = ''
                    sql = "INSERT INTO `law_entity` (`id`, `entity_id`, `entity_name`, `publish_time`, `valid_time`, `invalid_time`, `author`, `doc_type`, `ctime`) VALUES (%d, '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s')" % (id, entity_id, entity_name, publish_time, valid_time, invalid_time, author, doc_type, ctime)
                    print(sql)
                    f.write(sql + "\n")
                    total += 1

            if total > 50:
                break
        f.close()

        print("publish:%f, valid:%f" % (statics['publish'] / statics['total'], statics['valid'] / statics['total']))

if __name__ == '__main__':
    gov = GovContentTokenizer()
    gov.processing()

