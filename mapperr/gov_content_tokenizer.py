#coding=utf-8
import os
import re
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

debug = False


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
        if title is None or len(title) == 0:
            return ''

        title = title.strip()
        tokens = self.tokenizer.tokenize(title)
        if debug:
            print(tokens)
        doc_type = []
        for w, flag in tokens:
            if w in DocType:
                doc_type.append(w)

        if len(doc_type) > 0:
            return doc_type.pop()

        if title.find('法') > 0:
            return '法'

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

    def skip_ids(self):
        self.cursor.execute("SELECT rid FROM `law_entity` WHERE 1")
        rows = self.cursor.fetchall()
        ids = []
        for row in rows:
            ids.append(row['rid'])
        return ids

    def processing(self):
        total = self.total()
        pagesize = 1000
        pages = math.ceil(total / pagesize)

        f = open("law_entity.sql", "w")
        total = 0

        statics = {'publish': 0, 'valid': 0, 'total': 0}

        skipids = self.skip_ids()

        print(skipids)

        from_id = 256197

        for page in range(pages):
            self.cursor.execute("SELECT * FROM gov WHERE id > %d LIMIT %d,%d" % (from_id, page * pagesize, pagesize))
            rows = self.cursor.fetchall()
            for row in rows:
                if row['id'] in skipids:
                    continue

                if len(row['content']) == 0:
                    continue
                doc_type = self.guess_doc_type_from_title(row['title'])
                if doc_type in ['法', '条例', '管理办法', '管理条例', '令', '办法', '准则', '通则', '宪法']:
                    self.fetch_first_lines(row['content'])
                    parser = DocContentParser(row['content'])
                    print('id: %d' % row['id'])
                    t = parser.parse_time()
                    author = parser.parse_author().strip()
                    if len(author) == 0:
                        author = t['author']
                    if len(author) == 0:
                        author = row['author']

                    id = int(row['id'])
                    entity_name = row['title'].strip()

                    if parser.is_skip_title(row['title']) is True:
                        continue

                    if entity_name == '中华人民共和国国务院令':
                        ti = parser.guowuyuanling_normalize()
                        if len(ti) > 0:
                            entity_name = ti

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
                    ctime = int(round(time.time() * 1000))

                    if len(publish_time) > 8:
                        publish_time, number = re.subn(r'([0-9]\.)', '', publish_time)
                    if len(valid_time) > 8:
                        valid_time, number = re.subn(r'([0-9]\.)', '', valid_time)

                    sql = "INSERT INTO `law_entity` (`rid`, `entity_id`, `entity_name`, `publish_time`, `valid_time`, `invalid_time`, `author`, `doc_type`, `ctime`) VALUES (%d, '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s')" % (id, entity_id, self.db.escape_string(entity_name), publish_time, valid_time, invalid_time, author, doc_type, ctime)
                    self.cursor.execute(sql)
                    #print(sql)
                    f.write(sql + ";\n")
                    total += 1
                    print(total)
            self.db.commit()
        f.close()

        print("publish:%f, valid:%f" % (statics['publish'] / statics['total'], statics['valid'] / statics['total']))

if __name__ == '__main__':
    gov = GovContentTokenizer()
    gov.processing()
    #print(gov.guess_doc_type_from_title('中华人民共和国公司法'))

