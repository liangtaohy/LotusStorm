import sys
from nltk.corpus import PlaintextCorpusReader
import nltk, re, pprint
from nltk import word_tokenize
from law import JiebaTokenizer
import pymysql
import math
import time
import hashlib
from law import Settings as settings
from law import InvalidEntity as StopEntity
from law import GovDict


class DocumentCitation:

    def __init__(self):
        self.db = pymysql.connect(
            host=settings.MYSQL_HOST,
            user=settings.MYSQL_USER,
            password=settings.MYSQL_PASS,
            db=settings.MYSQL_DB,
            charset=settings.CHARSET,
            cursorclass=pymysql.cursors.DictCursor
        )

        self.table = settings.SPIDER_TABLE

        self.cursor = self.db.cursor()
        print('db open')

    def __del__(self):
        self.db.close()
        print('db close')

    """
    auto test
    """
    def autotest(self):
        self.cursor.execute("SELECT VERSION()")
        data = self.cursor.fetchone()

        print("Database version : %s " % data)

    def total(self, id=0):
        self.cursor.execute("SELECT COUNT(*) AS total FROM %s WHERE id>=%d" % (self.table, id))
        data = self.cursor.fetchone()
        return data['total']

    def licence_entity_batch(self):

        return True

    def license_learning(self, from_id=0):
        self.cursor.execute("SELECT COUNT(*) AS total FROM document_citations WHERE 1")
        data = self.cursor.fetchone()
        total = data['total']
        pagesize = 1000
        pages = math.ceil(total / pagesize)

        titles = []
        for page in range(pages):
            self.cursor.execute("SELECT cite_doc_title FROM document_citations " + " WHERE id>=%d LIMIT %d,%d" % (from_id, page * pagesize, pagesize))
            rows = self.cursor.fetchall()

            valid_token = "许可"
            for row in rows:
                cite_doc_title = row['cite_doc_title']
                if cite_doc_title.find(valid_token) != -1:
                    titles.append(cite_doc_title)

        fdist = nltk.FreqDist(titles)
        fdist.plot(100, cumulative=False)
        f = open("license_entities.txt", 'w')
        for title in titles:
            f.write(title + "\n")
        f.close()
        #titles = list(set(titles))

    def str_normalization(self, str):
        str = str.replace("\n", "")
        str = str.strip()
        return str.replace(" ", "")

    def increment_insert_middle_entity(self, row):
        cite_doc_title = row['cite_doc_title']
        cite_type = row['cite_type']

        if cite_type == '简称':
            return False

        length = len(cite_doc_title) - 2  # remove 2 brackets

        cite_doc_title = self.str_normalization(cite_doc_title)

        if cite_doc_title in StopEntity.InvalidEntity:
            return False

        if length > 2 and length <= 40:
            md5 = hashlib.md5()
            md5.update(cite_doc_title.encode('UTF-8'))
            cite_title_md5 = md5.hexdigest()
            insert_sql = 'INSERT INTO middle_entity (entity_name, entity_name_md5, total, ctime) VALUES (\'%s\', \'%s\', %d, %d) ON DUPLICATE KEY UPDATE total=total+1' % (
            self.db.escape_string(cite_doc_title), cite_title_md5, 1, time.time() * 1000)
            self.cursor.execute(insert_sql)
            return True
        return False

    def load_into_middle_entity(self, delete_target_table=False):
        """
        从引文中加载中间实体
        实体过滤规则
        1. 实体字数>2且<40
        2. 实体不在stop entity词典里
        3. 实体类型不为『简称』
        此操作为批量操作
        :return:
        """

        #  if true, just delete target table
        if delete_target_table:
            self.cursor.execute("DELETE FROM middle_entity")

        pagesize = 1000
        self.cursor.execute("SELECT count(*) as total FROM `document_citations` WHERE 1")
        row = self.cursor.fetchone()

        total = int(row['total'])

        pages = math.ceil(total / pagesize)
        f = open("middle_entity_too_big.txt", "w+")
        for page in range(pages):
            self.cursor.execute("SELECT cite_doc_title, cite_type FROM `document_citations` WHERE 1 limit %d,%d" % (page * pagesize, pagesize))
            rows = self.cursor.fetchall()
            for row in rows:
                self.increment_insert_middle_entity(row)
            self.db.commit()
        f.close()

    def token_test(self):
        line = "《动产抵押登记办法》已经中华人民共和国国家工商行政管理总局局务会修订通过，现予公布，自2016年9月1日起施行。"
        line = "《城市房屋拆迁管理条例》已经2001年6月6日国务院第40次常务会议通过，现予公布"
        line = "国家认监委2017年第18号公 告《国家认监委关于对有关机构非法从事认证活动的公告》\n"

        tokens = JiebaTokenizer.JiebaTokenizer().tokenize(line)
        print(tokens)
        max = 100
        num = 1
        for (word, flag) in tokens:
            if flag != 'x':
                num += 1
            if num > max:
                return False
            if flag == 'nt':
                if GovDict.is_gov(word):
                    print(word)
                    return word
        return False

    def cite_entities(self):
        """

        :return:
        """
        self.cursor.execute("SELECT doc_id FROM `document_citations` order by ctime desc limit 1")
        last_record = self.cursor.fetchone()

        p = re.compile(u'\u300a.*?\u300b')
        from_id = last_record['doc_id']
        total = self.total(from_id)
        pagesize = 100
        pages = math.ceil(total / pagesize)
        for page in range(pages):
            self.cursor.execute("SELECT * FROM " + self.table + " WHERE id>=%d LIMIT %d,%d" % (from_id, page * pagesize, pagesize))
            rows = self.cursor.fetchall()
            for row in rows:
                # parse content into tokens
                tokens = JiebaTokenizer.JiebaTokenizer().tokenize(row['content'])

                citations = {'id': row['id']}
                entities = []
                i = 0

                nt_begin = 0
                nt_end = 0
                entity = ''

                for (word, flag) in tokens:
                    if flag == 'x' and word == '《':
                        nt_begin = i
                    if flag == 'x' and word == '》':
                        nt_end = i
                    if nt_begin > 0:
                        entity += word

                    if nt_begin < nt_end:
                        cite_type = tuple(tokens[nt_begin - 1])[0]
                        if cite_type == '\'':
                            cite_type = ''
                        entity = entity.strip()
                        if len(entity) <= 0:
                            continue
                        entities.append((entity, cite_type))
                        md5=hashlib.md5()
                        md5.update(entity.encode('UTF-8'))
                        cite_title_md5 = md5.hexdigest()
                        insert_sql = 'INSERT IGNORE INTO document_citations (doc_id, cite_doc_title, cite_title_md5, cite_doc_id, cite_type, status, ctime) VALUES (%d, \'%s\', \'%s\', \'%s\', \'%s\', %d, %d)' % (row['id'], self.db.escape_string(entity), cite_title_md5, 0, cite_type, 0, time.time()*1000)
                        print(insert_sql)
                        self.cursor.execute(insert_sql)
                        row1 = {
                            "cite_doc_title": entity,
                            "cite_type": cite_type
                        }
                        self.increment_insert_middle_entity(row1)
                        nt_begin = nt_end = 0
                        entity = ''
                    i += 1
                if len(entities):
                    citations['cites'] = entities
                    #print(citations)
                self.db.commit()
                #cites = list(set(p.findall(row['content'])))
                #if len(cites):
                #    print(cites)
                #    print(row['content'])
            #print(row['content'])


if __name__ == '__main__':
    cite = DocumentCitation()
    #cite.autotest()
    #print(cite.total())
    #cite.cite_entities()
    #cite.license_learning()
    #cite.load_into_middle_entity(delete_target_table=True)
    cite.token_test()
