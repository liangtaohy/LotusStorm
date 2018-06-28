#coding=utf-8
import os
import pymysql
from framework import JiebaTokenizer
from law import Settings as settings


class EntityRelation:
    link_id = 1
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

    def make_relation(self, entity_name, from_id=0):
        self.cursor.execute("SELECT law_entity.rid, law_entity.entity_name, law_entity.author, cite_type, gov.url, gov.title FROM `document_citations`, law_entity, gov WHERE law_entity.rid=gov.id and law_entity.rid=document_citations.doc_id and document_citations.cite_doc_title='%s'" % entity_name)
        rows = self.cursor.fetchall()

        print(entity_name)
        print(from_id)
        nodes = []
        if from_id == 0:
            nodes.append({'id': 0, 'name': entity_name, 'author': '', 'type': 1})
        links = []
        for row in rows:
            if row['entity_name'].find('失效') > 0:
                continue

            if entity_name == row['title'].strip():
                continue
            nodes.append({'id': row['rid'], 'name': row['entity_name'], 'author': row['author']})
            links.append({'id': self.link_id, 'from': from_id, 'to': row['rid'], 'rela': row['cite_type'], 'url': row['url']})
            self.link_id += 1
            n, l = self.make_relation(row['entity_name'], row['rid'])
            nodes += n
            links += l

        return nodes, links

if __name__ == '__main__':
    r = EntityRelation()
    print(r.make_relation("《药品生产企业许可证》"))