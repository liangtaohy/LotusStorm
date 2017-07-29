#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Created on 2017-06-01 14:58:14
# Project: safe_gov_zcfg

from pyspider.libs.base_handler import *
import MySQLdb
import md5
import time
import hashlib
import codecs

def md5(str):
    data = str
    m = hashlib.md5(data.encode("utf-8"))
    return (m.hexdigest())


class Handler(BaseHandler):
    crawl_config = {
    }

    conn = MySQLdb.connect(host="10.51.53.235", user="spider", passwd="n7TvYI5wlcEF9jpw", db="spider", charset="utf8",
                           port=3306)
    cursor = conn.cursor()

    @every(minutes=24 * 60)
    def on_start(self):
        self.crawl('http://www.seac.gov.cn/col/col144/index.html', fetch_type='js', callback=self.index_page)

    @config(age=10 * 24 * 60 * 60)
    def index_page(self, response):
        for each in response.doc('.default_pgContainer a').items():
            self.crawl(each.attr.href, fetch_type='js', callback=self.detail_page)
        for each in response.doc('#dynamicPage a').items():
            if each.attr.id.find('_nextPage') > 0:
                self.crawl(each.attr.href, fetch_type='js', callback=self.index_page)

    def insert_record(self, data):
        sql = "INSERT IGNORE INTO xlegal_law_content SET doc_id=%s,doc_ori_no=%s,type=3,author=%s,tags=%s,content=%s,url=%s,url_md5=%s,title=%s, status=1"
        param = (
            data['doc_id'], data['doc_ori_no'], data['author'], data['tags'], data['content'], data['url'], data['url_md5'], data['title']
        )

        self.cursor.execute(sql, param)
        self.conn.commit()

    @config(priority=2)
    def detail_page(self, response):
        url_md5 = md5(response.url)
        content = response.doc('#newsContent').text()
        doc_id = md5(content)
        response.doc('#Title script').remove()
        tags = response.doc('#lSubcat').text()

        fp = codecs.open("/mnt/open-xdp/spider/raw_data/tmp/" + url_md5 + ".html", "w", "utf-8")
        fp.write(response.doc('html').html())
        fp.close()

        data = {
            "doc_ori_no": response.doc('#wenhaotd span').text(),
            "author": "国家外汇管理局",
            "tags": tags,
            "type": 3,
            "doc_id": doc_id,
            "content": content,
            "url": response.url,
            "url_md5": url_md5,
            "title": response.doc('#Title').text(),
            "ctime": long(time.time() * 1000),
            "status": 1
        }
        self.insert_record(data)
        return data
