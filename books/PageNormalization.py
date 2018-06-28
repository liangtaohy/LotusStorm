# -*- coding: utf-8 -*-
#from __future__ import unicode_literals
import re
import os
import hashlib
import json
import time
import sys


class PageNormalization:
    def __init__(self, book_name, publish_time, dir):
        """
        Constructor
        :param book_name: 书名
        :param publish_time: 发布时间
        """
        self.v = '0.0.1'
        self.book_name = book_name
        md5 = hashlib.md5()
        md5.update(self.book_name.encode('UTF-8'))
        self.book_id = md5.hexdigest()
        time_arr = time.strptime(publish_time, "%Y-%m-%d")
        time_stamp = int(time.mktime(time_arr))
        self.publish_time = time_stamp * 1000
        self.dir = dir

    def content_norm(self, content):
        """
        try to normalize the `content`

        :param content: page content, a block of text with utf-8 encoder
        :return:
        """
        result, number = re.subn(r'o', '。', content)
        result, number = re.subn(r'〇', '0', result)
        result, number = re.subn(r'女口', '如', result)
        result, number = re.subn(r'll', '11', result)
        result, number = re.subn(r'l.', '1.', result)
        result, number = re.subn(r'S.', '5.', result)
        result, number = re.subn(r'叩', '00', result)
        result, number = re.subn(r'_条', '一条', result)
        result, number = re.subn(r'>欠', '次', result)

        return result

    def pages(self, dir, skip_para=False):
        """
        iterator of pages of a book

        :param dir: directory of a book
        :param skip_para: if True, skip paragraph parser, or paragraph parsing
        :return:
        """

        pat = re.compile(r'([0-9]+)\.txt')

        para_num = 1

        for root, dirs, files in os.walk(dir):
            for name in files:
                has_txt = pat.search(name)

                if has_txt and has_txt.groups():
                    page_num = int(has_txt.groups()[0])
                else:
                    continue

                with open(os.path.join(root, name)) as fp:
                    content = fp.read()
                    content = self.content_norm(content)
                    if skip_para:
                        yield page_num, content

                    if not skip_para:
                        if not content.strip():
                            paragraphes = '\n\n'.split(content)
                            for p in paragraphes:
                                yield page_num, para_num, p
                                para_num += 1

    def build_page_json(self, dir):
        """
        build json data for page(s)
        :param dir: the directory of dir(s)
        :return:
        """
        pages = []
        for page_num, content in self.pages(dir + "/内容", skip_para=True):
            if len(content):
                r = {
                    "book_id": self.book_id,
                    "page_num": page_num,
                    "page_content": content,
                    "page_type": "content",
                    "data_type": 3,
                    "publish_time": self.publish_time
                }
                pages.append(r)
        f = open(dir + "/pages.json", "w")
        f.write(json.dumps(pages, ensure_ascii=False))
        f.close()
        return pages

    def build_category_json(self, dir):
        cates = []
        for page_num, content in self.pages(dir + "/目录", skip_para=True):
            r = {
                "book_id": self.book_id,
                "page_num": page_num,
                "page_content": content,
                "page_type": "category",
                "data_type": 3,
                "publish_time": self.publish_time
            }
            cates.append(r)

        f = open(dir + "/category.json", "w")
        f.write(json.dumps(cates, ensure_ascii=False))
        f.close()
        return cates

    def book_dump(self):
        self.build_page_json(self.dir)
        self.build_category_json(self.dir)

    def test_build_page_json(self, dir):
        self.build_page_json(dir)

    def test(self, dir):
        """
        test cases
        :param dir:
        :return:
        """
        print("book name: " + self.book_name)
        print("book id: " + self.book_id)

        for page_num, p in self.pages(dir, True):
            print('------------------------------')
            print(p)

if __name__ == '__main__':
    book_name = sys.argv[1]
    publish_time = sys.argv[2]
    dir = sys.argv[3]

    print("process book " + book_name + " begin")
    begin = time.time()
    pager = PageNormalization(book_name, publish_time, dir)
    #pager.test('/mnt/open-xdp/books/已处理/中国娱乐法/目录')
    #pager.test_build_page_json('/mnt/open-xdp/books/已处理/中国娱乐法/目录')
    pager.book_dump()
    end = time.time()

    diff = int(end - begin)
    print("finished")