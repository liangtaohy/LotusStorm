# -*- coding: utf-8 -*-
import re

"""
class Extractor
"""


class Extractor:
    def __init__(self):
        self.version = '0.0.1'
        self.author = 'LiangTao <liangtaohy@163.com>'
        self.regex_chapter = re.compile(r"第[一二三四五六七八九十]+章")
        self.regex_section = ["#第[一二三四五六七八九十]+条 # i"]

    def extract_contents(self, document, type='txt'):
        """
        extract contents
        :param document:
        :param type:
        :return:
        """
        if type == 'txt':
            return self.extract_contents_txt(document)
        else:
            return None

    def extract_contents_txt(self, document):
        """
        extract contents from txt file
        :param document:
        :return:
        """
        contents = []
        """
        lines = document.split("\n")
        for line in lines:
            print(line)
            line = line.strip()
            result = self.regex_chapter.match(line)

            if result:
                contents.append(line)
        """
        if not isinstance(document, str):
            return contents
        total = len(document)
        if not total:
            return contents
        result = self.regex_chapter.match(document)
        while result:
            (pos, endpos) = result.span()
            while document[endpos]:
                endpos += 1
                if endpos >= total:
                    break
                if document[endpos] == '\n':
                    break

            contents.append((pos, endpos))
            result = self.regex_chapter.match(document, endpos)
        return contents
