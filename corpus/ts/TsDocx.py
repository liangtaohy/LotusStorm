#
# :author Liang Tao (liangtaohy@gmail.com)
#
# 处理TS文档，要求文档为docx类型
# 生成合并后的条款文件：all_ts_terms.txt, 格式为: term,term real content
# 目前，仅支持table类型的条款清单
# 对term做如下简单预处理：
#   term = re.sub(r'([：0-9\.．A-Z\u0020]+)', "", term)
#   term = re.sub(r's+', '', term)
#

import zipfile
import xml.etree.ElementTree as ET
import re
import os
import json


word_schema = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def get_document_xml(docx_filename):
    with open(docx_filename, 'rb') as f:
        zip = zipfile.ZipFile(f)
        xml_content = zip.read('word/document.xml')
    return xml_content


def get_xml_tree(xml_string):
    return ET.fromstring(xml_string)


def _check_element_is(element, type_char):
    return element.tag == '{%s}%s' % (word_schema, type_char)


def _tag(tag):
    return '{%s}%s' % (word_schema, tag)


def _itertext(my_etree):
    """Iterator to go through xml tree's text nodes"""
    for node in my_etree.iter():
        if _check_element_is(node, 't'):
            yield (node, node.text)


def to_text(xml_tree):
    for node, txt in _itertext(xml_tree):
        print(txt)


def get_all_terms(ts_dir):
    if not os.path.exists(ts_dir):
        print("FATAL %s not exited" % (ts_dir))
        exit(1)

    c = []
    for parent, _, filenames in os.walk(ts_dir):
        for filename in filenames:
            if filename[0] == '.':
                continue
            if os.path.isfile(os.path.join(parent, filename)) and filename.endswith('.docx'):
                c.extend(get_terms_from_file(os.path.join(parent, filename)))

    return c


def get_terms_from_file(ts_file):
    print("load %s" % ts_file)
    xml_from_file = get_document_xml(ts_file)

    xml_tree = get_xml_tree(xml_from_file)

    root = xml_tree

    c = []

    for body in root.iter('{%s}%s' % (word_schema, 'body')):
        for child in body:
            if _check_element_is(child, 'tbl'):
                for tr in child:
                    if _check_element_is(tr, 'tr'):
                        tcs = tr.findall(_tag('tc'))
                        if tcs is None:
                            continue
                        size = len(tcs)

                        term = [text for node, text in _itertext(tcs[0])]

                        if size >= 2:
                            content = []
                            for n in tcs[1:]:
                                content.extend([text for node, text in _itertext(n)])

                            term = "".join(term)
                            term = term.strip()
                            if len(term) == 0:
                                continue
                            term = re.sub(r'([：0-9\.．A-Z\u0020:]+)', "", term)
                            term = re.sub(r's+', '', term)
                            c.append("{},{}".format(term, "".join(content)))

    return c


if __name__ == '__main__':
    ts_dir = '/Users/xlegal/百度云同步盘/TS/corpus'
    content = get_all_terms(ts_dir)
    print(json.dumps(content))
    f = open("./all_ts_terms_1.txt", "w", encoding="utf-8")
    for c in content:
        f.write("%s\n" % c)
    f.close()
    print("get terms %d" % len(content))
