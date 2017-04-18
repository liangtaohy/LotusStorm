# -*- coding: utf-8 -*-
import sys
from optparse import OptionParser
import os
import re
import json
import html5lib
from html5lib.constants import namespaces
from xml.etree import ElementTree
from framework import MLog


def parse_js(file):
    """
    parse js object
    :param file:
    :return:
    """
    f = open(file, 'r', -1, 'utf-8')
    raw_js = f.read()
    m = re.search(r'jsonHtmlData = (\"\{.*\}\");', raw_js, re.MULTILINE | re.M | re.I)
    if m:
        json_obj = json.loads(m.group(1))
        json_obj = json.loads(json_obj)
        if json_obj['Html'] and isinstance(json_obj['Html'], str):
            parser = html5lib.HTMLParser(namespaceHTMLElements=False)
            document = parser.parse('<body>' + json_obj['Html'] + '</body>')
            document = text_etree_element(document)
            return document.replace(" ", "").replace("　", "")
    return None


def text_etree_element(element):
    segment = ""
    if not element:
        return ""
    if element is None:
        return ""

    segment = "".join(element.itertext()).strip()
    return segment
    """
    for child in element.findall("*"):
        if child is None:
            print("child None")
            continue
        tag = child.tag.lower()
        if tag == 'head':
            continue
        elif tag == 'body':
            if child.text:
                segment = segment + child.text + text_etree_element(child)
            else:
                segment = segment + text_etree_element(child)
            if child.tail:
                segment += child.tail
            continue
        elif tag == 'div':
            segment += "\n"
        elif tag == 'p':
            segment += "\n\n"
        elif tag == 'br':
            segment += "\n"

        if child.text:
            segment = segment + child.text + text_etree_element(child)
        else:
            segment = segment + text_etree_element(child)
        if child.tail:
            segment += child.tail
    return segment
    """


def process(save_fd, path):
    if not os.path.exists(path):
        print(path + "not exists")
        return False

    print(path)
    files = os.listdir(path)

    for f in files:
        if os.path.isdir(path + '/' + f):
            if f[0] == '.':
                pass
            process(save_fd, path + '/' + f)
        else:
            doc_str = parse_js(path + '/' + f)
            if isinstance(doc_str, str):
                print("saving %s into file" % (path + '/' + f))
                save_fd.write(doc_str + "\n")


def get_opt_parser():
    parser = OptionParser(usage=__doc__)
    parser.add_option("-s", "--source", action="store", default=False,type="string",
                      dest="path", help="Source Data Dir")

    parser.add_option("-d", "--dest",
                      action="store", default=False, dest="dest",type="string",
                      help="Dest File Path")

    parser.add_option("-b", "--treebuilder", action="store", type="string",
                      dest="treebuilder", default="etree")
    return parser


if __name__ == '__main__':
    optParser = get_opt_parser()
    opts, args = optParser.parse_args()
    print(opts)
    save_fd = open(opts.dest, "w")
    process(save_fd, opts.path)
