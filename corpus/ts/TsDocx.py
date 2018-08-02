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

#from TsTermTokens import TsTermDict


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


def _xml_tag(tag):
    return '{%s}%s' % ("http://www.w3.org/XML/1998/namespace", tag)


def _itertext(my_etree):
    """Iterator to go through xml tree's text nodes"""
    for node in my_etree.iter():
        if _check_element_is(node, 't'):
            space = node.get(_xml_tag('space'))
            text = node.text
            text = text.strip()
            if len(text) > 0 and space:
                text = "@#" + text + "#@"
            yield (node, text)


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


def guess_term_line(line, line_no):
    if line_no < 3:
        return False


def guess_title_line(line, line_no):
    if line_no > 6:
        return False
    l = line.split(",")
    if l[1] == 'center':
        return True


def guess_signature_page(line, line_no):
    if line_no < 15:
        return False
    match = re.search(r"(签字页|签章页|签章|签署|签字|盖章|投资人|公司)[：_）]|(无正文)", line)
    if match:
        return True
    return False


def guess_wp_type(ts_file):
    xml_from_file = get_document_xml(ts_file)
    xml_tree = get_xml_tree(xml_from_file)

    root = xml_tree

    wps = []

    line_num = 1
    wp_total = 0
    tr_total = 0
    for body in root.iter(_tag('body')):
        for child in body:
            if _check_element_is(child, 'p'):
                jc_val = '#'
                pPrs = child.findall(_tag('pPr'))
                if pPrs:
                    jcs = pPrs[0].findall(_tag('jc'))
                    if jcs:
                        jc = jcs[0]
                        val = jc.get(_tag('val'))
                        if val:
                            jc_val = val

                texts = [text for node, text in _itertext(child)]
                text = "".join(texts)
                text = re.sub(r's+', '', text)
                text = text.strip()
                if len(text) == 0:
                    line_num += 1
                    continue
                wps.append("{0},{1},{2},{3}".format(line_num, jc_val, len(text), text))
                wp_total += 1
                line_num += 1

            if _check_element_is(child, 'tbl'):
                for tr in child:
                    if _check_element_is(tr, 'tr'):
                        tcs = tr.findall(_tag('tc'))
                        if tcs is None:
                            continue
                        size = len(tcs)

                        term = []
                        if size >= 3:
                            for tt in tcs[0:2]:
                                term.extend([text for node, text in _itertext(tt)])
                            print(term)
                        else:
                            term = [text for node, text in _itertext(tcs[0])]
                        term = "".join(term)
                        term = re.sub(r'([：0-9\.．A-Z\u0020:]+)', "", term)
                        term = re.sub(r's+', '', term)
                        term = term.strip()
                        if len(term) == 0:
                            line_num += 1
                            continue

                        if size >= 2:
                            content = []
                            for n in tcs[1:]:
                                content.extend([text for node, text in _itertext(n)])
                            c = "@#{0}#@{1}".format(term, "".join(content))
                            wps.append("{0},{1},{2},{3}".format(line_num, 'tr', len(c), c))
                        tr_total += 1
                        line_num += 1

    return wps, line_num, tr_total, wp_total


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

                        term = []
                        if size >= 3:
                            for tt in tcs[0:2]:
                                term.extend([text for node, text in _itertext(tt)])
                            print(term)
                        else:
                            term = [text for node, text in _itertext(tcs[0])]
                        term = "".join(term)
                        term = re.sub(r'([：0-9\.．A-Z\u0020:]+)', "", term)
                        term = re.sub(r's+', '', term)
                        term = term.strip()
                        if len(term) == 0:
                            continue

                        if size >= 2:
                            content = []
                            for n in tcs[1:]:
                                content.extend([text for node, text in _itertext(n)])

                            c.append("{},{}".format(term, "".join(content)))
    return c


def handle_wps(wps, term_dict):
    doc = {
        "title": [],
        "terms": []
    }

    for wp in wps:
        t = wp.split(",")
        line_no = int(t[0])
        jc = t[1]
        words = t[2]
        c = "".join(t[3:])

        if line_no <= 6 and jc == 'center':
            doc['title'].append(c)

        if line_no > 6:
            matches = re.search(r'@#(.*)#@', c)
            if matches:
                p_term = matches.group(1)
                term_len = len(p_term)
                if 2 > term_len or term_len > 12: # 字数应该在[2,12]区间里（根据已有term统计结果，均值为5.4，符合泊松分布）
                    continue

                x_len = 0
                for t1 in term_dict:
                    if t1 in p_term:
                        x_len += len(t1)
                x = round(x_len / term_len, 3)  # 保留三位精度
                t2 = {
                    'p': x,
                    'line_no': line_no,
                    'jc': jc,
                    'term': p_term,
                    'content': c[c.index("#@") + 2:]
                }
                doc['terms'].append(t2)

    return doc


if __name__ == '__main__':
    #ts_dir = '/Users/xlegal/百度云同步盘/TS/corpus'
    ts_dir = "/Users/xlegal/PycharmProjects/LotusStorm/corpus/samples/ts/docx"
    wps, lines, tr_total, wp_total = guess_wp_type(ts_file=ts_dir + "/Term Sheet-chacha-XL CMTS-20160722.docx")
    #wps, lines, tr_total, wp_total = guess_wp_type(ts_file=ts_dir + "/艾瑞投资条款清单.docx")

    doc = handle_wps(wps, TsTermDict)
    json.dump(doc, open("test.json", "w", encoding='utf-8'), ensure_ascii=False)
    print(doc)

    """
    signatures = []
    for wp in wps:
        line_no = int(wp.split(",")[0])
        if guess_title_line(wp, line_no):
            print("标题: " + wp)
        if guess_signature_page(wp, line_no):
            signatures.append(line_no)
    print("\n".join(wps))
    print("可能是签字页的:")
    print(signatures)
    print("lines: {0}, tr_total: {1}, wp_total: {2}, {3}".format(lines, tr_total, wp_total, tr_total / (wp_total + tr_total - len(signatures))))
    exit(0)
    content = get_all_terms(ts_dir)
    print(json.dumps(content, ensure_ascii=False))
    f = open("./test_ts.txt", "w", encoding="utf-8")
    for c in content:
        f.write("%s\n" % c)
    f.close()
    print("get terms %d" % len(content))
    """
