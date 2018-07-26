"""
示例：python3 TextToTokens.py ../corpus/samples/ts/regular_terms.txt term_tokens.txt ../corpus/ts/ts_dict
输出：regular_terms.txt的全部分词

:author Liang Tao (liangtaohy@163.com)
"""

import os
import sys
sys.path.append("../")
import re

from framework.JiebaTokenizer import JiebaTokenizer
jieba = JiebaTokenizer(stop_words_path=os.path.dirname(os.path.realpath(__file__)) +
                                                    '/../framework/stop_words_jieba.utf8.txt')


def tokenize(content):
    tokens = jieba.tokenize(content)
    segs = [w for (w, _) in tokens]
    return segs


if __name__ == '__main__':
    input = sys.argv[1]
    output = sys.argv[2]


    user_dict = None

    if len(sys.argv) >= 4:
        user_dict = sys.argv[3]

    if user_dict:
        jieba.set_user_dict(user_dict_file=user_dict)

    f = open(input, "r", encoding='utf-8')
    lines = []

    with open(input, "r", encoding='utf-8') as f:
        for l in f:
            l = f.readline()
            if not l:
                break
            l = re.sub(r"([0-9]+,)", "", l)
            tokens = tokenize(l)
            lines.append(" ".join(tokens) + "\n")

    f = open(output, "w", encoding="utf-8")
    f.writelines(lines)
    f.close()