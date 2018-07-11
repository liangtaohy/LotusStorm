# -*- coding: utf-8 -*-
import re


def remove_brackets(text):
    re_script = re.compile('\s*\u3010[^\u3011]*[^\u3010]*\s*\u3011\s*', re.I)  # Script
    text = re_script.sub('', text)
    return text


if __name__ == "__main__":
    print(remove_brackets("在公司首次公开发行上市前，如果公司发生清算(含公司并购、出售)、解散或结束营业的情况，则投资人有权获得原投资金额的【1.2】倍，或与其他股东同时按照股权（股份）比例对公司剩余财产进行分配，依据孰高原则，取两者之一。【XL:清算金额1-2倍是正常的，先改到1.2倍，谈判时如果需要可让步到1.5倍，2倍还是有点高。】"))