# -*- coding: utf-8 -*-
# 解析html类型的ts文档
#
import os
import re
from bs4 import BeautifulSoup


class TsHtml:
    def __init__(self, sample_dir):
        self.sample_dir = sample_dir

    def get_trees(self):
        with os.scandir(self.sample_dir) as it:
            for entry in it:
                print(entry)
                if not entry.name.startswith('.') and entry.is_file() and entry.name.endswith('.htm'):
                    tree = BeautifulSoup(open("%s/%s" % (self.sample_dir, entry.name), encoding="utf-8"), "html5lib", from_encoding="utf8")
                    yield tree
        return None

    def get_items(self):
        tree_index = 0

        ts_terms = []

        for tree in self.get_trees():
            if tree:
                tree_index += 1
                ts_arr = []
                for tr in tree.select('table tr'):
                    lines = []
                    for td in tr.select('td p'):
                        text = td.get_text().strip()
                        if len(text) > 0:
                            lines.append(text)
                    print("\n".join(lines))
                    i = 0
                    for l in lines:
                        i += 1
                        match = re.match(r'^[0-9]+', l)
                        if not match:
                            ts_terms.append(l)
                            ts_arr.append("%s\t%s" % (l, "".join(lines[1:])))
                            break

                ts_txt = open("%s/ts_%d.txt" % (self.sample_dir, tree_index), "w", encoding="utf-8")
                ts_txt.write("\n".join(ts_arr))
                ts_txt.close()

                ts_terms = list(set(ts_terms))
                ts_terms_txt = open("%s/ts_terms.txt" % self.sample_dir, "w", encoding="utf-8")
                ts_terms_txt.write("\n".join(ts_terms))
                ts_terms_txt.close()


if __name__ == '__main__':
    ts = TsHtml(sample_dir='../samples/ts/html')
    ts.get_items()
