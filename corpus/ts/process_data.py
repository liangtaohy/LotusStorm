import os
import json


f = open("./../samples/ts/regular_terms.txt", "r", encoding="utf-8")

terms = []
labels = {}
for line in f:
    l = line.strip().split(',')
    label = l[1]
    label_num = l[0]
    labels[label] = int(label_num)
    terms.append(l)

f.close()

f = open("./../samples/ts/labels.json", "w", encoding="utf-8")
json.dump(labels, open("./../samples/ts/labels.json", "w", encoding="utf-8"), ensure_ascii=False)

labeled_txt = []
for line in open("./../samples/ts/all_ts_terms.txt", "r", encoding="utf-8"):
    l = line.strip().split(',')
    matched = False
    for term in terms:
        if l[0] in term[1:]:
            labeled_txt.append("{0},{1}".format(term[0], line))
            matched = True
            break
    if matched is False:
        print(line)
f.close()

f = open("./../samples/ts/train_set.csv", "w", encoding="utf-8")
f.writelines(labeled_txt)
f.close()
