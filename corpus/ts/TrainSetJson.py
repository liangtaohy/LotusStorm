import os
import json

label_dict = json.load(open("../samples/ts/labels.json", "r", encoding="utf-8"))

keywords = {}

all_terms = []

with open("./../../tf/bayes/.label_keywords.txt", "r", encoding="utf-8") as f:
    for line in f:
        l = line.strip()
        l = l.split(",")
        keywords['t%s' % l[0]] = l[1]

for v, k in label_dict.items():
    item = {
        "id": k,
        "label": v,
        "keywords": keywords['t%d' % k],
        "samples": [],
        "total": 0,
    }

    all_terms.append(item)

with open("./../samples/ts/sort_train_set.csv", "r", encoding="utf-8") as f:
    for line in f:
        l = line.strip()
        l = l.split(",")
        id = int(l[0])
        term_name = l[1]
        content = "".join(l[2:])

        item = {
            'term': term_name,
            'content': content,
        }

        for i in range(len(all_terms)):
            if all_terms[i]['id'] == id:
                all_terms[i]['total'] += 1
                all_terms[i]['samples'].append(item)
                break

    json.dump(all_terms, open("tmp.json", "w", encoding="utf-8"), ensure_ascii=False)