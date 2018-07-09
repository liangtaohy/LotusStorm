# -*- coding: utf-8 -*-
# 对数据进行shuffle，分别生成train_set数据和test_set数据
# :author Liang Tao (liangtaohy@163.com)
#

import os
import sys
import re
import json
import random


file = sys.argv[1]

if not os.path.exists(file):
    print("FATAL could not find file: {0}".format(file))
    exit(0)

tmp_dir = os.path.basename(file)

if not os.path.isdir(tmp_dir):
    os.mkdir(tmp_dir)

samples = {}
lables = {}
k = 1

train_set = []
test_set = []

with open(file, 'r', encoding='utf-8') as f:
    for sample in f:
        tmp = sample.split(",")
        label = tmp[0]
        content = re.sub(r"[a-zA-Z0-9]+", '', "".join(tmp[1:]))
        if label in samples:
            samples[label].append(content)
        else:
            samples[label] = [content]
        if label not in lables:
            lables[label] = k
            k += 1

    json.dump(lables, open("./{0}/label.json".format(tmp_dir), "w", encoding="utf-8"), ensure_ascii=False)

    for label in samples:
        size = len(samples[label])
        k = lables[label]
        if size >= 10:
            random.shuffle(samples[label])
            l = samples[label]
            left = int(size / 5)
            for t in l[:size - left]:
                train_set.append("{0},{1}".format(k, t))
            for test in l[size-left:]:
                test_set.append("{0},{1}".format(k, test))
        elif size > 1:
            random.shuffle(samples[label])
            l = samples[label]
            left = 1
            for t in l[:size - left]:
                train_set.append("{0},{1}".format(k, t))
            for test in l[size-left:]:
                test_set.append("{0},{1}".format(k, test))
        else:
            for x in samples[label]:
                train_set.append("{0},{1}".format(k, x))

    random.shuffle(train_set)
    random.shuffle(test_set)

    f = open("./{0}/train_set.txt".format(tmp_dir), "w", encoding="utf-8")
    f.writelines(train_set)
    f.close()

    f = open("./{0}/test_set.txt".format(tmp_dir), "w", encoding="utf-8")
    f.writelines(test_set)
    f.close()
