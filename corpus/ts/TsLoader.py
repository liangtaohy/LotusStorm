# -*- coding: utf-8 -*-
import os
import json


with open("../samples/ts/train_set.csv", "r", encoding="utf-8") as f:
    num = 1
    for line in f:
        label = line.split(",")[0]
        tmp = line.split(",")[1:]
        tmp = "".join(tmp)
        if not os.path.exists("./data/{0}".format(label)):
            os.mkdir("./data/{0}".format(label))
        s = open("./data/{0}/{1}.txt".format(label, num), "w", encoding="utf-8")
        s.write(tmp)
        s.close()
        num += 1
