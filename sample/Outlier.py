#coding=utf-8

import numpy as np

def modzscore(file):
    with open(file, encoding='utf-8') as f:
        data = f.readlines()
    print(data)
    data = np.array(data, dtype=int)

    amax = np.amax(data, axis=0)
    print('max: ', amax)
    amin = np.amin(data, axis=0)
    print('min: ', amin)
    mean = np.mean(data)
    print('mean: ', mean)
    median = np.median(data)
    print('median: ', median)
    N = np.size(data)
    sumdiff = np.sum(pow((data - mean), 2)) / (N - 1)
    print("sumdiff: ", sumdiff)
    sqrtdiff = np.sqrt(sumdiff)
    print("sqrtdiff: ", sqrtdiff)
    mad = np.median(sqrtdiff)
    print('mad: ', mad)
    q = sqrtdiff

    #modzscore = (0.6745 * sumdiff) / mad
    modzscore = 3*q
    print(mean + modzscore, mean - modzscore, amin)

modzscore('股权类价格表.txt')
modzscore('主营业务类价格表.txt')
