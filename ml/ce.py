#!/usr/bin/env python3
# Copyright (c) 2019, Konstantin Burlachenko (burlachenkok@gmail.com)

import matplotlib.pyplot as plt
import numpy as np
import math, sys

def isProbabilityMassFunction(x):
    if np.min(x) < 0:
        return False
    elif abs(np.sum(x) - 1.0) > 0.001:
        return False
    else:
        return True

def ce(p,q):
    pArr = np.asarray(p)
    qArr = np.asarray(q)
    res = 0.0
    for i in range(pArr.size):
        res += -(pArr[i] * math.log2(qArr[i]))
    return res

def entropy(p):
    pArr = np.asarray(p)
    res = 0.0
    for i in range(pArr.size):
        if pArr[i] > 0.0001:
            res += -(pArr[i] * math.log2(pArr[i]))
    return res

def kl(p,q):
    pArr = np.asarray(p)
    qArr = np.asarray(q)
    res = 0.0
    for i in range(pArr.size):
        if qArr[i] > 0.0001:
            res += (pArr[i] * math.log2(pArr[i]/qArr[i]))
    return res

def testTwoDistributions(x, y):
    print("=========REPORT START================")
    if isProbabilityMassFunction(x):
        print("X=", x, " is probability mass function")
    else:
        print("X=", x, " is not a probability mass function")
        sys.exit(-1)
      
    if isProbabilityMassFunction(y):
        print("Y=", y, " is probability mass function")
    else:
        print("Y=", y, " is not a probability mass function")
        sys.exit(-1)
    
    if x.shape != y.shape:
        print("X and Y has different shapes")
        sys.exit(-1)

    print("CE(x,y) is ", ce(x,y))
    print("KL(x||y) is ", kl(x,y))
    print("Entropy(x) is ", entropy(x))
    print("KL(x||y) + Entropy(x) should be equal to CE(x,y) ", kl(x,y) + entropy(x))
    print("=========REPORT END================")
    print("")

if __name__ == "__main__":
    #========================================================
    x = np.array([0.2,0.2,0.2,0.2,0.2])
    y = np.array([0.2,0.2,0.2,0.2,0.2])
    testTwoDistributions(x, y)

    x = np.array([0.5,0.5])
    y = np.array([0.5,0.5])
    testTwoDistributions(x, y)
    #========================================================
        