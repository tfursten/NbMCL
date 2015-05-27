#!/usr/bin/env python
import sys
from math import *
import time
import numpy as np
from CML import *


fn = "{:0>4d}".format(int(sys.argv[1]))
start = time.time()

sigma = 1.0
density = 1.0
mu = 0.0001
fhat = 0.3
ploidy = 1
alpha = 0.05
ndc = 51
nSamples = 100
data = np.zeros(ndc)
dc = np.array([i+1 for i in xrange(ndc)])
sz = np.array([nSamples for i in xrange(ndc)])
acml = ApxCML(mu, sigma, density, data, dc, sz, ploidy, 30)
infn = str(sys.argv[3])  # infile name
count = 0  # count number of times CI includes true value
expNb = 2 * ploidy * pi * sigma ** 2 * density

# data files
file("rawJackData" + fn + ".txt", 'w').close()
file("summaryJackData" + fn + ".txt", 'w').close()
rawOut = file("rawJackData" + fn + ".txt", 'a')
sumOut = file("summaryJackData" + fn + ".txt", 'a')
cprobOut = file("covProbJack" + fn + ".txt", 'w')

dataReps = np.array(np.genfromtxt(infn, delimiter=",", dtype=float))
cProbReps = len(dataReps)
for i, row in enumerate(dataReps):
    # estimate nb size and jackknife CI
    jack = acml.jackknife_CI(row, alpha, sigma, density)
    rawJackData = np.array(jack[3])
    summaryJackData = np.array(jack[0:3])
    # check in CI includes real Nb size
    if summaryJackData[0] <= expNb and expNb <= summaryJackData[2]:
        count += 1
    # append to file
    for v in rawJackData:
        s = str("{:d},{:.4f}\n".format(i, v))
        rawOut.write(s)
    s = str("{:d},{:.4f},{:.4f},{:.4f}\n").format(i,
                                                  summaryJackData[0],
                                                  summaryJackData[1],
                                                  summaryJackData[2])
    sumOut.write(s)

# calculate proportion of reps contain real value
prob = count / float(cProbReps)
out = str("sigma: {}\ndensity: {}\nmu: {}\nfhat: {}\nploidy: {}\n"
          "alpha: {}\nnDC: {}\nnSamples: {}\ncovProbReps: {}\n"
          "expNb: {}\n"
          "CovProb: {}").format(sigma, density, mu, fhat, ploidy,
                                alpha, ndc, nSamples, cProbReps,
                                expNb, prob)
cprobOut.write(out)


rawOut.close()
sumOut.close()
cprobOut.close()
end = time.time()
print end - start
