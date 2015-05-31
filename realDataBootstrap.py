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
fhat = 0.3
mu = 0.0001
ploidy = 1.0
alpha = 0.05
ndc = 50
data = np.zeros(ndc)
dc = np.array([i + 1 for i in xrange(ndc)])
sz = np.array([100 for i in xrange(ndc - 1)])
sz = np.append(sz, 50)
acml = ApxCML(mu, sigma, density, data, dc, sz, ploidy, 30)
infn = str(sys.argv[2])
bootReps = int(sys.argv[3])
count = 0  # count number of times CI includes true value
expNb = 2 * ploidy * pi * sigma ** 2 * density
cProbReps = 0
# data files
file("rawRealBootData" + fn + ".txt", 'w').close()
file("summaryRealBootData" + fn + ".txt", 'w').close()
rawOut = file("rawRealBootData" + fn + ".txt", 'a')
sumOut = file("summaryRealBootData" + fn + ".txt", 'a')
cprobOut = file("covRealProb" + fn + ".txt", 'w')
# Simulate a number of data sets and estimate Nb using ML
dataReps = np.array(np.genfromtxt(infn, delimiter=",", dtype=float))
for i, r in enumerate(dataReps):
    # set data
    d, sz = acml.raw_to_dc(r)
    acml.set_data(d, dc, sz)
    if acml.fhat < 0.27 or acml.fhat > 0.33:
        continue
    # estimate nb size and bootstrap CI, returns false if fails to converge
    boot = acml.bootstrap_CI(bootReps, alpha, sigma, density)
    if boot is False:
        "BOOTSTRAP FAIL ON LINE:", i
        continue
    cProbReps += 1
    rawBootData = np.array(boot[6])
    summaryBootData = np.array(boot[0:6])
    # check in CI includes real Nb size
    if summaryBootData[0] <= expNb and expNb <= summaryBootData[2]:
        count += 1
    # append to file
    for v in rawBootData:
        s = str("{:d},{:.4f}\n".format(i, v))
        rawOut.write(s)
    s = str("{:d},{:.4f},{:.4f},"
            "{:.4f},{:.4f},{:.4f},{:.4f}\n").format(i,
                                                    summaryBootData[0],
                                                    summaryBootData[1],
                                                    summaryBootData[2],
                                                    summaryBootData[3],
                                                    summaryBootData[4],
                                                    summaryBootData[5])
    sumOut.write(s)
    #np.savetxt(rawOut, rawBootData, delimiter=',', newline="\n", fmt='%.4f')
    # np.savetxt(
    # sumOut, summaryBootData, delimiter=',', newline="\n", fmt='%.4f')

# calculate proportion of reps contain real value
prob = count
out = str("sigma: {}\ndensity: {}\nmu: {}\nfhat: {}\nploidy: {}\n"
          "alpha: {}\nnDC: {}\ncovProbReps: {}\n"
          "bootReps: {}\nexpNb: {}\n"
          "CovProb: {}").format(sigma, density, mu, fhat, ploidy,
                                alpha, ndc, cProbReps,
                                bootReps, expNb, prob)
cprobOut.write(out)


rawOut.close()
sumOut.close()
cprobOut.close()
end = time.time()
print end - start
