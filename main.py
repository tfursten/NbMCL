#!/usr/bin/env python
from math import *
import numpy as np
from CML import *


data = np.zeros(50)
sigma = 1.0
density = 1.0
mu = 0.0001
fhat = 0.3
ploidy = 1
alpha = 0.05
ndc = 50
nSamples = 100
dc = np.array([i + 1 for i in xrange(ndc)])
sz = np.array([nSamples for i in xrange(ndc)])
acml = ApxCML(mu, sigma, density, data, dc, sz, ploidy, 30)
cProbReps = 1  # number of replicates for coverage probability
bootReps = 10
go = cProbReps
count = 0  # count number of times CI includes true value
expNb = 2 * ploidy * pi * sigma ** 2 * density

# data files
file("rawBootData.txt", 'w').close()
file("summaryBootData.txt", 'w').close()
rawOut = file("rawBootData.txt", 'a')
sumOut = file("summaryBootData.txt", 'a')
dataOut = file("generatedData.txt", 'w')
cprobOut = file("covProb.txt", 'w')
# Simulate a number of data sets and estimate Nb using ML
dataReps = acml.gen_data(fhat, nSamples, sigma, density, cProbReps)
print dataReps
for i, r in enumerate(dataReps):
    print i
    # set data
    acml.set_data(r, dc, 100)
    # estimate nb size and bootstrap CI, returns false if fails to converge
    boot = acml.bootstrap_CI(bootReps, alpha, sigma, density)
    while not boot:
        print "Main: FAIL BOOTSTRAP on Prob rep:", i
        # repeat if bootstrap fails
        x = acml.gen_data(fhat, nSamples, sigma, density, 1)
        dataReps[i] = x
        acml.set_data(x[0], dc, 100)
        boot = acml.bootstrap_CI(bootReps, alpha, sigma, density)
    rawBootData = np.array(boot[3])
    summaryBootData = np.array(boot[0:3])
    print summaryBootData
    # check in CI includes real Nb size
    if summaryBootData[0] <= expNb and expNb <= summaryBootData[2]:
        count += 1
    # append to file
    np.savetxt(rawOut, rawBootData, delimiter=',', newline=" ", fmt='%.4f')
    np.savetxt(sumOut, summaryBootData, delimiter=',', newline=" ", fmt='%.4f')
    print i, "done"
# calculate proportion of reps contain real value
prob = count / float(cProbReps)
out = str("sigma: {}\ndensity: {}\nmu: {}\nfhat: {}\nploidy: {}\n"
          "alpha: {}\nnDC: {}\nnSamples: {}\ncovProbReps: {}\n"
          "bootReps: {}\nexpNb: {}\n"
          "CovProb: {}").format(sigma, density, mu, fhat, ploidy,
                                alpha, ndc, nSamples, cProbReps,
                                bootReps, expNb, prob)
cprobOut.write(out)

np.savetxt(dataOut, dataReps, delimiter=',', fmt='%02d')
rawOut.close()
sumOut.close()
dataOut.close()
cprobOut.close()
