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
bootReps = 1000
go = cProbReps
rawBootData = np.zeros((cProbReps, bootReps))
summaryBootData = np.zeros((cProbReps, 3))
expNb = 2 * ploidy * pi * sigma ** 2 * density
iii = 0
# Simulate a number of data sets and estimate Nb using ML
# and confidence intervals using bootstrap
while go:
    # step 1: Generate Data Set
    data = acml.gen_data(fhat, nSamples, sigma, density)
    acml.set_data(data, dc, 100)
    # Calculate Nb and CI estimates
    boot = acml.bootstrap_CI(bootReps, alpha, sigma, density)
    if not boot:
        print "FAIL"
        continue
    rawBootData[iii] = np.array(boot[3])  # add list of raw bootstap values
    summaryBootData[iii] = np.array(boot[0:3])
    iii += 1
    go += -1

# write data to file
np.savetxt("rawBootData.txt", rawBootData, delimiter=",")
np.savetxt("BootDataSummary.txt", summaryBootData, delimiter=",")

# Calculate Proportion of replicates that contain the real Nb size
count = 0
for rep in summaryBootData:
    if rep[0] <= expNb and expNb <= rep[2]:
        count += 1
p = count / float(cProbReps)
f = open("covProb.txt", 'w')
out = str("sigma: {}\ndensity: {}\nmu: {}\nfhat: {}\nploidy: {}\n"
          "alpha: {}\n nDC: {}\nnSamples: {}\ncovProbReps: {}\n"
          "bootReps: {}\nexpNb: {}\nCovProb: {}").format(sigma, density,
                                                         mu, fhat, ploidy,
                                                         alpha, ndc, nSamples,
                                                         cProbReps, bootReps,
                                                         expNb, p)
f.write(out)
f.close()
