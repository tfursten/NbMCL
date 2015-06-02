#!/usr/bin/env python
import sys
from math import *
import time
import numpy as np
from CML import *

start = time.time()
fn = "{:0>4d}".format(int(sys.argv[1]))
bootReps = int(sys.argv[2])
nReps = int(sys.argv[3])
sigma = float(sys.argv[4])
density = float(sys.argv[5])
fbar = float(sys.argv[6])
out = open("correction" + fn + ".txt", 'w')
mu = 0.0001
ploidy = 1.0
alpha = 0.05
ndc = 50
data = np.zeros(ndc)
dc = np.array([i + 1 for i in xrange(ndc)])
sz = np.array([100 for i in xrange(ndc - 1)])
sz = np.append(sz, 50)
acml = ApxCML(mu, sigma, density, data, dc, sz, ploidy, 30)
dataReps = acml.gen_data(fbar, sz, sigma, density, nReps)
U = np.zeros(nReps)
L = np.zeros(nReps)

for i, r in enumerate(dataReps):
    acml.set_data(r, dc, sz)
    boot = acml.bootstrap_CI(bootReps, alpha, sigma, density)
    while not boot:
        print fn, "BOOTSTRAP FAIL AT LINE:", i
        x = acml.gen_data(fbar, sz, sigma, density, 1)
        dataReps[i] = x
        acml.set_data(x[0], dc, sz)
        boot = acml.bootstrap_CI(bootReps, alpha, sigma, density)
    print boot
    L[i] = boot[0]
    U[i] = boot[2]
for ll, uu in zip(L, U):
    s = str("{:.4f},{:.4f}\n".format(ll, uu))
    out.write(s)

out.close()
end = time.time()
print end - start
