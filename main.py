from CML import *
import numpy as np

#Make object using approximated model likelihood
data = np.zeros(50)
sigma = 1.0
density = 1.0
dc = np.array([i+1 for i in xrange(50)])
sz = np.array([100 for i in xrange(50)])
acml = aprxCML(0.0001, sigma, density, data, dc, sz, 1, 30)
data = acml.genData(0.3,100,sigma, density)
acml.changeData(data,dc,100)
print acml.data
print acml.dist
print acml.sz
ml = acml.maxLike(sigma,density)
print(acml.get_nb())
print(ml)
print 2*1*pi*sigma*sigma*density

boot = acml.bootstrap_CI(5,0.05,sigma,density,verbose=True)
print boot