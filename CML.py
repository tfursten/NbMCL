import numpy as np
from math import *
import sympy.mpmath as sy
import scipy.misc as fac
import scipy.special as sp
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import sys
plt.style.use('fivethirtyeight')


class CML:
    def __init__(self, mu, sigma, density, dataIn, distIn, szIn, ploidy):
        self.k = ploidy
        self.mu = mu
        self.mu2 = -2.0*self.mu
        self.s = sigma
        self.ss = sigma*sigma
        self.de = density
        self.z = exp(self.mu2)
        self.sqrz = sqrt(1-self.z)
        self.g0 = log(1/self.sqrz)
        #self.infile = open(data_file,'r')
        #x,y,z = np.array(np.genfromtxt(self.infile,usecols=(0,1,2),dtype=float,unpack=True))
        
        #Calling changeData method will initialize these values
        self.ndc = 0
        self.tsz = 0
        self.fhat = 0
        self.dist = np.empty(0,dtype=float)
        self.dist2 = np.empty(0,dtype=float)
        self.data = np.empty(0,dtype=int)
        self.sz = np.empty(0,dtype=int)
        self.changeData(dataIn, distIn, szIn)
        self.ml = np.zeros(2,dtype=float)

    def get_nb(self):
        s = self.ml[0]
        d = self.ml[1]
        return 2*self.k*pi*s*s*d
    

class aprxCML(CML):
    def __init__(self, mu, sigma, density, dataIn, distIn, szIn, ploidy, n_terms):
        CML.__init__(self, mu, sigma, density, dataIn, distIn, szIn, ploidy)
        self.split = self.ndc
        self.n_t = n_terms
        self.plog = np.array([sy.polylog(i+1,self.z) for i in range(self.n_t)])
    
    def update(self, ar):
        self.s = ar[0]
        self.de = ar[1]
        #print "arguments", ar
        if(ar[0]<=0 or ar[1]<=0):
            print "negative arguments"
            return sys.maxint
        self.ss = self.s*self.s
        for i in xrange(self.ndc):
            if self.dist[i] > 6*self.s:
                self.split = i
                break
        return -self.likelihood()
    
    
    def maxLike(self, startSig, startDen, max_iter=10000, tol=0.00001, verbose=False):
        start = np.array([startSig,startDen])
        bnds = ((2**(-52),None),(2**(-52),None))
        x = minimize(self.update,start,options={'maxiter':max_iter,'disp':verbose},tol=tol,bounds=bnds,method='TNC')
        self.ml = x.x
        print self.ml
        return x
    
    def bootstrap_helper(self,stat,samples,d_class,sz,sigma,density,verbose):
        fail = 0
        for i,(r,d,s) in enumerate(zip(samples,d_class,sz)):
            #approximate model requires the distance classes to be sorted
            #to find the appropriate cutoff point
            self.sortData(r,d,s)
            x = self.maxLike(sigma,density)
            if x.success == False:
                fail += 1
                continue
            stat[np.where(stat==0)[0][0]] = self.get_nb()
        return stat,fail
            
    
    def bootstrap_CI(self, num_samples, alpha, sigma, density, verbose=False):
        #remember original data
        org_data = self.data
        org_dc = self.dist
        org_sz = self.sz
        self.maxLike(sigma,density,verbose=verbose)
        org_nb = self.get_nb()
        #make indexes for sampling with replacement 
        #carry over the distance classes as well
        n = len(self.data)
        idx = np.random.randint(0,n,(num_samples,n))
        samples = org_data[idx]
        d_class = org_dc[idx]
        sz = org_sz[idx]
        stat = np.zeros(num_samples)
        stat,fail = self.bootstrap_helper(stat,samples,d_class,sz,sigma,density,verbose)
        #Redo those that failed to converge
        while fail > 0:
            idx = np.random.randint(0,n,(fail,n))
            samples = org_data[idx]
            d_class = org_dc[idx]
            sz = org_sz[idx]
            stat,fail = self.bootstrap_helper(stat,samples,d_class,sz,sigma,density,verbose)
        stat.sort()
        #return data to original values
        self.data = org_data
        self.dist = org_dc
        return [stat[int((alpha/2.0)*num_samples)],org_nb,stat[int((1-alpha/2.0)*num_samples)],stat]
    
    
    def t_series(self, x):
        sum = 0.0
        pow2 = 1
        for t in xrange(self.n_t):
            dt = 2*t
            pow2 <<= 1
            powX = 1.0
            powS = 1.0
            for i in xrange(dt):
                powX *= x
                powS *= self.s
            s = (self.plog[t]*powX)/(fac.factorial2(dt,exact=True)*pow2*powS)
            if((t%2)==0):
                sum += s
            else:
                sum -= s
        return sum
    
    def bessel(self, x):
        t = (x/float(self.s))*self.sqrz
        if(t<650):
            return sp.k0(t)
        else: 
            return 0
    
    def likelihood(self):
        phi = np.zeros((self.ndc))
        phi_bar = 0
        denom = 2*self.k*self.ss*pi*self.de+self.g0
        p=0
        for s in xrange(self.split):
            if self.dist[s]==0:
                p = self.g0/float(denom)
            else:
                p = self.t_series(self.dist[s])/float(denom)
            phi_bar += p*self.sz[s]
            phi[s] = p
        for l in xrange(self.split,self.ndc):
            p = self.bessel(self.dist[l])/denom
            phi_bar += p*self.sz[l]
            phi[l] = p
        phi_bar /= self.tsz
        cml = 0
        for i in xrange(self.ndc):
            r = (phi[i] - phi_bar)/(1.0 - phi_bar)
            pIBD = self.fhat + (1-self.fhat) * r
            if pIBD<=0:
                print "WARNING: Prabability of IBD has fallen below zero for distance class",self.dist[i],"."
                print "This marginal likelihood will not be included in composite likelihood."
                continue
            cml += self.data[i]*log(pIBD)+(self.sz[i]-self.data[i])*log(1-pIBD)
        return cml

    def changeData(self, newData, dc, sz):
        if type(sz) is int:
            sz = [sz for i in xrange(len(newData))]
        if len(newData)==len(dc) and len(dc) == len(sz):
            self.sortData(newData,dc,sz)
            self.dist2 = self.dist**2
            self.tsz = np.sum(self.sz)
            self.ndc = len(self.dist)
            self.fhat = np.sum(self.data)/float(self.tsz)
        else:
            raise Exception("Error: Data and Distance class arrays are not equal length")
    


    def sortData(self,data,dc,sz):
    	z = zip(dc,data,sz)
    	z.sort()
    	self.data = np.array([j for i,j,k in z], dtype=int)
    	self.dist = np.array([i for i,j,k in z], dtype=float)
    	self.sz = np.array([k for i,j,k in z], dtype=int)
        
    def genData(self, fbar, nSamples, sigma, density):
        totalSize = self.ndc*nSamples
        ss = sigma*sigma
        split = self.ndc
        for i in xrange(self.ndc):
            if self.dist[i] > 6*sigma:
                split = i
                break
        denom = 2.0*self.k*pi*ss*density + self.g0
        phi = np.zeros((self.ndc))
        phi_bar = 0
        for s in xrange(split):
            if self.dist[s] == 0:
                p = self.g0/float(denom)
            else:
                p = self.t_series(self.dist[s])/float(denom)
            phi_bar += p*nSamples
            phi[s] = p
        for l in xrange(split,self.ndc):
            p = self.bessel(self.dist[l])/denom
            phi_bar += p*nSamples
            phi[l] = p
        phi_bar /= float(totalSize)
        r = (phi - phi_bar)/(1.0 - phi_bar)
        pIBD = fbar + (1.0-fbar) * r
        pIBD = np.array(pIBD,dtype=float) 
        #simulate values from binomial distribution
        #np.random.seed(1209840)
        counts = np.random.binomial(nSamples,pIBD)
        return counts
    
class fullCML(CML):
    def __init__(self, mu, sigma, density, dataIn, distIn, szIn, ploidy):
        CML.__init__(self, mu, sigma, density, dataIn, distIn, szIn, ploidy)
    
    def likelihood(self):
        phi = np.zeros((self.ndc))
        phi_bar = 0
        denom = 2*self.k*self.ss*pi*self.de+self.g0
        for i,d in enumerate(self.dist2):
            if d == 0:
                p = self.g0/denom
            else:
                p = sy.nsum(lambda t: exp(self.mu2*t)*exp(-d/(4.0*self.ss*t))/(2.0*t), [1,sy.inf],error=False,verbose=False,method='euler-maclaurin',steps=[100])
                p = p/denom
            phi_bar += p*self.sz[i]
            phi[i] = p        
        phi_bar /= float(self.tsz)
        cml = 0
        for i in xrange(self.ndc):
            r = (phi[i] - phi_bar)/(1.0 - phi_bar)
            pIBD = self.fhat + (1-self.fhat) * r
            cml += self.data[i]*log(pIBD)+(self.sz[i]-self.data[i])*log(1-pIBD)
        return cml

    def update(self, ar):
        self.s = ar[0]
        self.de = ar[1]
        self.ss = self.s*self.s
        return -self.likelihood()
    
    def maxLike(self, startSig, startDen, max_iter=10000, tol=0.00001, verbose=False):
        start = np.array([startSig,startDen])
        bnds = ((2**(-52),None),(2**(-52),None))
        x = minimize(self.update,start,options={'maxiter':max_iter,'disp':verbose},tol=tol,bounds=bnds,method='TNC')
        self.ml = x
        return x
    
    def bootstrap_CI(self, num_samples, alpha, sigma, density):
        #remember original data
        org_data = self.data
        org_dc = self.dist
        self.maxLike(sigma,density)
        org_nb = self.get_nb()
        #make indexes for sampling with replacement 
        #carry over the distance classes as well
        n = len(self.data)
        idx = np.random.randint(0,n,(num_samples,n))
        samples = org_data[idx]
        d_class = org_dc[idx]
        stat = np.zeros(num_samples)
        for i,(r,d) in enumerate(zip(samples,d_class)):
            self.data = r
            self.dist = d
            self.maxLike(sigma,density)
            stat[i] = self.get_nb()
        stat.sort()
        #return data to original values
        self.data = org_data
        self.dist = org_dc
        return [stat[int((alpha/2.0)*num_samples)],org_nb,stat[int((1-alpha/2.0)*num_samples)],stat]
    
    def changeData(self, newData):
        self.data = newData
        #x,y,z = np.arr
        self.fhat = np.sum(self.data)/float(np.sum(self.sz))
    
    def genData(self, fbar, nSamples, sigma, density):
        totalSize = self.ndc*nSamples
        ss = sigma*sigma        
        denom = 2.0*self.k*pi*ss*density + self.g0
        phi = np.zeros((self.ndc))
        
        phi_bar = 0
        for i,d in enumerate(self.dist2):
            if d==0:
                p = self.g0/denom
            else:
                p = sy.nsum(lambda t: exp(self.mu2*t)*exp(-d/(4.0*ss*t))/(2.0*t), [1,sy.inf],error=False,verbose=False,method='euler-maclaurin',steps=[100])
                p = p/denom
            phi_bar += p*nSamples
            phi[i] = p
    
        phi_bar /= float(totalSize)
        r = (phi - phi_bar)/(1.0 - phi_bar)
        pIBD = fbar + (1.0-fbar) * r
        pIBD = np.array(pIBD,dtype=float) 
        #simulate values from binomial distribution
        #np.random.seed(1209840)
        counts = np.random.binomial(nSamples,pIBD)
        return counts

