import sys
from math import *
import numpy as np
import sympy.mpmath as sy
import scipy.misc as fac
import scipy.special as sp
from scipy.optimize import minimize
import matplotlib.pyplot as plt


plt.style.use('fivethirtyeight')


class CML:

    def __init__(self, mu, sigma, density, dataIn,
                 distIn, szIn, ploidy, approx=True):
        self.k = ploidy
        self.mu = mu
        self.mu2 = -2.0 * self.mu
        self.s = sigma
        self.ss = sigma * sigma
        self.de = density
        self.z = exp(self.mu2)
        self.sqrz = sqrt(1 - self.z)
        self.g0 = log(1 / self.sqrz)

        # Calling set_data method will initialize these values
        self.ndc = 0
        self.tsz = 0
        self.fhat = 0
        self.dist = np.empty(0, dtype=float)
        self.dist2 = np.empty(0, dtype=float)
        self.data = np.empty(0, dtype=int)
        self.sz = np.empty(0, dtype=int)
        self.set_data(dataIn, distIn, szIn)
        self.ml = np.zeros(2, dtype=float)
        self.approx = approx

        # switch between modeling the full model or approximated model
        if approx:
            self.update = self.apx_update
            self.likelihood = self.apx_likelihood
            self.bootstrap_helper = self.apx_bootstrap_helper
            self.gen_data = self.apx_gen_data
        else:
            self.update = self.full_update
            self.likelihood = self.full_likelihood
            self.bootstrap_helper = self.full_bootstrap_helper
            self.gen_data = self.full_gen_data

    def apx_update(self, ar):
        self.s = ar[0]
        self.de = ar[1]
        # print "arguments", ar
        if(ar[0] <= 0 or ar[1] <= 0):
            print "negative arguments"
            return sys.maxint
        self.ss = self.s * self.s
        for i in xrange(self.ndc):
            if self.dist[i] > 6 * self.s:
                self.split = i
                break
        return -self.likelihood()

    def full_update(self, ar):
        self.s = ar[0]
        self.de = ar[1]
        self.ss = self.s * self.s
        if(ar[0] <= 0 or ar[1] <= 0):
            print "negative arguments"
            return sys.maxint
        return -self.likelihood()

    def apx_likelihood(self):
        phi = np.zeros((self.ndc))
        phi_bar = 0
        denom = 2 * self.k * self.ss * pi * self.de + self.g0
        p = 0
        for s in xrange(self.split):
            if self.dist[s] == 0:
                p = self.g0 / float(denom)
            else:
                p = self.t_series(self.dist[s]) / float(denom)
            phi_bar += p * self.sz[s]
            phi[s] = p
        for l in xrange(self.split, self.ndc):
            p = self.bessel(self.dist[l]) / denom
            phi_bar += p * self.sz[l]
            phi[l] = p
        phi_bar /= self.tsz
        cml = 0
        for i in xrange(self.ndc):
            r = (phi[i] - phi_bar) / (1.0 - phi_bar)
            pIBD = self.fhat + (1 - self.fhat) * r
            if pIBD <= 0:
                print "WARNING: Prabability of IBD has fallen \
                       below zero for distance class", self.dist[i], "."
                print "This marginal likelihood will not be \
                       included in composite likelihood."
                continue
            cml += self.data[i] * log(pIBD) + \
                (self.sz[i] - self.data[i]) * log(1 - pIBD)
        return cml

    def full_likelihood(self):
        phi = np.zeros((self.ndc))
        phi_bar = 0
        denom = 2 * self.k * self.ss * pi * self.de + self.g0
        for i, d in enumerate(self.dist2):
            if d == 0:
                p = self.g0 / denom
            else:
                p = sy.nsum(lambda t: exp(self.mu2 * t) *
                            exp(-d / (4.0 * self.ss * t)) / (2.0 * t), [
                            1, sy.inf], error=False, verbose=False,
                            method='euler-maclaurin', steps=[100])
                p = p / denom
            phi_bar += p * self.sz[i]
            phi[i] = p
        phi_bar /= float(self.tsz)
        cml = 0
        for i in xrange(self.ndc):
            r = (phi[i] - phi_bar) / (1.0 - phi_bar)
            pIBD = self.fhat + (1 - self.fhat) * r
            if pIBD <= 0:
                print "WARNING: Prabability of IBD has fallen \
                       below zero for distance class", self.dist[i], "."
                print "This marginal likelihood will not be \
                       included in composite likelihood."
                continue
            cml += self.data[i] * log(pIBD) + \
                (self.sz[i] - self.data[i]) * log(1 - pIBD)
        return cml

    def apx_bootstrap_helper(self, stat, samples, dClass,
                             sz, sigma, density, verbose):
        fail = 0
        for i, (r, d, s) in enumerate(zip(samples, dClass, sz)):
            # approximate model requires the distance classes to be sorted
            # to find the appropriate cutoff point
            self.sort_data(r, d, s)
            x = self.max_likelihood(sigma, density)
            if x.success is False:
                fail += 1
                continue
            stat[np.where(stat == 0)[0][0]] = self.get_nb()
        return stat, fail

    def full_bootstrap_helper(self, stat, samples, dClass,
                              sz, sigma, density, verbose):
        fail = 0
        for i, (r, d, s) in enumerate(zip(samples, dClass, sz)):
            self.data = r
            self.dist = d
            self.sz = s
            x = self.max_likelihood(sigma, density)
            if x.success is False:
                fail += 1
                continue
            stat[np.where(stat == 0)[0][0]] = self.get_nb()
        return stat, fail

    def bootstrap_CI(self, nSamples, alpha, sigma, density, verbose=False):
        # remember original data
        org_data = self.data
        org_dc = self.dist
        org_sz = self.sz
        self.max_likelihood(sigma, density, verbose=verbose)
        org_nb = self.get_nb()
        # make indexes for sampling with replacement
        # carry over the distance classes as well
        n = len(self.data)
        idx = np.random.randint(0, n, (nSamples, n))
        samples = org_data[idx]
        dClass = org_dc[idx]
        sz = org_sz[idx]
        stat = np.zeros(nSamples)
        stat, fail = self.bootstrap_helper(
            stat, samples, dClass, sz, sigma, density, verbose)
        # Redo those that failed to converge
        while fail > 0:
            idx = np.random.randint(0, n, (fail, n))
            samples = org_data[idx]
            dClass = org_dc[idx]
            sz = org_sz[idx]
            stat, fail = self.bootstrap_helper(
                stat, samples, dClass, sz, sigma, density, verbose)
        stat.sort()
        # return data to original values
        self.data = org_data
        self.dist = org_dc
        return [stat[int((alpha / 2.0) * nSamples)],
                org_nb, stat[int((1 - alpha / 2.0) * nSamples)], stat]

    def landscape_plot(self, res=0.1, sigLow=0.1, sigUp=4.0,
                       denLow=0.1, denUp=6.0, fileName=None):
        # run after max_likelihood
        sig = np.arange(sigLow, sigUp, res)
        den = np.arange(denLow, denUp, res)
        X, Y = np.meshgrid(sig, den)
        Z = np.array([[-log(self.update(np.array([i, j])))
                       for i in sig] for j in den])
        plt.rcParams['xtick.direction'] = 'out'
        plt.rcParams['ytick.direction'] = 'out'
        plt.figure(figsize=(10, 10))
        CS = plt.contour(X, Y, Z)
        plt.clabel(CS, inline=1, fontsize=10)
        plt.title('Likelihood Landscape of $\sigma$ vs Density')
        plt.xlabel('$\sigma$')
        plt.ylabel('density')
        plt.plot(self.ml[0], self.ml[1], "*k", markersize=20)
        plt.show()
        if fileName is not None:
            plt.savefig(fileName, format='pdf')

    def max_likelihood(self, startSig, startDen, max_iter=10000,
                       tol=0.00001, verbose=False):
        start = np.array([startSig, startDen])
        bnds = ((2 ** (-52), None), (2 ** (-52), None))
        x = minimize(self.update, start, options={
                     'maxiter': max_iter, 'disp': verbose},
                     tol=tol, bounds=bnds, method='TNC')
        self.ml = x.x
        return x

    def sort_data(self, data, dc, sz):
        z = zip(dc, data, sz)
        z.sort()
        self.data = np.array([j for i, j, k in z], dtype=int)
        self.dist = np.array([i for i, j, k in z], dtype=float)
        self.sz = np.array([k for i, j, k in z], dtype=int)

    def apx_gen_data(self, fbar, nSamples, sigma, density):
        totalSize = self.ndc * nSamples
        ss = sigma * sigma
        split = self.ndc
        for i in xrange(self.ndc):
            if self.dist[i] > 6 * sigma:
                split = i
                break
        denom = 2.0 * self.k * pi * ss * density + self.g0
        phi = np.zeros((self.ndc))
        phi_bar = 0
        for s in xrange(split):
            if self.dist[s] == 0:
                p = self.g0 / float(denom)
            else:
                p = self.t_series(self.dist[s]) / float(denom)
            phi_bar += p * nSamples
            phi[s] = p
        for l in xrange(split, self.ndc):
            p = self.bessel(self.dist[l]) / denom
            phi_bar += p * nSamples
            phi[l] = p
        phi_bar /= float(totalSize)
        r = (phi - phi_bar) / (1.0 - phi_bar)
        pIBD = fbar + (1.0 - fbar) * r
        pIBD = np.array(pIBD, dtype=float)
        # simulate values from binomial distribution
        # np.random.seed(1209840)
        counts = np.random.binomial(nSamples, pIBD)
        return counts

    def full_gen_data(self, fbar, nSamples, sigma, density):
        totalSize = self.ndc * nSamples
        ss = sigma * sigma
        denom = 2.0 * self.k * pi * ss * density + self.g0
        phi = np.zeros((self.ndc))

        phi_bar = 0
        for i, d in enumerate(self.dist2):
            if d == 0:
                p = self.g0 / denom
            else:
                p = sy.nsum(lambda t: exp(self.mu2 * t) *
                            exp(-d / (4.0 * ss * t)) / (2.0 * t), [
                            1, sy.inf], error=False, verbose=False,
                            method='euler-maclaurin', steps=[1000])
                p = p / denom
            phi_bar += p * nSamples
            phi[i] = p

        phi_bar /= float(totalSize)
        r = (phi - phi_bar) / (1.0 - phi_bar)
        pIBD = fbar + (1.0 - fbar) * r
        pIBD = np.array(pIBD, dtype=float)
        # simulate values from binomial distribution
        # np.random.seed(1209840)
        counts = np.random.binomial(nSamples, pIBD)
        return counts

    def set_data(self, newData, dc, sz):
        if type(sz) is int:
            sz = [sz for i in xrange(len(newData))]
        if len(newData) == len(dc) and len(dc) == len(sz):
            self.sort_data(newData, dc, sz)
            self.dist2 = self.dist ** 2
            self.tsz = np.sum(self.sz)
            self.ndc = len(self.dist)
            self.fhat = np.sum(self.data) / float(self.tsz)
        else:
            raise Exception(
                "ERROR: data and distance class arrays are not equal length")

    def get_nb(self):
        s = self.ml[0]
        d = self.ml[1]
        return 2 * self.k * pi * s * s * d


class ApxCML(CML):

    def __init__(self, mu, sigma, density, dataIn,
                 distIn, szIn, ploidy, n_terms):
        CML.__init__(
            self, mu, sigma, density, dataIn, distIn, szIn, ploidy, True)
        self.split = self.ndc
        self.n_t = n_terms
        self.plog = np.array([sy.polylog(i + 1, self.z)
                              for i in range(self.n_t)])

    def t_series(self, x):
        sum = 0.0
        pow2 = 1
        for t in xrange(self.n_t):
            dt = 2 * t
            pow2 <<= 1
            powX = 1.0
            powS = 1.0
            for i in xrange(dt):
                powX *= x
                powS *= self.s
            s = (self.plog[t] * powX) / \
                (fac.factorial2(dt, exact=True) * pow2 * powS)
            if((t % 2) == 0):
                sum += s
            else:
                sum -= s
        return sum

    def bessel(self, x):
        t = (x / float(self.s)) * self.sqrz
        if(t < 650):
            return sp.k0(t)
        else:
            return 0


class FullCML(CML):

    def __init__(self, mu, sigma, density, dataIn, distIn, szIn, ploidy):
        CML.__init__(
            self, mu, sigma, density, dataIn, distIn, szIn, ploidy, False)
