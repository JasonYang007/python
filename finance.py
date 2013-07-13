import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.stats import norm, gamma
from mymodule import *
from itertools import *

class Brownian(object):
    def __init__(self,Nsteps,Npaths,dT):
        self.Nsteps=Nsteps
        self.Npaths=Npaths
        self.dT=dT
        self.ti=np.cumsum(np.ones(Nsteps)*dT)
        self.dWt=np.random.randn(Nsteps,Npaths)*np.sqrt(dT); self.dWt[0,:]=0
        self.Wt=np.cumsum(self.dWt, axis=0)
    def get_tidx(self, *Targs):
        """ Targs must be in increasing order for time
        """
        res=[]
        for T in Targs:
            res.append(len(list(takewhile(lambda x: x <= T, self.ti[len(res):])))-1)
        if len(res)==1:
            return res[0]
        else:
            return tuple(res)
        
        
class SpotRate(object):
    def __init__(self, r0, alpha, theta, sigma, Brownian=None):
        self.r0=r0
        self.alpha=alpha
        self.theta=theta
        self.sigma=sigma
        self.BM=Brownian
        if Brownian is not None:
           self.set_BM(Brownian)
    def set_BM(self, Brownian):
        self.rt_avg, self.rt_stdev=[self.r0],[0.]
        self.rt=self.BM.dWt
        self.rt[0,:]=self.r0
        for tt in xrange(1,self.BM.Nsteps):
            self.rt[tt,:]=self.rt[tt-1,:]+(self.theta-self.alpha*self.rt[tt-1,:])*self.BM.dT+\
                          self.sigma*self.BM.dWt[tt,:]
            self.rt_avg.append(np.average(self.rt[tt,:]))
            self.rt_stdev.append(np.std(self.rt[tt,:]))
        self.rt_avg_th=np.exp(-self.alpha*self.BM.ti)*self.r0+self.theta/self.alpha*(1-np.exp(-self.alpha*self.BM.ti))
        self.rt_stdev_th=self.sigma*np.sqrt((1-np.exp(-2.*self.alpha*self.BM.ti))/(2.*self.alpha))

class Bond(object):
    def __init__(self, spotRate):
        self.spotRate=spotRate
        lnP=np.copy(spotRate.rt)*self.spotRate.BM.dT
        lnP=np.cumsum(lnP[::-1,:], axis=0)[::-1,:]
        lnP -= (spotRate.rt+spotRate.rt[-1,:])*self.spotRate.BM.dT/2.
        self.P=np.exp(-lnP)
        self.ap=None
    def doPricing(self):
        aP=np.average(self.P,axis=1)
        return aP
    def forwardRate(self, T1, T2):
        tidx1, tidx2=self.spotRate.BM.get_tidx(T1,T2)
        ratio=np.average(self.P[tidx2,:])/np.average(self.P[tidx1,:])
        return (ratio-1.0)/(T2-T1)

class CallableBond(Bond):
    def __init__(self, spotRate):
        super(CallableBond, self).__init__(spotRate)

    def doPricing(self, K, Tex):
        idx_ex=self.spotRate.BM.get_tidx(Tex)
        cB=[ (bp if bp <= K else K)*self.P[0][idx_path]/self.P[idx_ex][idx_path] \
                    for idx_path, bp in enumerate(self.P[idx_ex][:]) ]
        Pc=np.average(cB)
        return Pc

def Hull_Whilte_1f():
    """
    dr(t)=(theta(t)-alpha*r(t))dt+sigma*dWt
    """
    dT,     theta,     alpha,     sigma,     r0 = \
    0.1,     0.04,     0.7,     0.009,     0.04
    
    Npaths, Nsteps = 1000, 100
    
    fig=plt.figure()
    Idx_lst=[1,2,4,5,6,7,8,9,3]
    Nplots=len(Idx_lst)
    rr,cc=findRowCol(Nplots)
    axlist=[ fig.add_subplot(rr,cc, idx) for idx in Idx_lst]
    pidx=0
    
    ## brownian motion
    BM=Brownian(Nsteps, Npaths, dT)
    ax=axlist[pidx]; pidx += 1
    ax.plot(BM.ti, BM.Wt)
    ax.set_title("Standard BM")

    # Normality check
    ax=axlist[pidx]; pidx += 1
    normplot(BM.Wt,"-", axes=ax)
    ax.set_title("Normality Check")
    
    ## Spot rate
    spotRate=SpotRate(r0,alpha,theta,sigma,BM)
    ax=axlist[pidx]; pidx += 1
    ax.plot(spotRate.BM.ti, spotRate.rt, "-")
    ax.plot(spotRate.BM.ti, spotRate.rt_avg, "yo-", ms=3, mew=0.1, mec="#ababab")
    ax.set_title("Spot rate: paths & sample average")

    # Spot rate: Avg vs. Theory
    ax=axlist[pidx]; pidx += 1
    alpha,theta,BM,ti=spotRate.alpha,spotRate.theta,spotRate.BM, spotRate.BM.ti
    ax.plot(ti, spotRate.rt_avg, "o", ms=3, mew=0)
    ax.plot(ti, spotRate.rt_avg_th,"-")
    ax.set_title("Average vs. Theory")
    
    # Spot rate: Stdev vs. Theory
    ax=axlist[pidx]; pidx += 1
    ax.plot(BM.ti, spotRate.rt_stdev, "o", ms=3, mew=0)
    ax.plot(BM.ti, spotRate.rt_stdev_th,"-")
    ax.set_title("Stdev vs. Theory")
    
    # Bond price
    bond=Bond(spotRate)
    ti,P,aP=bond.spotRate.BM.ti, bond.P, bond.doPricing()
    ax=axlist[pidx]; pidx += 1
    ax.plot(ti, P, "-")
    ax.plot(ti, aP, "yo", ms=3, mew=0)
    ax.set_title("Bond price")
    ax.set_ylim(0,1.)
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    print "Straight Bond Price={0:.2f}".format(aP[0])

    # Callable bond price
    idx_ex=Nsteps/2; Tex=BM.ti[idx_ex]
    cbond=CallableBond(spotRate)
    ti,P=cbond.spotRate.BM.ti, cbond.P
    Klst=np.linspace(0.1, 0.99, 10)
    Pc_lst=[]
    for K in Klst:
        Pc=cbond.doPricing(K,Tex)
        Pc_lst.append(Pc)
    ax=axlist[pidx]; pidx += 1
    ax.plot(Klst, Pc_lst, "o-", ms=3, mew=0)
    ax.set_xlabel("Strike")
    ax.set_title("Callable bond at Tex={0:.1f}".format(Tex))
    
    # Callable bond price
    idx_ex=int(Nsteps*0.8); Tex=BM.ti[idx_ex]
    Klst=np.linspace(0.1, 0.99, 10)
    Pc_lst=[]
    for K in Klst:
        Pc=cbond.doPricing(K,Tex)
        Pc_lst.append(Pc)
    ax=axlist[pidx]; pidx += 1
    ax.plot(Klst, Pc_lst, "o-", ms=3, mew=0)
    ax.set_xlabel("Strike")
    ax.set_title("Callable bond at Tex={0:.1f}".format(Tex))

    # Forward rates
    fwdTimes=[(t,t+1) for t in np.linspace(1,9,50)]
    fwdRates=[bond.forwardRate(T1,T2) for T1,T2 in fwdTimes]
    ax=axlist[pidx]; pidx += 1
    ax.plot(zip(*fwdTimes)[0], fwdRates, "o-", ms=3, mew=0)
    ax.set_xlabel("Time")
    ax.set_ylabel("Forward Rate")
    ax.set_title("1yr Forward Rate")
    
    ## show the plot    
    plt.show()

def rand_uball(ndim, Nsamples):
    xx=np.random.randn(ndim, Nsamples)
    rr=np.random.randn(Nsamples)**(1.0/ndim)
    nfactor=rr/np.sqrt(np.sum(xx**2, axis=0))
    for ipath in xrange(Nsamples):
        xx[:, ipath] *= nfactor[ipath]
    return xx

def rand_uball_(ndim, Nsamples):
    xx=np.linspace(1.0/(1+Nsamples), Nsamples*1.0/(1+Nsamples), Nsamples)
    nxx=sp.stats.norm.ppf(np.tile(xx, (ndim,1))) 
    for itime in xrange(ndim):
        np.random.shuffle(nxx[itime,:])
    
    rr=np.power(xx, 1.0/ndim)
    np.random.shuffle(rr)

    rad=np.sqrt(np.sum(nxx**2, axis=0))
    for itime in xrange(ndim):
        nxx[itime,:] *= rr/rad
    
    return nxx

################

def rand_uball2(ndim, Nsamples):
    xx=np.tile(np.linspace( -1, 1, Nsamples ), (ndim,1))
    for idx in xrange(Nsamples):
        while True:
            for idim in xrange(ndim):
                np.random.shuffle(xx[idim,idx:])
            break
            if np.sum(xx[:,idx]**2) <= 1.0:
                break
    return xx[:,:Nsamples]

def uball2_test():
    xx=rand_uball2(2,500)
    plt.plot(xx[0],xx[1], "o",ms=3, mew=0)
    plt.show()
                
def zradius(ndim, siglevel=6):
    q=1-2.0*norm.cdf(-siglevel)
    xx=gamma.ppf(q, ndim*0.5)
    zz=np.sqrt(2*xx)
    return zz

def feynman_paths(nsteps, npaths, siglevel=6):
    fig=plt.figure()
    Idx_lst=[1,2,3,4]
    Nplots=len(Idx_lst)
    rr,cc=findRowCol(Nplots)
    axlist=[ fig.add_subplot(rr,cc, idx) for idx in Idx_lst]
    pidx=0
    suptitle_fs=14
    title_fs=13
    label_fs=12
    legend_fs=10
    Nvars=2
    SIGLEVEL=6
 
    ax=axlist[pidx]; pidx += 1
#    dimarr=range(1,7); siglevel=6
#    ax.plot(dimarr, [zradius(ndim,siglevel) for ndim in dimarr], "o-", ms=3, mew=0)
#    ax.set_xlabel("dimension", fontsize=label_fs)
#    ax.set_ylabel("z radius", fontsize=label_fs)
#    ax.set_title("siglevel=%d" % siglevel, fontsize=title_fs)
    siglevels=np.linspace(0.5, SIGLEVEL, 100)
    ax.plot(siglevels, zradius(Nvars, siglevels),"o-", ms=3, mew=0)
    ax.set_xlim(0, SIGLEVEL)
    ax.set_ylim(0, SIGLEVEL)
    ax.plot([0,SIGLEVEL], [0,SIGLEVEL], "--", lw=0.5, color="#ababab" )


    ax=axlist[pidx]; pidx += 1
    NN=1000
    xvars=[ np.random.randn(NN) for i in xrange(Nvars)]
    rr=0
    for xx in xvars:
        rr += xx**2
    rr=np.sqrt(rr)
    
    siglevels=np.linspace(0.5, SIGLEVEL, 100)
    freq=[]
    thfreq=[]
    for slevel in siglevels:
        zz=zradius(Nvars,slevel)
        freq.append( np.sum( rr <= zz )*1.0/NN )
        thfreq.append(1.-2*norm.cdf(-slevel))
    
    ax.plot(freq, thfreq, "o-", ms=3, mew=0)
    ax.set_xlabel("simulated freq", fontsize=label_fs)
    ax.set_ylabel("theoretical freq", fontsize=label_fs)
    ax.set_title("Dim=%d" % Nvars, fontsize=title_fs)
    ax.set_xlim(0.3, 1.0)
    ax.set_ylim(0.3, 1.0)
    
    ax=axlist[pidx]; pidx += 1
    ax.plot(xvars[0], xvars[1], "o", ms=2, mew=0)
    ax.set_xlim(-SIGLEVEL,SIGLEVEL)
    ax.set_ylim(-SIGLEVEL,SIGLEVEL)
    
    ax=axlist[pidx]; pidx += 1
    NN=300
    zRADIUS=zradius(Nvars,SIGLEVEL)
    uxx=rand_uball2(Nvars, NN)*zRADIUS
    ax.plot(uxx[0,:], uxx[1,:], "o", ms=2, mew=0)
    zRADIUS=10
    ax.set_xlim(-zRADIUS,zRADIUS)
    ax.set_ylim(-zRADIUS,zRADIUS)
    
    plt.show()

def exp_test():
    Nsteps=10; Npaths=1000
    eshape=0.8
    nxx=np.random.randn(Nsteps, Npaths)
    BM=np.cumsum(nxx, axis=0)
    
    pidx_lst=[1,2]
    Nplots=len(pidx_lst)
    rr,cc=findRowCol(Nplots)
    fig=plt.figure()
    ax_lst=[ fig.add_subplot(rr,cc, ii) for ii in pidx_lst]
    pidx=0
    
    ax=ax_lst[pidx]; pidx += 1
    ss=np.random.rand(Nsteps, Npaths)
    idx=(ss >= 0.5)
    ss[idx]=1; ss[~idx]=-1
    ee=np.random.exponential(eshape, (Nsteps, Npaths))
    ee *= ss
    EM=np.cumsum(ee, axis=0)
    pp=np.exp(-0.5*np.sum(ee**2,axis=0))/np.exp(-eshape*np.sum(np.abs(ee), axis=0))
    pp /= np.sum(pp)
    normplot(EM, "x-", freq=pp, ms=2, mew=0.1, axes=ax)
    normplot(BM, "o", ms=2, mew=0.1, axes=ax)
    
    ax=ax_lst[pidx]; pidx += 1
    payoff=lambda x: np.exp(x)
    BM_prices=np.average(payoff(BM), axis=1)
    EM_prices= np.dot(payoff(EM), pp)
    ax.plot(range(Nsteps), zip(BM_prices, EM_prices), "o-", ms=2, mew=0.1)
#    ax.legend(["Brownian", "Exponential"], loc=2)
    
    plt.show()

def feynman_test():
    Npaths, Nsteps = 1000, 100
    T=1.0; dT=T/Nsteps
    ti=np.linspace(0,1,Nsteps)
    
#    fig=plt.figure()
#    nxx=rand_uball_(2,500)
#    plt.plot(nxx[0],nxx[1], "o",ms=2, mew=0.1)
#    plt.show()

    fig=plt.figure()
    Idx_lst=[1,3,4,2]
    Nplots=len(Idx_lst)
    rr,cc=findRowCol(Nplots)
    axlist=[ fig.add_subplot(rr,cc, idx) for idx in Idx_lst]
    pidx=0
    suptitle_fs=14
    title_fs=13
    label_fs=12
    
    ## brownian motion
    dWt=np.random.randn(Nsteps, Npaths)*np.sqrt(dT); dWt[0,:]=0.0
    Wt=np.cumsum(dWt, axis=0)
    
    ax=axlist[pidx]; pidx += 1
    ax.plot(ti, Wt)
    ax.set_title("Standard BM", fontsize=title_fs)

#    # Normality check
#    ax=axlist[pidx]; pidx += 1
#    normplot(Wt,"-", axes=ax)
#    ax.set_title("Normality Check", fontsize=title_fs)


    # Feyman paths
    dFt=rand_uball_(Nsteps, Npaths)*6; dFt[0,:]=0.0
    weight= np.exp(-0.5*np.sum(dFt**2, axis=0))
    weight /= np.sum(weight)
    dFt *= np.sqrt(dT)
    Ft=np.cumsum(dFt, axis=0)

    ax=axlist[pidx]; pidx += 1
    ax.plot(ti, Ft)
    ax.set_title("Feynman paths", fontsize=title_fs)

    # Reproduction of BM
    ax=axlist[pidx]; pidx += 1
    CM=randcolors(Nsteps)
    for itime in xrange(Nsteps):
        normplot(Ft[itime,:], "-", axes=ax, freq=weight, color=CM[itime])
        normplot(Wt[itime,:], "o", axes=ax, color=CM[itime], ms=2, mew=0.1)
    ax.set_title("Normal reproduction", fontsize=title_fs)

    # payoff calc
    ax=axlist[pidx]; pidx += 1
    payoff= lambda x: x**2
    BM_prices=np.average(payoff(Wt), axis=1)
    FM_prices=[ np.dot(payoff(Ft)[itime], weight) for itime in xrange(Nsteps) ]
    ax.plot(range(Nsteps), zip(BM_prices, FM_prices), "o-", ms=3, mew=0.1)
    plt.legend(["Brownian","Feynman"], loc=2)

    # show plot
    plt.show()

def test_RungeKutta():
	"""
	dx/dt = v
	dv/dt = -k x
	Solution: x(t)=sin(sqrt(k) t)

	"""
	k=1.5
	f=lambda t, y: np.array((y[1], -k*y[0]))
	x0=0
	v0=np.sqrt(k)
	T=8
	nsteps=200
	tt=np.linspace(0,T,nsteps)

	xx,vv = RungeKutta(tt, f, (x0,v0) )
	xxe,vve = Euler(tt, f, (x0,v0) )

	xx_th=np.sin(np.sqrt(k)*tt)
	plt.plot(tt, xx, "o-", ms=2, mew=0.1)
	plt.plot(tt, xxe, "o-", ms=2, mew=0.1)
	plt.plot(tt, xx_th)
	plt.legend(["RungeKutta", "Euler", "Exact"], loc=2)
	plt.show()

def RungeKutta(tt, f, init_val):
	nsteps=len(tt)
	h=(tt[-1]-tt[0])*1.0/nsteps
	yy=np.zeros( (nsteps, 2) ) # [ [xi,vi] ]
	yy[0,:]=init_val
	for ii in xrange(1,nsteps):
		tn=tt[ii-1]
		yn=yy[ii-1,:]
		k1 = h*f(tn, yn)
		k2 = h*f(tn+h/2, yn+k1/2)
		k3 = h*f(tn+h/2, yn+k2/2)
		k4 = h*f(tn+h, yn+k3)
		yy[ii,:]=yy[ii-1,:]+(k1+2*k2+2*k3+k4)/6
	xx=zip(*yy)
	return xx
		
def Euler(tt, f, init_val):
	nsteps=len(tt)
	h=(tt[-1]-tt[0])*1.0/nsteps
	yy=np.zeros( (nsteps, 2) ) # [ [xi,vi] ]
	yy[0,:]=init_val
	for ii in xrange(1,nsteps):
		tn=tt[ii-1]
		yn=yy[ii-1,:]
		yy[ii,:]=yy[ii-1,:]+h*f(tn, yn)
	xx=zip(*yy)
	return xx


if __name__ == '__main__':
#    Hull_Whilte_1f()
#    feynman_test()
#    exp_test()
#    uball2_test()
	test_RungeKutta()
