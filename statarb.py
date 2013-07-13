import numpy as np
import os
import string
import matplotlib.pyplot as plt
import scipy
import scipy.optimize
import cPickle
from mymodule import *
import itertools as it

dir_name = r'C:\Jhyang\Python\market_data'
#dir_name = r'C:\Users\jyang193\StatArb'
DoSelectList = False
doPlot = DoSelectList
doWritePickled = not DoSelectList
gainLimit = 50


expc = lambda k, t: (1.0-np.exp(-k*t))/k
risk_free_rate = 0.01


def fformat(ff, format=None, suffix=""):
    if format is None:
        format = '%.2f'
    if type(ff) is list:
        return [ fformat(vv, format, suffix) for vv in ff]
    return (format % ff) + suffix

def SetSelectList(br):
    global DoSelectList, doPlot, doWritePickled
    DoSelectList = br
    doPlot = DoSelectList
    doWritePickled = not DoSelectList

def remove_all(lst, x):
    res = []
    for itm in lst:
        if itm != x:
            res.append(itm)
    return res

def tofloat(x):
    try:
        if x=="":
            res = 0.0
        elif "N/A" in x:
            res = np.nan
        else:
            res = float(x) # if "N/A" not in x else 0.0 if x=="" else np.nan
        return res
    except ValueError:
        return np.nan


def read_csv(dir_name, filename):
    strip_func = lambda x: x.strip()
    fd = open(os.path.join(dir_name,filename), "r")
    res=[]
    for idx, rline in enumerate(fd.xreadlines()):
        line = rline.strip()
        if idx==0:
            equities = remove_all(map(strip_func, string.split(line,",")),"")
        elif idx==1:
            indices = map(strip_func, string.split(line,","))
        elif line != '':
            res.append(map(strip_func, string.split(line,",")))
        pass
    eqidx=-1
    dates = None
    darray = None
    tag = []
    dict = []
    for dataidx, data in zip(indices, zip(*res)):
        if dataidx == "Date":
            eqidx += 1
            if eqidx > 0:
                dict.append(np.array(darray).T)
            if dates is None:
                dates = data
            darray=[]
        elif dataidx != "":
            if eqidx==0:
                tag.append(dataidx)
            darray.append(map(tofloat,data))
    else:
        dict.append(np.array(darray).T)
    return equities, tag, dates, np.array(dict)

def filter_list(lst, limit=1):
    nn = len(lst)
    slst = sorted(zip(np.abs(np.array(lst)), xrange(nn)))
    res = np.zeros(nn, dtype=complex)
    for ii in xrange(limit):
        _, idx= slst[-(ii+1)]
        res[idx] = lst[idx]
    return np.array(res)

def normalize(data, axis):
    norm_data = (data.T-np.average(data, axis)).T
    norm_data = np.dot(np.diag(1.0/(np.std(norm_data, axis))), norm_data)
    return norm_data

def orthogonalize(data, axis):
    norm_data = normalize(data, axis)
    ## Calc rho, eigne val/vectors and then orthogonal coordinates
    rho = np.corrcoef(norm_data)
    eval, evecs = np.linalg.eigh(rho)
    evecs = evecs.T
    eval=eval[::-1]
    evecs=evecs[::-1]
    odata = np.dot(evecs, norm_data)
    return odata, eval, evecs, rho, norm_data

def find_UO_params(dX, X, tau=1.0):
    # dXt = mu*dt + kappa*(m-Xt)*dt + sigma*dWt
    # Return: (mu, kappa, m)
    # sig_dWt = lambda mu, kappa, m: dX - (mu*tau+kappa*(m-X)*tau)
    times = tau*(np.arange(len(X))+1.0)
    errFunc = lambda cc, mu, kappa, m: (X - cc - (mu + kappa*m)*expc(kappa, times))
    func = lambda args: np.sum(errFunc(*args)**2)
    cc_mu_kappa_m_init = [0.0, 0.01, 0.01, 0.05]
    cc, mu, kappa, m = scipy.optimize.fmin(func, cc_mu_kappa_m_init)
    sigma = np.std(errFunc(cc, mu, kappa, m)/expc(2*kappa,times))
    return (cc, mu, kappa, m, sigma)

def GetMinMax(ref, *arr):
    farr = np.array(arr).flatten()
    minmax = (np.min(farr), np.max(farr))
    if ref is None:
        return minmax
    minmax = (np.min(ref[0], minmax[0]),  np.max(ref[1], minmax[1]))
    return minmax
    
def plot_data(equities, ETF_equities, data, ETF_data, X, QX, QX_drift, dates, model_param, loc=2, Sharpe="" ):
    if not doPlot:
        return
    cc, mu, kappa, m, sigma = model_param
    SharpeRatio, (STGain,STSharpe,STEvent, STPath) = Sharpe
    
    pltCtx = PlotContext(nrows=2, ncols=2)
    pltCtx.NewFigure()
    if SharpeRatio != "":
        plt.suptitle("Sharpe: " + fformat(SharpeRatio) + \
                     ", Strat Gain: " + fformat(STGain) + \
                     ", Strat Sharpe: " + fformat(STSharpe) )
    fontP = getFontProp('small')
    
    pltCtx.NewSubplot()
    plt.title("Equities")
    plt.plot(X.T, ms=2, mew=0.1)
    plt.legend(equities, loc=loc, prop=fontP)
    plt.plot(QX.T, "k--", ms=2, mew=0.1, lw=2)
            
    pltCtx.NewSubplot()
    plt.title("ETF equities")
    plt.plot(np.cumsum(ETF_data,axis=1).T, ms=2, mew=0.1)
    plt.legend(ETF_equities, loc=loc, prop=fontP)
    
    pltCtx.NewSubplot()
    plt.title("Portfolio Cum. Return")
    plt.plot(QX.T, ms=2, mew=0.1)
    plt.plot(QX_drift.T, "k-", ms=2, mew=0.1)
    siglevel = 1.0
    QX_drift_psig = QX_drift + siglevel * sigma * np.sqrt(expc(2*kappa, dates))
    QX_drift_msig = QX_drift - siglevel * sigma * np.sqrt(expc(2*kappa, dates))
    plt.plot(QX_drift_psig.T, "--", ms=2, mew=0.1)
    plt.plot(QX_drift_msig.T, "--", ms=2, mew=0.1)
    pmin, pmax = GetMinMax(None, QX, QX_drift, QX_drift_psig, QX_drift_msig)
    if pmin < -gainLimit:
        plt.ylim(ymin = -gainLimit)
    if pmax > gainLimit:
        plt.ylim(ymax = gainLimit)
    
    pltCtx.NewSubplot()    
#    pCnt +=1
#    plt.subplot(nrows,ncols,pCnt)
    plt.title("Sigma Level")
    plt.plot((QX-QX_drift)/(sigma * np.sqrt(expc(2*kappa, dates))), "o-", ms=2,mew=0.1)
 
def back_test(SelectEquities, evecs, beta, Q, stock_data, ETF_data, model_param, strat, loc=3):
    global dir_name
    fname = 'ETF-2011(2).csv' #'ETF-2011.csv'
    ETF_equities, ETF_dtag, ETF_dates, darray= read_csv(dir_name, fname)
    dtag_idx=ETF_dtag.index("CHG_PCT_1D")
    ETF_data = darray[:,:,dtag_idx] # [asset_index, time_index]
    fname = 'Equity-2011(2).csv' #Equity-2011.csv'
    equities, dtag, eq_dates, darray= read_csv(dir_name, fname)
    equities_mask = FilterEquities(equities, SelectEquities)
    equities = TrimEquities(equities, equities_mask)
    dtag_idx=dtag.index("CHG_PCT_1D")
    stock_data = darray[equities_mask,:,dtag_idx] # [asset_index, time_index]
    ntimes = stock_data.shape[1]
    dates = np.arange(ntimes)+1.0
    
    cc, mu, kappa, m, sigma = model_param
    norm_data = normalize(ETF_data, axis=1)
    odata = np.dot(evecs, norm_data)
    odata = ETF_data
    try:
        dX = stock_data - np.dot(beta, odata)
    except ValueError:
        pass
    X = np.cumsum(dX, axis=1)
    QdX = np.dot(Q,dX)
    QX = np.dot(Q,X)   
    QX_drift = cc + (mu+kappa*m)*expc(kappa, dates)
    QX_final = QX[-1]
    Gain = strat(QX)
    SharpeRatio = (QX_final-risk_free_rate*ntimes*1.0/365)/np.std(QX-QX_drift)
    plot_data(equities, ETF_equities, stock_data, ETF_data, X, QX, QX_drift, dates, model_param, \
              loc=loc, Sharpe=(SharpeRatio,Gain) )
    return SharpeRatio, QX_final, Gain

def FilterEquities(equities, SelectEquities):
    mask = np.zeros( len(equities), dtype=bool )
    if SelectEquities is None:
        return mask+1
    for idx, eq in enumerate(equities):
        if eq in SelectEquities:
            mask[idx]=True
    return mask

def TrimEquities(lst, mask):
    res=[]
    for br, vv in zip(mask, lst):
        if br:
            res.append(vv)
    return res

def run_test():
    global dir_name
    strat = Strategy(3)
    SelectEquitiesList = [
        ## momentum
           ('QCOM UW Equity', 'TXN US Equity', 'INTL US Equity', 'PRU UN Equity', 'GS US Equity', 'KWK US Equity'),
           ('QCOM UW Equity', 'TXN US Equity', 'INTL US Equity', 'PRU UN Equity', 'MS US Equity', 'KWK US Equity'),
           ('AAPL US Equity', 'BRCM US Equity', 'PRU UN Equity', 'SU US Equity', 'KWK US Equity', 'UNT US Equity'),
           ('QCOM UW Equity', 'BRCM US Equity', 'INTL US Equity', 'SU US Equity', 'KWK US Equity', 'XOM US Equity'),
        ## mean-reversion 
#           ('QCOM UW Equity', 'NVDA US Equity', 'PRU UN Equity', 'MS US Equity', 'UNT US Equity', 'XOM US Equity'),
#           ('QCOM UW Equity', 'INTL US Equity', 'PRU UN Equity', 'GS US Equity', 'MS US Equity', 'SU US Equity'),
#           ('TXN US Equity', 'NVDA US Equity', 'BRCM US Equity', 'INTL US Equity', 'GS US Equity', 'KWK US Equity'),
        ]
    ## Read-in ETF data
    fname = 'ETF-2012(2).csv'
    ETF_equities, ETF_dtag, ETF_dates, darray= read_csv(dir_name, fname)
    dtag_idx=ETF_dtag.index("CHG_PCT_1D")
    ETF_data = darray[:,:,dtag_idx] # [asset_index, time_index]
    data = ETF_data
    ntimes = data.shape[1]
    model_param = orthogonalize(data, axis=1)
    odata, eval, evecs, rho, norm_data = model_param
    odata = ETF_data
    ## Read-in Stock data
    fname = 'Equity-2012(2).csv'
    equities_orig, dtag, dates, darray = read_csv(dir_name, fname)
    SharpeList=[]
    List = (SelectEquitiesList if DoSelectList else it.combinations(equities_orig, 6))
    for idx, SelectEquities in enumerate(List):
        print "*"*20
        print "Checking equities: ", SelectEquities
        equities_mask = FilterEquities(equities_orig, SelectEquities)
        equities = TrimEquities(equities_orig, equities_mask)
        dtag_idx=ETF_dtag.index("CHG_PCT_1D")
        data = darray[equities_mask,:,dtag_idx] # [asset_index, time_index]
        ## find coordinates of equity data in terms of ETF
        beta = np.linalg.lstsq(odata.T, data.T)[0].T
        dX = data - np.dot(beta,odata) ## residual which is supposed to follow Uhlenbeck-Ornstein process
        X = np.cumsum(dX,axis=1)
        ## find the optimal portfolio
        u = np.linalg.svd(beta)[0]
        Q = -u[:,-1] ## optimal portfolio which yields np.dot(Q.T, beta)==0.0
        QdX = np.dot(Q,dX)
        QX = np.dot(Q,X)
        if QX[-1] < 0.0:
            Q *= -1
            QdX *= -1
            QX *= -1
        dates = np.arange(ntimes)+1.0
        model_param = find_UO_params(QdX, QX, 1.0)
        cc, mu, kappa, m, sigma = model_param
        QX_drift = cc + (mu+kappa*m)*expc(kappa, dates)
        risk_free_rate = 0.011*ntimes*1.0/365  # 3 month rate
        STGain = strat(QX)
        QX_final = QX[-1]
        SharpeRatio = (QX_final-risk_free_rate)/np.std(QX-QX_drift)

        
        plot_data(equities, ETF_equities, data, ETF_data, X, QX, QX_drift, dates, model_param, 2, (SharpeRatio, STGain) )
        BTSharpeRatio, BTGain, STgain = back_test(SelectEquities, evecs, beta, Q, data, ETF_data, model_param, strat)
        resDict = dict(SharpeRatio=SharpeRatio, BTSharpeRatio=BTSharpeRatio, SelectEquities=SelectEquities, \
                       Gain=QX_final, BTGain=BTGain, STGain=STgain, \
                       model_param=model_param, rho=rho, eval=eval, evecs=evecs, portfolio=(Q, -np.dot(Q,beta)))
#        SharpeList.append( (SharpeRatio, BTSharpeRatio, SelectEquities, QX_final, \
#                            (BTGain, STgain), model_param, rho, eval, evecs, (Q, -np.dot(Q,beta)) ) )
        SharpeList.append(resDict)
#    SharepeListSorted = sorted(SharpeList, key=lambda x: x[0]*x[1], reverse=True)
    SharepeListSorted = sorted(SharpeList, key=lambda x: x['SharpeRatio']*x['BTSharpeRatio'], reverse=True)
    return SharepeListSorted

def write_pickled(dir_name, fname, obj):
    path = os.path.join(dir_name, fname)
    fs = open(path, "wb")
    cPickle.dump(obj, fs)
    fs.close()
    
def read_pickled(dir_name, fname):
    path = os.path.join(dir_name, fname)
    fs = open(path, "rb")
    obj = cPickle.load(fs)
    fs.close()
    return obj    

def dictValues(d, keys):
    return [ d[kk] for kk in keys]

def printItem(itm, idx=-1):
#    SharpeRatio, BTSharpeRatio, SelectEquities, gain, BTgain, model_param, rho, eval, evecs, Q = itm
    SharpeRatio, BTSharpeRatio, SelectEquities, gain, BTgain, STGain, model_param, rho, eval, evecs, Q = \
        dictValues(itm, ('SharpeRatio', 'BTSharpeRatio', 'SelectEquities', 'Gain', 'BTGain', 'STGain', \
                         'model_param', 'rho', 'eval', 'evecs', 'portfolio'))
    idxStr=""
    if idx >= 0:
        idxStr = '[' + str(idx) + ']'
    print ">>>>>" + idxStr
    print "Equities:", SelectEquities
    print "SharpeRatio: ", SharpeRatio, " , Gain: ", fformat(gain, '%2.1f%%')
    bt_gain, (st_gain, st_sharpe, st_event, st_path) = BTgain, STGain
    print "BackTest SharpeRatio: ", BTSharpeRatio, \
            ", Gain: ", fformat(bt_gain), \
            ", Strategy: ", fformat(st_gain), \
            ', Strategy Sharpe: ', fformat(st_sharpe)
    print "rho:\n", rho
    print "eigenvals:", eval
    print "eigenvectors:", evecs
    print "portfolio:", Q[0], "\n", Q[1]
    print "(X0, mu, kappa, m, sigma): ", model_param
    print ""       
    
def printInfo(SharepeListSorted):
    print "\n", "="*40
    print "Sorted list of informations: total ", len(SharepeListSorted), " cases studied"
    for idx, itm in enumerate(SharepeListSorted):
        printItem(itm)


def main():
    SetSelectList(True)
    if True:
        SharepeListSorted=run_test()
        if doWritePickled:
            write_pickled(dir_name, 'SharpeList.pkl', SharepeListSorted)
        printInfo(SharepeListSorted)
    else:
        SharepeListSorted=read_pickled(dir_name, "SharpeList.pkl")
#        shRatio_gain = np.array([ (BTSharpeRatio, BTgain[0]) for SharpeRatio, BTSharpeRatio, SelectEquities, \
#                                 gain, BTgain, model_param, rho, eval, evecs, Q in SharepeListSorted ])
        shRatio_gain = np.array([ (dd['BTSharpeRatio'], dd['BTGain']) for dd in SharepeListSorted ])        
        plt.plot(shRatio_gain.T[0], shRatio_gain.T[1],  "o", ms=2, mew=0.1)
        plt.title("Gain vs. SharpeRatio")
        plt.xlabel("SharpeRatio")
        plt.ylabel("Gain(%)")
        Cnt = 0
        for idx, itm in enumerate(SharepeListSorted):
            SharpeRatio, BTSharpeRatio, SelectEquities, gain, BTgain, STgain, Q = \
            dictValues(itm, ['SharpeRatio', 'BTSharpeRatio', 'SelectEquities', 'Gain', 'BTGain', 'STGain', 'portfolio'])
            if  False and 3.8 <= SharpeRatio and \
                3.8 <= BTSharpeRatio and \
                10 <= gain and \
                10 <= BTgain :
                Cnt += 1
                printItem(itm, idx)
            elif SharpeRatio < 1 and \
                 BTSharpeRatio < 1 and \
                 abs(BTgain) < 5 and abs(gain) < 5 and \
                 STgain > 10:
                pass
                Cnt +=1
                printItem(itm, idx)
        print "Total ", Cnt, " item(s) displayed"
                
    plt.show()

def WrapKalman(arr):
    return it.repeat(arr) if len(arr.shape) <= 2 else arr

def matrix_prod(*arr):
    res = None
    for mat in arr:
        if res is None:
            res = np.matrix(mat)
        else:
            res = np.dot(res, mat)
    return res

class Kalman:
    def __init__(self, Ft, Bt, Ut, P0, Qt, Rt, Ht):
        # time index must be on the first axis
        # Predict:
        #     x(t|t-1) = Ft * x(t-1|t-1) + Bt * Ut
        #     P(t|t-1) = Ft * P(t-1|t-1) * Ft.T + Qt
        # Update:
        #     x(t|t) = x(t|x-1) + Kt * (yt - Ht*x(t|t-1))
        #     Kt = P(t|t-1)*Ht.T*(Ht*P(t|t-1)*Ht.T + Rt)^(-1)
        #     P(t|t) = (I - Kt*Ht)*P(t|t-1)
        self.Ft = WrapKalman(Ft)
        self.Bt = WrapKalman(Bt)
        self.Ut = WrapKalman(Ut)
        self.P0 = P0
        self.Qt = WrapKalman(Qt)
        self.Rt = WrapKalman(Rt)
        self.Ht = WrapKalman(Ht)
    def __call__(self, yt):
        xt=[ np.array([yt[0], 0.0]) ]
        Pt=[ self.P0 ]
        P = self.P0
        II = np.eye(*P.shape)
        for idx, Ft, Bt, Ut, Qt, Ht, Rt in \
                it.izip(xrange(1, len(yt)), self.Ft, self.Bt, self.Ut, self.Qt, self.Ht, self.Rt):
            if idx==0:
                continue
            x_est = np.dot(Ft, xt[-1]) + np.dot(Bt, Ut)
            P_est = matrix_prod(Ft, Pt[-1], Ft.T) + Qt
            
            PH = np.dot(P_est, np.matrix(Ht).T)
            FPH = np.dot(Ft, PH)
            HPH = np.dot(Ht, PH)
            St = HPH + np.matrix(Rt)
            StInv = np.linalg.inv(St)
            K = np.dot(PH, StInv)
            KH = np.dot(K, np.matrix(Ht))
            
            residual = yt[idx]-np.dot(Ht, x_est)
            xx = np.matrix(x_est).T + np.dot(K, residual)
            xt.append(np.array(xx.T)[0])
            PP = P_est - np.dot(KH, P_est)
            Pt.append(PP)
        return xt, Pt                

def check_Kalman():
    ntimes = 200
    dT = 1.0
    v = 1.5
    A = 1.2
    P00 = 2.5
    P01 = 0
    P11 = 2.0
    qf = 0.01
    mvar = 5000
    mu = 0.00
    np.random.seed(seed=9871321) #101451267)
    times = np.arange(ntimes)
    dyt = (1.0*np.random.randn(ntimes) + mu)/10.0
    yt = np.cumsum(dyt)
    yyt = np.zeros( (len(yt), 2) )
    yyt[:,0] = yt
    
    Ft = np.array([[1,dT],[0,1]], dtype=float)      
    Bt = np.zeros_like(Ft) 
    Ut = np.zeros(2)

    P0 = 0.1* np.array([[P00,P01],[P01,P11]])
    Qt = np.array([[qf/3.0, qf/2.0],[qf/2.0, qf]])

    Ht = np.array([1, 0], dtype=float)
    Rt = np.array([[mvar]])
    
    kalman = Kalman(Ft, Bt, Ut, P0, Qt, Rt, Ht)
    xt, Pt = kalman(yt)
    xxt = np.array(xt)[:,0]
    
    pltCtx = PlotContext(1,2)
    pltCtx.NewFigure()
    pltCtx.NewSubplot()
    plt.plot(times, yt, "o-", ms=2, mew=0.1)
    plt.plot(times, xxt, "-", ms=2, mew=0.1)
    plt.ylim(np.min(yt)*1.5, np.max(yt)*1.5)
    
    pltCtx.NewSubplot()
    stdev = np.array([np.std(yt[:idx+1]-xxt[:idx+1]) for idx in xrange(len(yt))])
    plt.plot(times, stdev, ms=2, mew=0.1)
    snratio = xxt / stdev
    plt.plot(times, snratio, ms=2, mew=0.1)
    
    plt.show()


def cum_apply(func1d):
    func = lambda arr:  np.array( [ func1d(arr[0:idx+1]) for idx in xrange(len(arr)) ] )
    return func

def Siglevel(arr, SS, nn):
    df = len(arr)
    if len(arr) <= 2:
        return SS, df+nn, 0.0, (0.0, 0.0)
    xvar = np.arange(len(arr))
    xx = np.array([xvar, np.ones(len(arr))], dtype=float).T
    res = np.linalg.lstsq(xx, arr)
    m, c = res[0]
    SSres = res[1][0]+SS
    Sxx = np.var(xvar)*len(arr)
    Sig_m = np.sqrt(SSres/(df+nn)/(Sxx/df))
    return SSres, df+nn, m/Sig_m, (m, c)
    
def get_Siglevel(arr, Sth, index=0, DriftOn=True):
#    return cum_apply(Siglevel)(arr)
    delay = 3
    ref_idx=0
    SSres = 0.0
    position = 0.0
    df = 1
    Lstates, LsigLevel, Lposition, Lgain = [], [], [], []
    precond = lambda idx: idx < delay
    for idx in xrange(len(arr)):
        Lposition.append(position)
        if idx==0:
            LsigLevel.append(0.0)
            Lstates.append(0.0)
            Lposition.append(position)
            Lgain.append(0.0)
            continue
        Lgain.append(position*(arr[idx]-arr[idx-1]))
        ## Check the position
        SSresT, dfT, slevel = Siglevel(arr[ref_idx:idx+1], SSres, df)
        state = Lstates[-1]
        if precond(idx):
            pass
        elif state==0:
            if DriftOn and Sth < np.abs(slevel):
                ## No drift -> Drift(+/-)
                position = 1.0 if slevel >= 0 else -1.0
                state = position
            else:
                vv = arr[idx] - np.average(arr[ref_idx:idx+1])
                stdev = np.sqrt(SSres/df)
                if vv > Sth*stdev:
                    position = -1
                elif vv < -Sth*stdev:
                    position = 1
        elif state != 0:
            if DriftOn and np.abs(slevel) < Sth:
                ## Drift -> No drift
                ref_idx = idx
                SSres += SSresT
                df += dfT
                delta = arr[idx] - arr[idx-1]
                position = 1.0 if delta > 0 else -1.0
                state = 0.0
        LsigLevel.append(slevel)
        Lstates.append(state)
    LsigLevel = np.array(LsigLevel)
    Lgain = np.cumsum(np.array(Lgain))
    Lposition = np.array(Lposition)
    return (LsigLevel, Lstates, Lgain)[index]  


#class Strategy:
#    def __init__(self, Sth, delay, stdev_exp=1.5):
#        self.Sth = Sth
#        self.delay = delay
#        self.stdev_exp = stdev_exp
#        self.riskFreeRate = 1.0*1./360 ## daily rate from yearly 1%
#        self.avgspan = 1
#        self.reset()
#
#    def reset(self):
#        self.idx, self.ref_idx, self.SSres, self.position, self.df = 0, 0, 0.0, 0, 0
#        self.df = 1
#        self.Lstate, self.LdriftLevel, self.Lposition, self.Lgain, \
#                     self.Lstdev, self.arr, self.raw, self.Lsharpe, self.Lswitch, \
#                     self.Lconvexity \
#                     = [], [], [], [], [], [], [], [], [], []
#        self.precond = lambda idx: idx < self.delay
#        self.DriftOn = True
#        
#    def getSharpe(self):
#        Lgain = np.array(self.Lgain)
#        daily_gain = np.average(Lgain)
#        eff_daily_gain = np.average(Lgain-self.riskFreeRate)
#        stdev = np.std(np.cumsum(Lgain-daily_gain))
#        stdev = max(stdev, 1.0e-5)
#        return (eff_daily_gain*self.idx) / stdev
#            
#    def apply(self, val):
#        self.raw.append(val)
#        self.arr.append(np.average(self.raw[-self.avgspan:]))
#        self.Lposition.append(self.position)
#        if self.idx==0:
#            self.LdriftLevel.append(0.0)
#            self.Lstate.append(0.0)
#            self.Lgain.append(0.0)
#            self.Lstdev.append(0.0)
#            self.Lsharpe.append(0.0)
#            self.Lswitch.append(0)
#            self.Lconvexity.append(0)
#            self.idx += 1
#            return
#        Sth, ref_idx, idx = self.Sth, self.ref_idx, self.idx
#        self.Lgain.append(self.position*(self.raw[idx]-self.raw[idx-1]))
#        self.Lswitch.append(1 if self.Lposition[-2] != self.Lposition[-1] else 0)
#        ## Check the self.position
#        SSresT, dfT, dlevel, (m, c) = Siglevel(self.arr[ref_idx:idx+1], self.SSres, self.df)
#        state = self.Lstate[-1]
#        stdev = np.sqrt(SSresT/dfT**self.stdev_exp)
#        self.Lstdev.append(stdev)
#        self.Lsharpe.append(self.getSharpe())
#        self.Lconvexity.append( dlevel - self.LdriftLevel[-1])
#        if self.idx >= 11:
#            pass
#        if self.precond(self.idx):
#            pass
#        elif state==0:
#            if self.DriftOn and Sth < np.abs(dlevel):
#                ## No drift -> Drift(+/-)
##                self.SSres += SSresT
##                self.df += dfT
##                self.ref_idx = idx
#                prePos = self.position
#                self.position = 1.0 if dlevel >= 0 else -1.0
#                state = self.position
#                switch = 1
#            else:
#                delta = val - np.average(self.arr[ref_idx:idx+1])
##                self.position = 1 if dlevel*(delta - Sth*stdev) >= 0 else -1
##                switch = 1 if np.abs(delta) > np.abs(Sth*stdev) else 0
#                if delta > Sth*stdev:
#                    if False and dlevel > 0.0:
#                        self.position = 1
#                    else:
#                        self.position = -1
#                    switch = 1
#                elif delta < -Sth*stdev:
#                    if False and dlevel > 0.0:
#                        self.position = -1
#                    else:
#                        self.position = 1
#                    switch = 1
#                prePos = self.position
#                self.position = 1.0 if dlevel >= 0 else -1.0
#        elif state != 0:
#            delta = (self.arr[idx] - self.arr[idx-self.delay])/self.delay/1.5
#            yest = (m*(idx-ref_idx)+c)
#            Sey = np.sqrt(SSresT/dfT)
#            if np.abs(self.arr[idx]-yest)/(Sey) > 100*Sth:
#                self.SSres += SSresT
#                self.df += dfT
#                self.ref_idx = idx
#                self.position = 1.0 if self.arr[idx]-yest >= 0 else -1.0
#                state = 0.0
#                switch = 1
#            elif self.DriftOn and np.abs(dlevel) < Sth:
#                ## Drift -> No drift
#                self.SSres += SSresT
#                self.df += dfT
#                self.ref_idx = idx
#                delta = self.arr[idx] - self.arr[idx-1]
##                self.position = 1.0 if delta > 0 else -1.0
#                self.position = 1.0 if dlevel <= 0 else -1.0
#                state = 0.0
#                switch = 1
#            else:
#                pass
#        self.LdriftLevel.append(dlevel)
#        self.Lstate.append(state)
#        self.idx += 1
#    
#    def get(self):
#        ## prepare the return values
#        Lgain = np.cumsum(np.array(self.Lgain))
#        Lswitch = np.cumsum(np.array(self.Lswitch))
#        Lposition, LdriftLevel, Lstate, Lstdev, Lsharpe= map(np.array, \
#                [self.Lposition, self.LdriftLevel, self.Lstate, self.Lstdev, self.Lsharpe])
#        return np.array([LdriftLevel, Lstate, Lgain, Lstdev, Lsharpe, Lswitch, Lposition]).T
#        
#    def __call__(self, val):
#        if not hasattr(val, '__iter__'):
#            val = [ val ]
#        map(self.apply, val)
#        return self.get()


def Driftlevel(arr, SS, nn):
    df = len(arr)
    if len(arr) <= 1:
        return SS, df+nn, 0.2
    m = np.average(arr)
    var = np.var(arr)
    SSres = var +SS
    Sig_m = np.sqrt(SSres/(df+nn)) #/np.sqrt(df)
    return SSres, df+nn, m/Sig_m if Sig_m > 0 else 0.0
 
def np_smooth(arr, span, axis=0):
    if len(arr) == 0:
        return np.array([0.0])
    arr = np.array(arr)
    weight = np.exp(-np.arange(span*5)*1.0/span)
    weight /= np.sum(weight)
    if len(arr.shape) > 1:
        axlen = arr.shape[axis]
        func = lambda a: np.convolve(a, weight, 'full')[:axlen]
        return np.apply_along_axis(func, axis, arr)
    else:
        return np.convolve(arr, weight, 'full')[:len(arr)]
     
def SmoothDiff(arr, span):
    return np_smooth(arr[-span:], span)[-1] - np_smooth(arr[-span-1:-1], span)[-1]
 
def cumdiff(arr):
    return np.array(arr)[1:] - np.array(arr)[:-1]

def check_pattern(arr, *pats):
    res = False, None
    for pat in pats:
        patLen = len(pat)
        if len(arr) <= patLen:
            continue
        nparr = np.array(arr)
        cdiff = cumdiff(nparr[-patLen-1:])
        ss = np.sum(np.sign(cdiff)*np.array(pat))
        if ss == patLen:
            return True, pat
    return res
  
class Strategy:
    class patDB:
        def __init__(self):
            self.iDB = {}
            self.update = self.update_default
            self.smooth = 10
            self.normFactor = 1.0/np.sum(np.exp(-np.arange(self.smooth)*1.0/self.smooth))
            self.exp_smooth = True
            
        def __getitem__(self, key):
            self.iDB.setdefault(key, (0, 0.0, 0.0, 0, 0.0, 0.0)) # count, gain, gain**2, gcount, siglevel_arr, actgain, actgain**2
            return self.iDB[key]
        
        def __str__(self):
            head='\n(mode, pattern): (count, sum_gain, sum_gain^2,gain_count, gain_siglevel, sum_act_gain, sum_act_gain^2)\n'
            lst = self.iDB.iteritems()
            slst = sorted(lst, key=lambda x: abs(x[1][1]), reverse=True)
            return head + '\n'.join(map(lambda x: str(x[0])+': '+str(tuple(x[1] + (self.sigLevel(x[0]),))), slst))

        def update_default(self, key, cnt, gain, gcount, actgain):
            cc, g, g2, gc, ag, ag2 = self.__getitem__( key )
            cc += cnt
            if not self.exp_smooth:
                g += gain
                g2 += gain**2
                ag += actgain
                ag2 += actgain**2
            else:
                sfactor = np.exp(-1.0/self.smooth)
                g = g*sfactor + gain*self.normFactor
                g2 = g2*sfactor + (gain**2)*self.normFactor
                ag = ag*sfactor + actgain*self.normFactor
                ag2 = ag2*sfactor + (actgain**2)*self.normFactor
            gc += gcount
            self.iDB[key] = (cc, g, g2, gc, ag, ag2)
            return cc, g, g2, gc, self.sigLevel(key), ag, ag2
        
        def sigLevel(self, key):
            cc, g, g2, gc, ag, ag2 = self.__getitem__(key)
            if gc == 0:
                return (0.0,0.0)
            if not self.exp_smooth:
                mg = g*1.0/gc
                mg2 = g2*1.0/gc
                mag = ag*1.0/gc
                mag2 = ag2*1.0/gc
            else:
                mg =g 
                mg2 = g2
                mag = ag
                mag2 = ag2
            stdev = np.sqrt(mg2-mg**2)
            slevel = mg/stdev
            act_stdev = np.sqrt(mag2-mag**2)
            act_slevel = mag/act_stdev
#            return slevel, act_slevel
            return stdev, act_stdev
        
        def get(self):
            return self.iDB
            
    def __init__(self, Sth, delay, stdev_exp=1.5):
        self.Sth = Sth
        self.delay = delay
        self.stdev_exp = stdev_exp
        self.riskFreeRate = 1.0*1./360 ## daily rate from yearly 1%
        self.reset()
        self.apply = self.apply_drift
#        self.apply = self.apply_raw
        self.doPrint = False

    def reset(self):
        self.idx, self.ref_idx, self.SSres, self.SSresX, self.position, self.df = 0, 0, 0.0, 0.0, 0, 0
        self.df = 1
        self.Lstate, self.LdriftLevel, self.Lposition, self.Lgain, \
                     self.Lstdev, self.drift, self.Lsharpe, self.Lswitch, \
                     self.raw, self.sraw, self.accel, self.LaccelLevel, \
                     self.osc, self.LIntGain \
                     = [], [], [], [], [], [], [], [], [], [], [], [], [], []
#        self.precond = lambda idx: idx - self.ref_idx < self.delay
        self.precond = lambda idx: idx < self.delay
        self.smooth = 5
        self.DriftOn = False
        self.check_convexity = (False, None)
        self.mode = 0
        self.pattern = ()
        self.DB = self.patDB()
        self.activeFlag = 0
        
    def getSharpe(self):
        Lgain = np.array(self.Lgain)
        daily_gain = np.average(Lgain)
        eff_daily_gain = np.average(Lgain-self.riskFreeRate)
        stdev = np.std(np.cumsum(Lgain-daily_gain))
        stdev = max(stdev, 1.0e-5)
        return (eff_daily_gain*self.idx) / stdev
  
    def getConvexity(self, idx):
        raw_drift = cumdiff(self.raw[idx-self.delay:idx+1])
        m_accel = raw_drift[-1] - np.sum(raw_drift[:-2])
        stdev = np.std(raw_drift[:-2])
        convexity_level = m_accel / stdev if stdev > 0.0 else 0.0
        return convexity_level
    
    def getDriftAccel(self,idx):
        drift_accLevel = (self.LdriftLevel[idx] - self.LdriftLevel[idx-self.delay])
        drift_accLevelStd = np.std(cumdiff(self.LdriftLevel[idx-self.delay:idx+1]))
        drift_accLevel /= drift_accLevelStd
        return drift_accLevel
          
    def setNewSegment(self, idx, SSresT, SSresXT, dfT):
        Lgain = np.array(self.Lgain)
        self.LIntGain.append( (np.sum(Lgain[self.ref_idx:idx+1]), self.ref_idx, idx) )
#        print ">> idx: gain(-delay:idx) ==> ", idx, ": ", np.sum(Lgain[-self.delay:idx+1])
#        print "   idx: gain(ref_idx:idx-delay) ==> ", idx, ": ", np.sum(Lgain[self.ref_idx:idx-self.delay])
        self.ref_idx = idx # max(0, idx - self.delay)
        self.SSres = SSresT
        self.SSresX = SSresXT
        self.df = dfT

        
    def slopeLevel(self, drift, ref_idx, idx, raw):
#===============================================================================
#         New implementation by using self.drift (slope of smoothened raw)
#===============================================================================
        avgFunc = lambda arr: np_smooth(arr, 1)[-1]
#        avgFunc = np.average
        df = idx - ref_idx
        if df < self.smooth:
            darr = np.array(self.drift[idx-self.smooth:idx])
        else:
            darr = np.array(self.drift[ref_idx:idx])
        m_drift = avgFunc(darr)
        std_drift = np.std(darr)
        dfT = self.df + df
        if df <= self.delay:
            SSresT = self.SSres*dfT*1.0/self.df
            SSresT = max(SSresT, 0.1)
            stdevT = np.sqrt(SSresT/dfT)
            crit_level = 0.0
        else:    
            SSresT = self.SSres + df*std_drift**2
            stdevT = np.sqrt(SSresT/dfT)
            crit_level = (self.drift[idx]-m_drift)/stdevT
#        return SSresT, dfT, stdevT, crit_level, m_drift
        return SSresT, dfT, stdevT, m_drift/stdevT, m_drift

    def check_sign(self, arr, pivot):
        nn = len(arr)
        if nn <= pivot:
            return False
        larr, rarr = arr[:-pivot], arr[-pivot:]
        sgn_func = lambda arr: 2*np.sign(np.sign(arr)+1)-1
        lsarr = sgn_func(larr)
        rsarr = sgn_func(rarr)
        lnum, rnum = np.sum(lsarr), np.sum(rsarr)
        res = arr[-pivot-1]*arr[-pivot] <= 0 and \
              (nn == np.abs(lnum) + np.abs(rnum)) and lnum*rnum < 0
        return res #, np.sign(rnum - lnum)
    
    def apply_drift_init(self,val):
        appFunc = lambda x, v: x.append(v)
        map(lambda x: appFunc(x, 0.0), [self.LdriftLevel, self.Lstate, self.Lgain, self.Lstdev, self.Lsharpe, self.accel, self.LaccelLevel, self.osc])
        map(lambda x: appFunc(x, 0), [self.drift, self.Lswitch])
        map(lambda x: appFunc(x, val), [self.raw, self.sraw])
        self.pattern=()
        
    def apply_drift(self, val):
        self.Lposition.append(self.position)
        if self.idx==0:
            self.apply_drift_init(val)
            self.idx += 1
            self.activeFlag = False
            return
        if self.idx >= 78:
            pass
        ## execute the position
        self.raw.append(val)
        position = self.position
        pattern = self.pattern
        mode = self.mode
        patLen = len(pattern)
        act_position = position
        igain = self.raw[-1] - self.raw[-2]
        act_gain = act_position * igain
        self.Lgain.append(self.activeFlag * act_gain)
        self.Lswitch.append(1 if self.Lposition[-2] != self.Lposition[-1] else 0)
         
        if True: # pattern is not None: # and self.idx - self.ref_idx <= patLen:
            self.DB.update((mode,pattern), 0, igain, 1, act_gain) # cnt, gain, gain_count, actual_gain
#            print 'idx: %d, gain: %f, pattern: %s, activeFlag: %d, Lgain: %f'%(self.idx, gain, pattern, self.activeFlag, self.Lgain[-1])
#            self.activeFlag = 0 if self.idx - self.ref_idx == patLen else 1
        ## smoothen the raw array and store to sraw
        self.sraw.append(np_smooth(self.raw, self.smooth)[-1])
        self.osc.append(val - self.sraw[-1])
        self.drift.append(self.sraw[-1] - self.sraw[-2])
        self.Lsharpe.append(self.getSharpe())
        ## Check drift
        if self.idx >= 29:
            pass
        Sth, ref_idx, idx = self.Sth, self.ref_idx, self.idx
        SSresT, dfT, stdevT, crit_level, m_drift = self.slopeLevel(self.drift, ref_idx, idx, self.sraw)
        self.Lstdev.append(stdevT)
        self.LdriftLevel.append(crit_level)
        ## Check acceleration
        self.accel.append(self.LdriftLevel[-1] - self.LdriftLevel[-2])
        SSresXT, dfT, stdevXT, crit_levelX, m_accel = self.slopeLevel(self.accel, ref_idx, idx, self.LdriftLevel)
        self.LaccelLevel.append(crit_levelX)
        state = 0

        
        drift_sig1 = self.check_sign(self.LdriftLevel[-3:], 1)
#        drift_sig2 = self.check_sign(self.LdriftLevel[-3:], 2)
        DriftLevel = self.LdriftLevel[-1]
        #------------------------------------------------------------------------------
        # mode setting (1) 
        mode_type=1
        if mode_type==1:
            m_DriftLevel = np_smooth(self.LdriftLevel[-self.delay:self.ref_idx:-1], self.smooth)[-1]
            mode = 0 if np.abs(m_DriftLevel) < 0.1 else 1 if m_DriftLevel > 0.0 else -1
        # mode setting (2)
        elif mode_type==2:
            m_DriftLevel = np_smooth(cumdiff(self.sraw), 2*self.smooth)[-1]
            mode = 0 if np.abs(m_DriftLevel) < 0.05 else 1 if m_DriftLevel > 0.0 else -1
        #=======================================================================
        #  Candadiate points for decision making 
        #=======================================================================
        if self.precond(self.idx):
            pass
        elif drift_sig1:
            convexity = self.getConvexity(self.idx)
            drift_accel = self.getDriftAccel(self.idx)
            toggle = 1
#            self.activeFlag, pattern = check_pattern(self.raw, (-1,1,-1,-1),(-1,1,1,-1),(1,-1,1,-1))
#            self.activeFlag, pattern = check_pattern(self.raw, (-1,1,1,1),(-1,1,1,-1))
            self.activeFlag, pattern = check_pattern(self.raw, (-1,1,1,-1))
            if not self.activeFlag:
#                self.activeFlag, pattern = check_pattern(self.raw, (-1,-1,-1,1))
#                self.activeFlag, pattern = check_pattern(self.raw, (-1,-1,-1,1), (1,-1,-1,-1))
                self.activeFlag, pattern = check_pattern(self.raw, (-1,-1,-1,1))
                toggle *= -1
        if True or not self.activeFlag:
            patLen = 4
            tpattern = np.array(np.sign(cumdiff(self.raw[-(patLen+1):])), dtype=int)
            pattern = tuple(tpattern)
        ###
        toggle, self.activeFlag = 1, True
        ###
        #===================================================================
        # Determine the right position
        #===================================================================
        self.mode = mode
        self.pattern = pattern
        key = (mode, pattern)              
        c, g, g2, gc, (glevel, aglevel), ag, ag2 = self.DB.update(key, 1, 0.0, 0, 0.0) # count, gain, gain**2, gain_count, gain_siglevel
        ## set position
        gTh = 0.5*self.Sth
        self.activeFlag = 0 if abs(glevel) < gTh else 1
        toggle = 1 if glevel >= 0.0 else -1
        if aglevel <= -gTh:
            toggle *= -1
        self.position = toggle # * np.copysign(1, DriftLevel)
        if self.doPrint: 
            print '>> idx:%d, mode:%d, pattern: %s, gain_level(gcnt): %.3f(%d), dlevel:%.3f, pos: %d' % \
                     (self.idx, self.mode, self.pattern, glevel, gc, m_DriftLevel, self.position)        
        if position != self.position:
            self.setNewSegment(idx, SSresT, SSresXT, dfT)
            
#            self.activeFlag = True
#            print '  idx: %d, convexity: %f, drift_accel: %f' % (self.idx, convexity, drift_accel)
#            if np.abs(convexity) >= max(self.Sth, np.abs(drift_accel)):
##                self.position *= np.copysign(1, self.position * convexity)
#                self.position = np.copysign(1, convexity)
#            elif np.abs(drift_accel) >= max(self.Sth, np.abs(convexity)):
#                self.position = np.copysign(1, drift_accel)
        else:
           pass # self.position = np.copysign(1, DriftLevel)
        crit_Th = 0.5
        crit_level = self.osc[-1] - np.average(self.osc[ref_idx:idx])
        stdev = np.std(self.osc[ref_idx:idx])
        if stdev==0.0:
            crit_level = 0.0
        else:
            crit_level /= stdev
        self.Lstate.append(state)
        self.idx += 1
                    
    def apply_raw(self, val):
        self.drift.append(val)
        self.Lposition.append(self.position)
        if self.idx==0:
            self.LdriftLevel.append(0.0)
            self.Lstate.append(0.0)
            self.Lgain.append(0.0)
            self.Lstdev.append(0.0)
            self.Lsharpe.append(0.0)
            self.Lswitch.append(0)
            self.idx += 1
            return
        Sth, ref_idx, idx = self.Sth, self.ref_idx, self.idx
        self.Lgain.append(self.position*(self.drift[idx]-self.drift[idx-1]))
        self.Lswitch.append(1 if self.Lposition[-2] != self.Lposition[-1] else 0)
        ## Check the self.position
        SSresT, dfT, slevel, (m,c) = Siglevel(self.drift[ref_idx:idx+1], self.SSres, self.df)
        state = self.Lstate[-1]
        stdev = np.sqrt(SSresT/dfT**self.stdev_exp)
        self.Lstdev.append(stdev)
        self.Lsharpe.append(self.getSharpe())
        if self.precond(self.idx):
            pass
        elif state==0:
            if self.DriftOn and Sth < np.abs(slevel):
                ## No drift -> Drift(+/-)
                prePos = self.position
                self.position = 1.0 if slevel >= 0 else -1.0
                state = self.position
            else:
                vv = val - np.average(self.drift[ref_idx:idx+1])
                if vv > Sth*stdev:
                    self.position = -1
                    switch = 1
                elif vv < -Sth*stdev:
                    self.position = 1
                    switch = 1
        elif state != 0:
            if self.DriftOn and np.abs(slevel) < Sth:
                ## Drift -> No drift
                self.ref_idx = idx
                self.SSres += SSresT
                self.df += dfT
                delta = self.drift[idx] - self.drift[idx-1]
                self.position = 1.0 if delta > 0 else -1.0
                state = 0.0
            else:
                pass
        self.LdriftLevel.append(slevel)
        self.Lstate.append(state)
        self.idx += 1
    
    def get(self):
        ## prepare the return values
        Lgain = np.cumsum(np.array(self.Lgain))
        Lswitch = np.cumsum(np.array(self.Lswitch))
        Lposition, LdriftLevel, Lstate, Lstdev, Lsharpe, \
            LaccelLevel, drift, osc, sraw= map(np.array, \
                [self.Lposition, self.LdriftLevel, self.Lstate, \
                    self.Lstdev, self.Lsharpe, \
                 self.LaccelLevel, self.drift, self.osc, self.sraw])
#        return np.array([LdriftLevel, Lstate, Lgain, Lstdev, Lsharpe, Lswitch, Lposition, LaccelLevel]).T
        return np.array([LdriftLevel, Lstate, Lgain, Lstdev, Lsharpe, \
                         Lswitch, Lposition, drift, LaccelLevel, sraw]).T , self.DB.get()
        
    def __call__(self, val):
        if not hasattr(val, '__iter__'):
            val = [ val ]
        map(self.apply, val)
        return self.get()
        
def get_combibations(arr, count, total):
    res = []
    for idx, subarr in enumerate(it.combinations(arr, count)):
        if idx >= total:
            break
        res.append(subarr)
    return np.array(res)

def get_rng(arr, *idx_arr):
    get_rng = lambda arr, idx: arr[0:,idx:idx+1]
    return arr[0:, np.array(idx_arr)]

def get_random_data(Sth, vol):
    npaths = 200
    ntimes = 300
#    npaths = 2
#    ntimes = 5
    mu =  0.1 * np.random.randn(npaths)
    theta =  0.5 * np.random.randn(npaths)
    kappa = 0.1 * np.abs(np.random.randn(npaths))
   
    times = np.arange(ntimes)
    paths = vol * np.random.randn(ntimes, npaths)
    prev = np.zeros(npaths)
    for idx in xrange(ntimes):
        dd = mu + kappa*(theta-prev)
        paths[idx] += prev + dd
        prev = paths[idx]
    cutoff = Sth*np.std(paths)
    ## Trim for testing
    if False:
        get_rng = lambda arr, idx: arr[:,idx:idx+1]
        paths = get_rng(paths, 21)
    return paths

def get_market_data():
    equities, dtag, dates, darray = read_csv(dir_name, 'SnP100-20070105.csv')
#    equities, dtag, dates, darray = read_csv(dir_name, 'SnP100-20090105.csv')
    dtag_idx=dtag.index("CHG_PCT_1D")
    data = darray[:, :, dtag_idx] # [asset_index, time_index]
    nequities = data.shape[0]
    paths = np.cumsum(data, axis=1).T
    if False:
        eq_idx = np.arange(nequities)
        import random
        random.shuffle(eq_idx)
        comb_data = np.array( [np.average(data[idx_arr,:], axis=0) for idx_arr in get_combibations(eq_idx, 10, 500)] )
        paths = np.cumsum(comb_data, axis=1).T
    ## Smooth the paths
    if False:
        weight = np.ones(3, dtype=float)
        weight = weight / len(weight)
        paths = np.apply_along_axis(lambda p: np.convolve(weight, p, 'valid'), 0, paths)
    ## Trim for testing
#    if True:
    if False:
#        paths = get_rng(paths, 8)
#        paths = get_rng(paths, 13)
#        paths = get_rng(paths, 63)
#        paths = get_rng(paths, 78)
#        paths = get_rng(paths, 10)
        paths = get_rng(paths, 39)
#        paths = get_rng(paths, 8, 13, 63)
#        paths = get_rng(paths, 95)
#        paths = get_rng(paths, 95, 80)
    return equities, dtag, dates, paths


def check_strategy():
    #    np.random.seed(seed=9871321)
    np.random.seed(seed=101451267)
    testMarket = True
    if testMarket:
        Sth = 1.45
        stdev_exp = 1.0
        delay = 4
        equities, dtag, dates, paths = get_market_data()
        ntimes = paths.shape[0]
        npaths = paths.shape[1]
        times = np.arange(ntimes)
    else:
        vol = 2.0 
        Sth = 1.45
        stdev_exp = 1.5
        delay = 3
        paths = get_random_data(Sth, vol)
        ntimes = paths.shape[0]
        npaths = paths.shape[1]
    #------------------------------------
    ## Generate gains by applying Strategy
#    tmp = np.apply_along_axis(lambda arr, arg: Strategy(*arg)(arr), 0, paths, (Sth, delay))
    result = [ Strategy(Sth, delay, stdev_exp)(arr) for arr in paths.T]
    tmp = np.array([ res[0] for res in result])
    patterns = Strategy.patDB()
    Tgain = 0.0
    for dd in [ res[1] for res in result ]:
        for k,(v,g,g2,gc,ag,ag2) in dd.iteritems():
            patterns.update(k, v, g, gc, ag)
#            patterns.setdefault(k, (0,0) )
#            vv, gg = patterns[k]
#            vv += v
#            gg += g
            Tgain += g #  np.abs(g)
#            print 'g:%f, Tgain:%f' % (g, Tgain)
#            patterns[k] = vv, gg
    print "pattern summary:\n", patterns
    print "total gain: ", Tgain*1.0/npaths
    LdriftLevel, Lstates, Lgain, Lstdev, Lsharpe, Lswitch, \
        Lposition, drift, osc, sraw = np.transpose(tmp,[2,1,0])
    gainArr, sharpeArr, switchArr = Lgain[-1,:], Lsharpe[-1,:], Lswitch[-1,:]
    yrGainArr = gainArr * 360./ntimes
    print 'min gain: ', np.min(gainArr), ' at idx=', np.argmin(gainArr)
    print 'max gain: ', np.max(gainArr), ' at idx=', np.argmax(gainArr)
    print 'avg gain: ', np.average(gainArr)
    
    ## Prepare plotting
    DoPlotLpos = True
    fontP = getFontProp('small')    
    pltCtx = PlotContext(nrows=2, ncols=3)
    pltCtx.NewFigure()
    
    pltCtx.NewSubplot()
    spaths = sraw
    plt.grid(True, 'major')
    plt.plot(paths, "o-", ms=2, mew=0.1)
    plt.plot(spaths , "o-", ms=2, mew=0.1)
    plt.plot(np.average(paths, axis=1), "w--", lw=1)
    plt.plot(Lgain,"o-", ms=2, mew=0.1)
    plt.hlines(0, 0, ntimes, 'k', lw=0.5)
    if DoPlotLpos: 
        pass
        plt.plot(Lposition, "-", ms=2, mew=0.1)
    
    pltCtx.NewSubplot()
#    plt.plot(yrGainArr, sharpeArr, "o", ms=2, mew=0.1)
#    plt.plot(np.median(yrGainArr), np.median(sharpeArr), "r*", ms=5, mew=0.1)
#    plt.xlabel('Gain/yr')
#    plt.ylabel('Sharpe')
    plt.plot(drift, "o-", ms=2, mew=0.1)
    if DoPlotLpos: 
        plt.plot(Lposition, "-", ms=2, mew=0.1)
#        plt.plot(Lstates, "-", ms=2, mew=0.1)  
#    plt.ylim(-2,2)
    plt.hlines(0.0, 0.0, ntimes,'k', lw=0.5)
    plt.grid(True, 'major')
    plt.title('Drift')
    
    pltCtx.NewSubplot()
    normplot(gainArr, "o-", ms=2, mew=0.1)
    normplot(paths[-1,:], "o-", ms=2, mew=0.1)
    normplot(sharpeArr, "o-", ms=2, mew=0.1)
    plt.legend( ('Gain', 'Market', 'Sharpe'), loc=2, prop = fontP )
    plt.hlines(0.0, np.min(yrGainArr)*1.1, np.max(yrGainArr)*1.1, 'k', linewidth=0.5)
    plt.vlines(0.0, -4, 4, 'k', linewidth=0.5)
        
    pltCtx.NewSubplot()
#    pltCtx.SetCurrent(1)
    plt.plot(paths, "o-", ms=2, mew=0.1)
    plt.plot(spaths, "o-", ms=2, mew=0.1) 
    plt.plot(LdriftLevel, "o-", ms=2, mew=0.1)
#    plt.plot(drift, "o-", ms=2, mew=0.1)
    plt.hlines(0, 0, ntimes)
    plt.hlines(Sth, 0, ntimes, 'y', lw=0.5)
    plt.hlines(-Sth, 0, ntimes, 'y', lw=0.5)
    if np.max(np.abs(LdriftLevel)) > 10.0:
#        plt.ylim(-10, 10)
#        plt.ylim(-2*Sth,  2*Sth)
        pass
    if DoPlotLpos: 
        pass
        plt.plot(Lposition, "o-", ms=2, mew=0.1)
#        plt.plot(Lstates, "o-", ms=2, mew=0.1)
    plt.grid(True, 'major')
    plt.title('Drift Level')
        
    pltCtx.NewSubplot()
    if True:
        plt.plot(osc,"o-", ms=2, mew=0.1)
        if DoPlotLpos: 
            plt.plot(Lposition, "o-", ms=2, mew=0.1)
#            plt.plot(Lstates, "o-", ms=2, mew=0.1)
#        plt.hlines(2, 0, ntimes)
#        plt.hlines(-2, 0, ntimes)
        ylim = 20    
        plt.ylim(-ylim, ylim)
        plt.grid(True, 'major')
        plt.title('AccelLevel')
    else:
        plt.plot(gainArr, switchArr, "o", ms=2, mew=0.1)
        plt.xlabel('Gain')
        plt.ylabel('No. swiches')

    pltCtx.NewSubplot()
#    pltCtx.SetCurrent(4)
    plt.plot(Lgain,"o-", ms=2, mew=0.1)
    Lgain_diff = np.zeros_like(Lgain)
    Lgain_diff[1:,:] = Lgain[1:,:]-Lgain[:-1,:]
    Lloss_diff = np.copy(Lgain_diff)
    Lgain_diff *= (Lgain_diff > 0.0)
    Lloss_diff *= (Lloss_diff < 0.0)
#    plt.plot(np.cumsum(Lgain_diff, axis=0),"o-", ms=2, mew=0.1)
#    plt.plot(np.cumsum(Lloss_diff, axis=0),"o-", ms=2, mew=0.1)
    plt.plot(np.average(Lgain,axis=1), 'w--', lw=1)
    plt.plot(np.average(paths, axis=1), "y--", lw=1)
    if DoPlotLpos: 
        plt.plot(Lposition, "o-", ms=2, mew=0.1)
    plt.title('Gain')

    plt.show()
    
if __name__ == "__main__":
#    main()
    check_strategy()
#    check_Kalman()
        
