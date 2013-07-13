##*********************************
# *  mymodule: V1.5
# *
# *** Changes from V1.4 *********** 
# *  Two separate implementations for Popen and Process are merged together into myPool
# *********************************

import os, sys , time, string
from multiprocessing import Process, Manager, current_process, Lock, RLock, Queue
from subprocess import Popen
from threading import Thread
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy import stats
from scipy.stats import norm
import cPickle as pickle
from matplotlib.colors import LinearSegmentedColormap
import colorsys
import random
import itertools as it
import signal
from bisect import insort

__debug_mode__ = True
__debug_console__=open("debug.log","w")
__debugLock__ = Lock() 

########  Utility routines ############

def nop(*args, **kwargs):
    return None

def drop_none(lst):
    return it.dropwhile(lambda x: x is None, lst)

def lmatch(source, ref):
    """ matches the length of source to ref by expanding the elements of ref
    """
    for i in xrange(len(source), len(ref)):
        source.append(ref[i])

def combine_num_word(num, word, plural="s", zero=None):
    ss=word
    nn=str(num)
    if num > 1:
        ss += plural
    elif num==0 and zero is not None:
        nn=zero
    return " ".join([nn, ss])
        
def assure_dir(o_dir_path, interactive=False):
    dir_path=os.path.expanduser(o_dir_path) 
    if os.path.exists(dir_path):
        if not os.path.isdir(dir_path):
            print >>sys.stderr, ("<%s> exists, but not a directory"%dir_path)
    else:
        bpath = os.path.dirname(dir_path)
        if bpath != "":
            assure_dir(bpath, interactive)
        
        ans="y"
        if interactive:
            while True:
                ans=raw_input("Directory does not exist. Want to create ?[y/n]")
                ans=string.lower(ans)
                if ans=="y" or ans=="n":
                    break
                
        if ans=="y":
            if "/" in dir_path:
                parent,base=os.path.split(dir_path)
                assure_dir(parent)
#            print "Creating <%s>... "%dir_path,
            os.mkdir(dir_path)
#            print "Done"

def backup_trunc(*args):
    for fname in args:
        if os.path.exists(fname):
            bak_fname=fname+"~"
            if os.path.exists(bak_fname):
                os.remove(bak_fname)
            try:
                os.rename(fname, bak_fname)
            except Exception,e:
                sys.stderr.write("Error: %s in renaming the file <%s>\n"%(
                         e.message, fname))
                sys.stderr.flush()
                raise e

        fd=open(fname,"w")
        fd.close()

def list_sprint(ll, delim=","):
    return delim.join([str(v) for v in ll])

def qstring(v):
    res = v
    if type(v) is str and ' ' in v:
        res='"' + v + '"'
    else:
        res = str(v)
    return res

def list_qprint(ll, delim=","):
    return delim.join([qstring(v) for v in ll])

def prefix_list(ss, ll):
    return [ (ss + vv) for vv in ll]

def suffix_list(ll,ss):
    return [ (vv+ss) for vv in ll]

def readcsv(fname, format=None, header=False):
    with open(fname,"r") as ifd:
        res=[]
        firstline=True
        for ll in ifd.xreadlines():
            ll=string.strip(ll)
            arr=string.split(ll,",")
            arr=[ string.strip(ss) for ss in arr] 
            if (format is not None) and not (firstline and header):
                for idx, ff in enumerate(format):
                    v=arr[idx]
                    if ff=='i':
                        v=int(v) 
                    elif ff=='f':
                        v=float(v)
                    arr[idx]=v
            res.append(arr)
            firstline=False
    return res

def randstring(n,prefix="", suffix=""):
    """
    n: length of the random string
    returns random string of alphanumeric characters
    """
    chrlst=string.ascii_uppercase+string.ascii_lowercase+string.digits
    return prefix+''.join( random.choice(chrlst) for i  in xrange(n) )+suffix

def swapbits(num, mask1, mask2):
    v1=1 if num & mask1 else 0; v2=1 if num & mask2 else 0;
    v=num
    if v1 != v2:
        v ^= mask1
        v ^= mask2
    return v

def reversed(x, num_bits):
    res=0
    for i in xrange(num_bits):
        if x & (1 << i):
            res |= (1 << (num_bits-1-i))
    return res

def bitrev_shuffle(lst):
    mm=np.max(lst)
    nbits=int(np.ceil(np.log2(mm)))
    res=[]
    for i,v in enumerate(lst):
        res.append( (reversed(v, nbits), i) )
    res=sorted(res)
    return [lst[i] for v,i in res]

def statSummary(arr):
    #arr must be sorted in increasing order
    NN=len(arr)
    MIN=arr[0]; MAX=arr[-1]
    Q1_idx=int(round(0.25*NN)); Q3_idx=int(round(0.75*NN));
    Q1=arr[Q1_idx]; Q3=arr[Q3_idx]
    if NN%2==0:
        Q2=0.5*(arr[NN/2-1]+arr[NN/2])
    else:
        Q2=arr[(NN+1)/2-1]
    return (np.mean(arr), np.std(arr), MIN, Q1, Q2, Q3, MAX)

def slog(arr):
    return np.sign(arr)*np.log(1+np.abs(arr))

def inv_slog(arr):
    return np.sign(arr)*(np.exp(np.abs(nparr))-1.0)

def slog10(arr):
    return np.sign(arr)*np.log10(1+np.abs(nparr))

def inv_slog10(arr):
    return np.sign(arr)*(10**(np.abs(arr))-1.0)

def msplit(ss, delim, joinadj=False):
    res=[ss]
    for dchar in delim:
        for idx,vv in enumerate(res):
            res[idx]=string.split(vv,dchar)
        res=reduce(lambda x,y: x+y, res)
    if joinadj:
        res=[v for v in res if v != ""]
    return res    

class PlotContext:
    def __init__(self,nrows,ncols):
        self.nrows=nrows
        self.ncols=ncols
        self.pCnt=0
    
    def NewFigure(self):
        plt.figure()
        self.pCnt=0
        
    def NewSubplot(self):
        self.pCnt += 1
        plt.subplot(self.nrows, self.ncols, self.pCnt)
    
    def SetCurrent(self,pCnt):
        self.pCnt = pCnt
        plt.subplot(self.nrows, self.ncols, self.pCnt)
        

def mytime(ss):
    """ converts ss (sec) to a string showing days, hours,min and sec
    """
    m,s=divmod(ss,60)
    h,m=divmod(m,60)
    d,h=divmod(h,24)
    ll=(d,h,m,s)
    uu=('day','hr','min','sec')
    res=[]
    for t,u in zip(ll,uu):
        if t: res.append('%2d%s' % (t,u))
    return ' '.join(res)

def mymsg(idx,N,stime,msg):
    symblst=['|','\\', '-', '/']
    pp=int(round(((min(idx+1,N)*100.0)/N)))
    el=time.time()-stime
    ss=symblst[idx % len(symblst)]
    print "\r{0} [{1:{2}}/{3}:{4:3}%] ({5}): {6}".format(ss,idx,len(str(N-1)),N,pp,mytime(el),msg),
    sys.stdout.flush()

class myMSG():
    def __init__(self, Nsteps,stime=None, interval=(0.0,1.0)):
        if stime is None:
            self.stime=time.time()
        else:
            self.stime=stime
        self.symblst=['|','\\', '-', '/']
        self.lst=[(0, interval, Nsteps)]
        self.plen=0
        self.pcent=0 # percent of progress (0~1)

    def write(self, msg, post=False):
        idx, itv, NN = self.lst[-1]
        pp=min(idx+(1 if post else 0),NN)*1.0/NN
        itv=self.lst[-1][1]
        self.pcent=round((1.-pp)*itv[0]+pp*itv[1], 2)
        el=time.time()-self.stime
        ss=self.symblst[idx % len(self.symblst)]
        mstr="\r{0} [{1:{2}}/{3}:{4:3}%] ({5}): {6}".format(
            ss,idx,len(str(NN-1)),NN,int(self.pcent*100),mytime(el),msg)
        if self.plen > len(mstr):
            print "\r"+" "* self.plen,
        print mstr,
        self.plen=len(mstr)
        sys.stdout.flush()

    def push(self,Nsteps):
        idx, itv, NN = self.lst[-1]
        ilen=(itv[1]-itv[0])/NN
        itv2=( self.pcent, self.pcent+ilen)
        self.lst.append( (0, itv2, Nsteps) )

    def inc(self):
        while True:
            idx, itv, NN = self.lst[-1]
            if idx == NN:
                self.lst.pop()
            else:
                break
        idx += 1
        self.lst[-1]=(idx, itv, NN)

def myMSG_test():
    NN=10
    mobj=myMSG(NN)
    for i in xrange(NN):
        mobj.write("HELLO WORLD ...")
        if i == 4:
            mobj.push(NN)
            for j in xrange(NN):
                mobj.write( "hello...")
                time.sleep(0.2)
                mobj.write( "hello...", True)
                mobj.inc()
        mobj.write("HELLO WORLD ...", True)
        mobj.inc()
        time.sleep(0.2)


def str_replace(str, repl_list):
    res=str
    for pat,repl in repl_list:
        res=string.replace(res,pat,repl)
    return res

######## extended list for tuple indexing/str concat. ####
class elist(list):
    def __init__(self,arg=[]):
        super(elist,self).__init__(arg)
    def __getitem__(self, tupIdx):
        if not hasattr(tupIdx, "__iter__"):
            return list.__getitem__(self, tupIdx)
        v=self
        for idx in tupIdx:
            v=list.__getitem__(v,idx)
        return v
    @property
    def shape(self):
        shdim=[]
        v=self
        while(True):
            shdim.append(len(v))
            if len(v) > 0 and hasattr(v[0],"__iter__"):
                v=v[0]
            else:
                break
        return tuple(shdim)
    ### for future implementation
    #@PROPERTY.setter #PROPERTY is the name of the property to be implemented
    #@PROPERTY.deleter #PROPERTY is the name of the property to be implemented

    def __rshift__(self, ss):
        if isinstance(ss, list):
            return elist([v + sv  for v in self for sv in ss])
        else:
            return elist([v + ss for v in self])
    
    def __rrshift__(self, ss):
        if isinstance(ss, list):
            return elist([sv + v for sv in ss for v in self])
        else:
            return elist([ss + v  for v in self])
 	
    def __lshift__(self, ss):
        if isinstance(ss, list):
            return elist([ elist([v + sv for sv in ss]) for v in self])
        else:
            return elist([v + ss for v in self])
    
    def __rlshift__(self, ss):
        if isinstance(ss, list):
            return elist([ elist([sv + v for v in self]) for sv in ss])
        else:
            return elist([ss + v for v in self])

    def __mod__(self, ss):
        if isinstance(ss, list):
            return elist([ elist([v % sv for sv in ss]) for v in self])
        else:
            return elist([v % ss for v in self])

    def format(self, *args):
        return elist([v.format(*args) for v in self])
        
######## Plot function ################
_segData= \
        {'red':((0, 0, 0),
                (0.1666,0,0),
                (0.3333, 1, 1),
                (0.6666, 1, 1),
                (0.8333, 0, 0),
                (1, 0, 0)
                ),
         'green': ((0, 1, 1),
                (0.1666, 0, 0),
                (0.5, 0, 0),
                (0.6555, 1, 1),
                (1, 1, 1)
                ),
         'blue': ((0, 1, 1),
                (0.3333, 1, 1),
                (0.5, 0, 0),
                (0.8333, 0, 0),
                (1, 1, 1)
                )         
        }
zmap=LinearSegmentedColormap('zmap', _segData)

def randcolors(ncolors, cmap=zmap, offset=0.4, doRandom=False):
    if not doRandom: np.random.seed([1])
    numlst=range(1,ncolors+1); np.random.shuffle(numlst)
    fidx=[(ii*1.0/(ncolors+1.0)+offset) % 1.0 for ii in numlst]
#    fidx=[(ii*1.0/(ncolors+1.0)+offset) % 1.0 for ii in bitrev_shuffle(range(1, ncolors+1))]
    return [cmap(ff) for ff in fidx]
    
def gplot3d(fig, subplotID, X1, X2, Z, *args, **kargs):
    xx1,xx2=np.meshgrid(X1,X2)
    zz=mlab.griddata(X1, X2, Z, xx1, xx2)
    return plot3d(fig, subplotID, xx1, xx2, zz, *args, **kargs)

def plot3d(fig, subplotID, xx,yy,zz, *args, **kargs):
   ax=kargs.pop('axes', fig.add_subplot(subplotID, projection='3d') )
   ax.plot_surface(xx,yy,zz, *args, **kargs)
   return ax

def slogplot(x,y, *args, **kargs):
    ax=kargs.pop('axes',plt)
    xx=slog10(x)
    yy=slog10(y)
    return ax.plot(xx,yy, *args, **kargs)

def semislogx(x,y, *args, **kargs):
    ax=kargs.pop('axes',plt)
    xx=slog10(x)
    return ax.plot(xx,y, *args, **kargs)

def semislogy(x,y, *args, **kargs):
    ax=kargs.pop('axes',plt)
    yy=slog10(y)
    return ax.plot(x,yy, *args, **kargs)

def __funcplot__(xarr,func, pltfunc, *args,**kargs):
    if not hasattr(xarr[0],'__iter__'):
        x=[xarr]
    else:
        x=xarr
    NN=len(x[0])
    farr=kargs.pop("freq",np.ones(NN))
    res = []
    for xx in x:
        xx, iarr=zip(*sorted(zip(xx,farr)))
        iarr /= (np.sum(iarr)+iarr[-1])
        iarr=np.cumsum(iarr)
        iarr=func(iarr)

        reg = np.linalg.lstsq(np.vstack( (np.ones(len(xx)),iarr) ).T, xx)
        ax=pltfunc(xx,iarr,*args,**kargs)
        res.append(reg)
    return res
#    return ax

def norm_analysis(x):
    return __funcplot__(x, norm.ppf, nop)[0]

def normplot(x,*args,**kargs):
    ax=kargs.pop('axes',plt)
    return __funcplot__(x, norm.ppf, ax.plot, *args, **kargs)

def distplot(x,*args,**kargs):
    ax=kargs.pop('axes',plt)
    return __funcplot__(x, lambda x:x , ax.plot, *args, **kargs)

def normlogplot(x,*args,**kargs):
    ax=kargs.pop('axes',plt)
    return __funcplot__(x, norm.ppf, ax.semilogx, *args, **kargs)

def distlogplot(x,*args,**kargs):
    ax=kargs.pop('axes',plt)
    return __funcplot__(x, lambda x:x , ax.semilogx, *args, **kargs)

def normslogplot(x,*args,**kargs):
    ax=kargs.pop('axes',plt)
    return __funcplot__(slog10(x), norm.ppf, ax.plot, *args, **kargs)

def distslogplot(x,*args,**kargs):
    ax=kargs.pop('axes',plt)
    return __funcplot__(slog10(x), lambda x:x , ax.plot, *args, **kargs)

def weibullplot(x,*args,**kargs):
    ax=kargs.pop('axes',plt)
    return __funcplot__(np.log(x), lambda x: np.log(-np.log(1.0-x)) , ax.plot, *args, **kargs)

def getRowCol(nn):
    h=int(np.round(np.sqrt(nn)))
    w=(nn+h-1)/h
    return w,h

def getFontProp(size):
    import matplotlib.font_manager as fm
    fontP = fm.FontProperties()
    fontP.set_size(size)
    return fontP

findRowCol=getRowCol

def HLStoRGB(h,l,s):
    cc=(1.0-np.abs(2.0*l-1.0))*s
    hp=(h % 1.0)*6
    x=cc*(1-np.abs(hp % 2.0 -1))
#    vals=((cc,x,0), (x,cc,0), (0,cc,x), (0,x,cc), (x,0,cc), (cc,0,x))
#    r1,g1,b1=vals[int(hp)]
    r1=( (hp<1.0) | (5.0<=hp) & (hp<6.0) )*cc+((1.0<=hp) & (hp <2.0) | (4.0<=hp) & (hp <5.0))*x
    g1=( (1.0<=hp) & (hp <3.0) )*cc+((hp<1.0) | (3.0<=hp) & (hp<4))*x
    b1=( (3.0<=hp) & (hp <5.0) )*cc+((2.0<=hp) & (hp <3.0) | (5.0<=hp) & (hp <6.0))*x
    m=l-0.5*cc
    r,g,b = r1+m, g1+m, b1+m
    return r,g,b

def cmplx_to_rgb(carray):
    ABS=np.abs(carray)
    THETA=np.angle(carray)/(2*np.pi) % 1.0
    FF=ABS/(1+ABS)
    # faster version
    R,G,B=HLStoRGB(THETA,FF,1.0)

    ### even faster (~ 2x) operation w/o loop
    return np.array([R,G,B]).transpose(1,2,0)

    ## fast operation
#    rr,cc=carray.shape
#    return [[(R[i,j],G[i,j],B[i,j]) for j in xrange(cc) ] for i in xrange(rr)]

    # original(very slow) version
#    return [[colorsys.hls_to_rgb(THETA[i,j], FF[i,j], 1.0) for j in xrange(cc) ] for i in xrange(rr)]

#########  Multiprocessing ############
#########  Class: myPool   ############

eofStr=chr(3)

def wrap_output(funcLst, outfile, errfile=None):
    sys.stdout.flush(); sys.stderr.flush()
    _ofd,_efd=sys.stdout.fileno(), sys.stderr.fileno()
    o_wfd=os.dup(_ofd); o_efd=os.dup(_efd)
    func, args, kwargs = (list(funcLst)+[{}])[:3]

    ofd=open(outfile,"w")
    os.dup2(ofd.fileno(),_ofd)

    if errfile:
        errfd=open(errfile,"w")
        os.dup2(errfd.fileno(),_efd)
        sys.stderr=errfd

    res=func(*args, **kwargs)
    sys.stdout.flush(); sys.stderr.flush()

    ofd.close()
    os.dup2(o_wfd,_ofd)

    if errfile: 
        errfd.close()
        os.dup2(o_efd,_efd)
    sys.stdout.flush(); sys.stderr.flush()
    return res

def __redirect__(stream, outfile, th_res):
    _ofd=stream.fileno()
    stream.flush()
    o_ofd=os.dup(_ofd)
    if outfile:
        ofd=open(outfile,"w")
        os.dup2(ofd.fileno(), _ofd)
        stream.flush()
        return (stream, _ofd, o_ofd, ofd, None)
    else:
        rpd,wpd=os.pipe()
        pc=threading.Thread(target=__pipe_collector__, args=(rpd, th_res)); 
        pc.start()
        os.dup2(wpd, _ofd)
        stream.flush()
        return (stream, _ofd, o_ofd, wpd, pc)

def __post_redirect__(redirect_info):
    stream, _ofd, o_ofd, wpd, thread=redirect_info
    if thread:
        os.close(wpd)
        os.dup2(o_ofd, _ofd)
        thread.join()
        os.close(o_ofd)
        stream.flush()
    else:
        ofd=wpd
        stream.flush()
        ofd.close()
        os.dup2(o_ofd,_ofd)

def __wrapper__(func, args, stdout=None, stderr=None):
    debug=True
    _debug_print(debug, "__wrapper__: calling the function <%s(%s)>" % (func,args))

    outfile=stdout #kargs.pop("stdout",None)
    outres=[]
    outinfo=__redirect__(sys.stdout, outfile, outres)

    errfile=stderr #kargs.pop("stderr",None)
    errres=[]
    errinfo=__redirect__(sys.stderr, errfile, errres)

    ret=func(*args)
#    ret="hello"
    _debug_print(debug, "__wrapper__: function <%s> returned <%s>" % (func, ret))
    sys.__stdout__.flush(); sys.__stderr__.flush()

    __post_redirect__(outinfo)
    __post_redirect__(errinfo)

    return (ret, "".join(outres), "".join(errres))


def __pipe_collector__(rfd, output=None):
    debug=False
    res=[]
    brkFlag=False
    _debug_print(debug, "pipe_collector[fd=%d] started" % rfd)
    while True:
        ss=os.read(rfd, 1024)
#        _debug_print(debug, "pipe_collector[fd=%d]:<%s>" % (rfd,ss))
        if not ss: break
        if ss[-1]==eofStr:
            ss=ss[:-1]
            brkFlag=True
        res.append(ss)
        if brkFlag: break
    resStr="".join(res)
    _debug_print(debug, "pipe_collector[fd=%d] result: <%s>" % (rfd,resStr))
    if debug: sys.stdout.flush()
    os.close(rfd)
    _debug_print(debug, "pipe_collector[fd=%d]: closed" % rfd)
    res="".join(res)
    if output is not None:
        output.append(res)
    _debug_print(debug, "pipe_collector[fd=%d]: complete!" % rfd)
    return res

class PoolExcpt(Exception):
    def __init__(self, msg):
        self.msg=msg
    def __str__(self):
        return self.msg

class PoolRet:
    def __init__(self, retList, idx):
        self.retList=retList
        self.idx=idx
    def get(self):
        return pickle.loads(self.retList[self.idx])

def _debug_print(debug_mode, *args,**kwargs):
    if debug_mode:
        with __debugLock__:
            for str in args:
                print >>__debug_console__, str, 
            if not kwargs.get("NoNewline",None): print >>__debug_console__, ""
            __debug_console__.flush()

def mp_default_print(idx, retObj):
    ret,out,err=retObj
    print out,
    print >>sys.stderr, err,
    _debug_print(__debug_mode__, 
            "mp_default_print(pid:%d):\n%s"%(
            current_process().pid, out[0:60]+"...") )
    sys.stdout.flush()
    sys.stderr.flush()

def dictappend(dd, key, itm):
    ll=dd.get(key,[])
    ll.append(itm)
    dd[key]=ll

def file_append(target, source, trunc=False):
    if trunc:
        tfd=open(target,"w")
    else:
        tfd=open(target,"a")
    with open(source,"r") as sfd:
        for line in sfd.xreadlines():
            tfd.write(line)
    tfd.close()

def subfilename(fname, idx):
    return fname + ("_%d" % idx)

class eQueue:
    def __init__(self, terminal=None, maxcount=None):
        self.mgr=Manager()
        self.queue=Queue()
        self.lock=RLock()
        self.count=0
        self.terminal=terminal
        self.closed=False
        self.maxcount=maxcount
        self.debug=False
        if self.maxcount is not None:
            self.semaphore=self.mgr.Semaphore(maxcount)

    def put(self, val, doClose=False, *args,  **kwargs):
        if self.maxcount is not None:
            self.semaphore.acquire() 
        _debug_print(self.debug, "eQueue::put: waiting for lock")
        with self.lock:
            _debug_print(self.debug, "eQueue::put: acquired a lock")
            self.queue.put(val,*args, **kwargs)
            _debug_print(self.debug, "eQueue::put: put a value to the lock")
            self.count += 1
            if doClose:
                self.closed=True
        _debug_print(self.debug, "eQueue::put: lock released")

    def close(self):
        if self.closed:
            return
        with self.lock:
            self.closed=True
        _debug_print(self.debug, "eQueue::close: Queue closed")
        
    def is_closed(self):
        with self.lock:
            return self.closed

    def get(self, *args, **kwargs):
        if self.count==0 and self.closed:
            return self.terminal
        _debug_print(self.debug, "eQueue::get: waiting to get from queue")
        vv=self.queue.get(*args, **kwargs)
        _debug_print(self.debug, "eQueue::get: waiting for a lock")
        with self.lock:
            _debug_print(self.debug, "eQueue::get: acquired a lock")
            self.count -= 1
            if self.maxcount is not None:
                self.semaphore.release()
        _debug_print(self.debug, "eQueue::get: released a lock")
        return vv

    def __len__(self):
        return self.count

    def is_available(self):
        return self.count > 0

    def empty(self):
        return self.count == 0

class eList:
    def __init__(self, maxcount=None): #terminal=None, maxcount=None):  #, *args, **kwargs):
        self.mgr=Manager()
        self.lst=self.mgr.list()
        self.evAvail=self.mgr.Event()
        self.evAvail.clear()
        self.lock=RLock()
        self.closed=False
        self.maxcount=maxcount
        if self.maxcount is not None:
            self.semaphore=self.mgr.Semaphore(maxcount)

    def __len__(self):
        return len(self.lst)

    def put(self, val, doClose=False):
        if self.maxcount is not None:
            self.semaphore.acquire() 
        with self.lock:
            self.lst.append(val)
            self.evAvail.set()
            if doClose:
                self.closed=True

    def append(self, val):
        self.put(val)
    
    def get(self, idx=0):
        with self.lock:
            self.evAvail.wait()  # Only one process is waiting
            vv=self.lst.pop(idx)
            if len(self)==0:
                self.evAvail.clear()
            if self.maxcount is not None:
                self.semaphore.release()
            return vv

    def is_available(self):
        return len(self) > 0

    def pop(self, idx=0):
        return self.get(idx)

    def empty(self):
        return len(self.lst)==0

    def close(self):
        with self.lock:
            self.closed=True
        
    def is_closed(self):
        with self.lock:
            return self.closed

    def __len__(self):
        return len(self.lst)

    def __getitem__(self, key):
        return self.lst[key]

    def __setitem__(self, key, val):
        with self.lock:
            self.lst[key]=val

    def __delitem__(self, key):
        with self.lock:
            del self.lst[key]
            if len(self)==0:
                self.evAvail.clear()

def win_plaform():
    return "win" in sys.platform

def run_parallel(Nparallel, jobs, outdir, outfiles, ID="", pool=None, dir=".qfd"):
    if pool is None:
        pool=myPool(processes=Nparallel, ID=ID, int_dir=dir)
    assure_dir(outdir)
    for idx, job in enumerate(jobs):
        ofname, efname = outfiles[idx]
        ofname=os.path.join(outdir,ofname)
        efname=os.path.join(outdir,efname)
        for proc in job:
            pool.apply_async( proc, stdout=ofname, stderr=efname )
    pool.close()
    pool.join()

class myPool:
    def __init__(self, processes=3, maxpoolcnt=1000, ID="", int_dir="."):
        self.int_dir=int_dir
        assure_dir(self.int_dir)
        self.mgr=Manager()
        self.threadList=[]
        self.semaphore=self.mgr.Semaphore(max(3,processes))
        self.maxPoolCnt=maxpoolcnt
        self.ID=ID
        # pool management
        self.poolQueue=eQueue(None)
        # list of return values
        self.retList=eList()
        self.fileList=eList()
        # filenames dict
        self.fdict={}
        # temporary files in int_dir
        if win_plaform():
            self.tmpFiles=self.mgr.list()
        # worker
        self.exitQueue=self.mgr.Queue()
        self.wproc=Thread(target=self.__worker__, args=[])
        self.wproc.start()

    def __exit_mgr__(self):
        _debug_print(__debug_mode__, "%s Exit_mgr(pid:%d) launched"%(self.ID, current_process().pid))
        exitLst=[]
        idx=0
        while not (self.retList.is_closed() and idx==len(self.retList)):
            while not exitLst or idx != exitLst[0]:
                _debug_print(__debug_mode__, "%s Exit_mgr(pid:%d) waiting for exitQueue.get()"%(self.ID, current_process().pid))
                retID=self.exitQueue.get() # get the index of the terminated process

                insort(exitLst, retID)

                _debug_print(__debug_mode__, "%s Exit_mgr(pid:%d): exit[%d] is queued (current: %d): [%s]"%(
                             self.ID, current_process().pid,retID, idx, 
                             list_sprint(exitLst)) ) 
            else:
                del exitLst[0]    
            
            for fname in drop_none(self.fileList[idx]):
                sfname=os.path.join(self.int_dir,subfilename(os.path.basename(fname),idx))
                file_append(fname, sfname)
                if not win_plaform():
                    os.remove(sfname)
                    
            ret, ostr, estr = pickle.loads(self.retList[idx])
            if ostr !="": print "%s"%ostr,
            if estr !="": print >>sys.stderr, "%s"%estr,
            sys.stdout.flush(); sys.stderr.flush()
            _debug_print(__debug_mode__, "%s Exit_mgr(pid:%d): exit[%d] processed: [%s],  %s remaining"%(
                         self.ID, current_process().pid, idx,
                         list_sprint(exitLst), combine_num_word(len(self.retList)-idx-1,"job")) )
            idx += 1
        _debug_print(__debug_mode__, "%s Exit_mgr(pid:%d) ended!" % (self.ID, current_process().pid))

    def __worker__(self):
        exThread=Thread(target=self.__exit_mgr__, args=[])
        exThread.start()
        _debug_print(__debug_mode__, "%s Worker(pid:%d) Initial ==> poolQueue: %d"%(self.ID, current_process().pid, len(self.poolQueue)))
        doFinish=False
        while not doFinish:
            _debug_print(__debug_mode__,"%s Worker(pid:%d) Queue.waiting...(poolQueue: %d, poolQueue.is_closed: %d)"%(self.ID,current_process().pid, len(self.poolQueue), self.poolQueue.is_closed()))

            # process a request in the queue or wait until available
            pinfo=self.poolQueue.get()
            if pinfo is None:
                doFinish=True
                continue
            _debug_print(__debug_mode__, "%s Worker(pid:%d) Queue: got item" % (self.ID, current_process().pid))
            _debug_print(__debug_mode__, "%s Worker(pid:%d) Waiting for semaphore" % (self.ID, current_process().pid))
            self.semaphore.acquire()
            _debug_print(__debug_mode__, "%s Worker(pid:%d) Acquired the semaphore" % (self.ID, current_process().pid))
            _debug_print(__debug_mode__, "%s Worker(pid:%d) Launching the dispatcher: "%(self.ID, current_process().pid),pinfo)
            self.threadList.append(Thread(target=self.__dispatcher__, args=pinfo))
            self.threadList[-1].start()
        else:
            _debug_print(__debug_mode__, "%s Worker(pid:%d) Joining all child processes... "%(self.ID, current_process().pid))
            for proc in self.threadList:
                proc.join()
            exThread.join()
            _debug_print(__debug_mode__, "%s Worker(pid:%d) Joining complete!"%(self.ID, current_process().pid))

        _debug_print(__debug_mode__, "%s Worker(pid:%d) exited!" % (self.ID, current_process().pid))

    def wrapper(self, func, args, ofname, efname, rIdx):
        res=__wrapper__(func, args, ofname, efname)
        _debug_print(__debug_mode__, "%s wrapper(pid:%d): [%d] function returned!" % (self.ID, current_process().pid, rIdx))
        self.retList[rIdx]=pickle.dumps(res)

    def __launcher_Popen(self, procLst, stdout, stderr, kwargs, rIdx):
        fdlst=[]
        for stname, key in zip([stdout, stderr],["stdout","stderr"]):
            if stname is not None:
                ofname=os.path.join(self.int_dir,subfilename(os.path.basename(stname),rIdx))
                if win_plaform():
                   self.tmpFiles.append(ofname)
                ofd=open(ofname,"w")
                kwargs[key]=ofd
                fdlst.append(ofd)

        pp=Popen(procLst, **kwargs)
        pp.wait()

        for fd in fdlst:
            fd.close()
        self.retList[rIdx]=pickle.dumps( ("","","") )

    def __launcher_Process(self, procLst, stdout, stderr, kwargs, rIdx):
        """ 
        kwargs is NOT used
        """
        fnameLst=[None, None]
        for idx, stname in enumerate(drop_none([stdout, stderr])):
            ofname=os.path.join(self.int_dir,subfilename(stname,rIdx))
            if win_plaform():
               self.tmpFiles.append(ofname)
            fnameLst[idx]=ofname
        ofname, efname = fnameLst

        func, args = procLst
        ### Python 2.7 is required to avoid the error message: "OSError: [Errno 10] No child processes"
        ## It comes from conflict between daemon and multiprocessing modules in handling SIGCLD signal 
        ## in earlier Python versions. Daemon sets SIGCLD to SIG_IGN when launching and it causes 
        ## terminated children to immediately be reaped (rather than becoming a zombie).
        ## But, multiprocessing's is_alive test invokes wait() to see if the process is alive which 
        ## fails if the process has already been reaped.
        proc=Process(target=self.wrapper, args=[func, args, ofname, efname, rIdx])
        proc.start()
        proc.join()

    def __dispatcher__(self, procLst, stdout, stderr, kwargs, rIdx):
        _debug_print(__debug_mode__, "%s __dispatcher__(pid:%d): [%d] args=" % (self.ID, current_process().pid, rIdx), procLst)

        if isinstance(procLst[0],str):
            self.__launcher_Popen(procLst, stdout, stderr, kwargs, rIdx)
        else:
            self.__launcher_Process(procLst, stdout, stderr, kwargs, rIdx)

        _debug_print(__debug_mode__, "%s __dispatcher__(pid:%d): [%d] function returned!" % (self.ID, current_process().pid, rIdx))
        self.exitQueue.put(rIdx)
        sys.stdout.flush();sys.stderr.flush()
        self.semaphore.release()

    def close(self):
        self.poolQueue.close()
        self.retList.close()
        _debug_print(__debug_mode__, "%s Pool closed [last job: %d]!"%(self.ID, len(self.retList)-1))

    def join(self):
        self.wproc.join()
        _debug_print(__debug_mode__, "%s Worker join completed!"%(self.ID))
        if win_plaform():
            map(os.remove, self.tmpFiles)

    def apply_async(self, procLst, stdout=None, stderr=None, kwargs={}):
        """
        kwargs is used only for Popen
        """
        rIdx=len(self.retList)
        retObj=PoolRet(self.retList, rIdx)
        if self.poolQueue.is_closed():
            raise PoolExcpt("Pool is closed. New process[%d] cannot be added." % rIdx)
        _debug_print(__debug_mode__, "%s apply_async(pid:%d) [job: %d]"%(self.ID, current_process().pid,rIdx))
        # Save out/err info into retList
        self.retList.append( ("","","") )
        self.fileList.append( (stdout,stderr) )

        for stname in drop_none([stdout, stderr]):
            if stname not in self.fdict:
                backup_trunc(stname)
                self.fdict[stname]=True

        # submit the job to the Pool
        if isinstance(procLst[0],str):
            self.poolQueue.put( (map(str,procLst), stdout, stderr, kwargs , rIdx) ) 
        else:
            self.poolQueue.put( ( procLst, stdout, stderr, {}, rIdx) ) 
        return retObj
        

#********************************************
#  test routines
#********************************************
def PrintSomething(title, n):
    for i in xrange(n):
        print >>sys.stdout, "stdout: %s(%d)" % (title, i); sys.stdout.flush()
        print >>sys.stderr, "stderr:%s(%d)" % (title, i); sys.stderr.flush()
        print >>sys.__stdout__, "__stdout__: %s(%d)" % (title, i); sys.__stdout__.flush()
        print >>sys.__stderr__, "__stderr__:%s(%d)" % (title, i); sys.__stderr__.flush()
    return "%s::%d"%(title,n)

def PoolTest(ID=""):
    from multiprocessing import Pool
#    pool=myPool(processes=5, func=mp_default_print)
    pool=myPool(processes=15, maxpoolcnt=1000, ID=ID)#, func=mp_default_print)
#    pool=Pool(processes=5)
    Nproc=20
    retlst=[]
    tagfunc=lambda i: "%s subproc[%d/%d]" %(ID,i+1,Nproc)
    ## Process based pooling
    for i in xrange(Nproc):
        tag=tagfunc(i) 
        retlst.append(pool.apply_async( (PrintSomething, [tag,1]), stdout="pool.out", stderr="pool.err"))
#        retlst.append(pool.apply_async( (PrintSomething, [tag,1]) ))
    ## Popen based pooling
    qPoolTest(ID, pool)
    
    pool.close()
    pool.join()
    print "*"*40
    for i,oo in enumerate(retlst):
        tag=tagfunc(i)
        print "Return value(%s) ==> " % tag,
        print oo.get()[0]
        sys.stdout.flush()
    for i,oo in enumerate(retlst):
        tag=tagfunc(i)
        print "Output to stdout(%s):" % tag
        print oo.get()[1]
        sys.stdout.flush()
    for i,oo in enumerate(retlst):
        tag=tagfunc(i)
        print "Output to stderr(%s):" % tag
        print oo.get()[2]
        sys.stdout.flush()

def qPoolTest(ID, pool=None):
    Nparallel=10
    progname="python"
    fname="test_print.py"
    outdir="OUTPUT"
    file_ext=elist(["out", "err"])
    outputs=[
            "jobs." >> file_ext,
            "jOBS." >> file_ext,
            ]
    Nlines=100
    Ncols=10

    Njobs=100
    Nsubj=10

    jobs=[
            [(progname, fname, "jobs_%d,%d" % (i,j), Nlines, Ncols)
                 for i in xrange(Njobs) for j in xrange(Nsubj)
            ],

            [(progname, fname, "jOBS_%d,%d" % (i,j), Nlines, Ncols)
                 for i in xrange(Njobs) for j in xrange(Nsubj)
            ]
         ]
    run_parallel(Nparallel, jobs, outdir, outputs, ID, pool)

def wrap_test():
#    wrap_output((PrintSomething, ("hello",1), {}), "wrap.out", "wrap.err")
    Process(target=wrap_output, args= [[PoolTest, ("hello1",)], "wrap1.out", "wrap1.err"]).start()
#    Process(target=wrap_output, args= [[PoolTest, ("hello2",)], "wrap2.out", "wrap2.err"]).start()
    print "After wrap"
    sys.stdout.flush()
#    Process(target=wrap_output, args=[ (PrintSomething, ("hello",1) ), "wrap.out", "wrap.err" ] ).start()
#    Process(target=wrap_output, args=[ (PrintSomething, ("HELLO",1) ), "WRAP.out", "WRAP.err" ] ).start()

def CondWorker(Cond, Cond2, ID, shVar):
    while True:
        with Cond:
            Cond.wait()
            print "%s woke up: shVar=%d" % (ID, shVar.value)
            sys.stdout.flush()
            time.sleep(3)
        time.sleep(0.5) 
        with Cond2:
            Cond2.notify()

def cond_test():
   mgr=Manager()
   Cond=mgr.Condition()
   Cond2=mgr.Condition()
   shVar=mgr.Value('i',0)
   NN=2
   plist=[]
   for i in xrange(NN):
       plist.append(Process(target=CondWorker, args=[Cond, Cond2, "Proc[%d]" % i, shVar]))
       plist[-1].start()
   while True:
       with Cond:
           print ">> Cond about to notify: shVar=%d" % shVar.value
           sys.stdout.flush()
           Cond.notify_all(); 
       sys.stdout.flush()
       with Cond2:
           shVar.value += 1
       time.sleep(2)
   for proc in plist:
        proc.join()



if __name__ == '__main__':
    pass
    PoolTest(ID="Hello")
#    qPoolTest(ID="Hello")
#    wrap_test()
#    cond_test()

