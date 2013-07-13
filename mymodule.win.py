##*****************************
# *  mymodule: V1.3
##*****************************

import os, sys , time, string
from multiprocessing import Process, Manager, current_process
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cPickle as pickle

import colorsys

__debug_mode__ = False

########  Utility routines ############
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
    nparr=np.array(arr)
    return np.sign(nparr)*np.log(1+np.abs(nparr))
   
def inv_slog(arr):
	nparr=np.array(arr)
	return np.sign(nparr)*(np.exp(np.abs(nparr))-1.0)

def slog10(arr):
    nparr=np.array(arr)
    return np.sign(nparr)*np.log10(1+np.abs(nparr))
   
def inv_slog10(arr):
	nparr=np.array(arr)
	return np.sign(nparr)*(10**np.abs(nparr)-1.0)
	
def msplit(ss, delim):
    res=[ss]
    for dchar in delim:
        for idx,vv in enumerate(res):
            res[idx]=split(vv,dchar)
        res=reduce(lambda x,y: x+y, res)
    return res    

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
        if t: res.append('%d%s' % (t,u))
    return ' '.join(res)

def wrap_output(funcLst, outfile, errfile=None):
    sys.stdout.flush(); sys.stderr.flush()
    o_wfd=os.dup(1); o_efd=os.dup(2)
    func, args, kwargs = (list(funcLst)+[{}])[:3]

    ofd=open(outfile,"w")
    os.dup2(ofd.fileno(),1)
    sys.stdout=ofd

    if errfile:
        errfd=open(errfile,"w")
        os.dup2(errfd.fileno(),2)
        sys.stderr=errfd

    res=func(*args, **kwargs)
    sys.stdout.flush(); sys.stderr.flush()

    ofd.close()
    sys.stdout=sys.__stdout__
    os.dup2(o_wfd,1)
    if errfile: 
        errfd.close()
        sys.stderr=sys.__stderr__
        os.dup2(o_efd,2)
    return res

def str_replace(str, repl_list):
    res=str
    for pat,repl in repl_list:
        res=string.replace(res,pat,repl)
    return res

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

def plot3d(fig, subplotID, xxi,yyi,zzi, *args, **kargs):
	ax=fig.add_subplot(subplotID, projection='3d')
	return ax.plot_surface(xxi,yyi,zzi, *args, **kargs)
    
def slogplot(x,y, *args, **kargs):
    xx=slog10(x)
    yy=slog10(y)
    plt.plot(xx,yy, *args, **kargs)

def findRowCol(nn):
    h=int(np.round(np.sqrt(nn)))
    w=(nn+h-1)/h
    return w,h

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
    rr,cc=carray.shape
    # faster version
    R,G,B=HLStoRGB(THETA,FF,1.0)
    return np.array([R,G,B]).transpose(1,2,0)
   
#    return [[(R[i,j],G[i,j],B[i,j]) for j in xrange(cc) ] for i in xrange(rr)]
    # original(slow) version
#    return [[colorsys.hls_to_rgb(THETA[i,j], FF[i,j], 1.0) for j in xrange(cc) ] for i in xrange(rr)]

#########  Multiprocessing ############
#########  Class: myPool   ############

eofStr=chr(3)

def __wrapper__( (wfd,wfd2), func, args, kargs):

#    print "__wrapper__: wfd=", wfd
#    print "__wrapper__: func=", func
#    print "__wrapper__: args=", args
#    print "__wrapper__: kargs=", kargs
    sys.stdout.flush(); sys.stderr.flush()
    o_wfd=os.dup(1); o_efd=os.dup(2)
    os.dup2(wfd, 1); os.dup2(wfd2, 2)
    ret=func(*args, **kargs)
    sys.stdout.flush(); sys.stderr.flush()
    os.write(wfd,eofStr); os.write(wfd2,eofStr)
    os.close(wfd); os.close(wfd2)
    os.dup2(o_wfd,1); os.dup2(o_efd,2)
    return ret

def __pipe_collector__(rfd):
    res=[]
    brkFlag=False
    while True:
        ss=os.read(rfd, 1024)
#        print "pipe_collector:<%s>" % ss
        if not ss: break
        if ss[-1]==eofStr:
            ss=ss[:-1]
            brkFlag=True
        res.append(ss)
        if brkFlag: break
    resStr="".join(res)
#    print "pipe_collector:<%s>" % resStr
#    sys.stdout.flush()
    os.close(rfd)
    return "".join(res)

class myPoolExcpt(Exception):
    def __init__(self, msg):
        self.msg=msg
    def __str__(self):
        return self.msg

class myPoolRet:
    def __init__(self, retList, idx):
        self.retList=retList
        self.idx=idx
    def get(self):
        return pickle.loads(self.retList[self.idx])

class myPoolRetCol:
    def __init__(self, *args):
        self.lst=args
    def get(self):
        return [x.get() for x in self.lst]

def _debug_print(debug_mode, *args):
    if debug_mode:
        for str in args:
            print >>sys.__stdout__, str, 
        print >>sys.__stdout__, ""
        sys.__stdout__.flush()

class myPool:
    def __init__(self, processes=3, maxpoolcnt=50, ID=""):
        self.mgr=Manager()
        self.procList=[] # list of launched processes
        self.semaphore=self.mgr.Semaphore(max(3,processes))
        self.maxPoolCnt=maxpoolcnt
        self.ID=ID
        # request Queue for worker
        self.rQueue=self.mgr.Queue()
        # pool management
        self.pClosedEv=self.mgr.Event()
        self.pAvailable=self.mgr.Event()
        self.pAvailable.set()
        self.pList=self.mgr.list()
        # pipe management
        self.pipeAvailableEv=self.mgr.Event()
        self.pipeQueue=self.mgr.Queue()
        # list of return values
        self.retList=self.mgr.list() 
        # worker
        self.wproc=Process(target=self.__worker__, args=[])
        self.wproc.start()

    def __worker__(self):
        while True:
            _debug_print(__debug_mode__, "%s Initial ==> pList: %d"%(self.ID, len(self.pList)))
            if self.rQueue.empty() and self.pClosedEv.is_set(): 
                _debug_print(__debug_mode__, "%s Worker(pid:%d) Joining all child processes"%(self.ID, current_process().pid))
                for proc in self.procList: proc.join()
                break
            _debug_print(__debug_mode__,"%s ReqQueue.waiting...(pList: %d, pClosed: %d)"%(self.ID, len(self.pList), self.pClosedEv.is_set()))
            # process a request in the queue or wait until available
            req=self.rQueue.get()
            _debug_print(__debug_mode__, "%s ReqQueue: got item" % (self.ID))
            if req=="PIPE":
                self.pipeQueue.put( os.pipe()+os.pipe() )
                self.pipeAvailableEv.set()
            if req=="POOL":
                _debug_print(__debug_mode__, "%s Waiting for semaphore" % (self.ID))
                self.semaphore.acquire()
                _debug_print(__debug_mode__, "%s Acquired the semaphore" % (self.ID))
                _debug_print(__debug_mode__, "%s Before pop ==> pList: %d"%(self.ID, len(self.pList)))
                self.procList.append(Process(target=self.__dispatcher__, args=self.pList.pop(0)))
                _debug_print(__debug_mode__, "%s After pop ==> pList: %d"%(self.ID, len(self.pList)))
                self.procList[-1].start()
                self.pAvailable.set()
        _debug_print(__debug_mode__, "%s Worker(pid: %d) terminated!" % (self.ID, current_process().pid))

    def __dispatcher__(self, func, args, kwargs, rIdx):
        _debug_print(__debug_mode__, "%s __dispatcher__(pid:%d): args=" % (self.ID, current_process().pid), args)
        res=func(*args, **kwargs)
        self.retList[rIdx]=pickle.dumps(res)
        sys.stdout.flush();sys.stderr.flush()
        self.semaphore.release()

    def close(self):
        self.pClosedEv.set()
        _debug_print(__debug_mode__, "%s Pool closed!"%(self.ID))

    def join(self):
        self.wproc.join()
        _debug_print(__debug_mode__, "%s worker join completed!"%(self.ID))

    def apply_async(self, func, args=[], kwargs={}):
        rfd, wfd, rfd2, wfd2=self.__pipe__()  # pipe handles for stdout & stderr
        _debug_print(__debug_mode__, "%s apply_async:: pipes=(%d %d) (%d %d)"%(self.ID, rfd, wfd, rfd2, wfd2))
        retObj=self.__apply_async__(__wrapper__, [(wfd,wfd2), func, args, kwargs])
        retObj_stdout=self.__apply_async__(__pipe_collector__, [rfd])
        retObj_stderr=self.__apply_async__(__pipe_collector__, [rfd2])
        return myPoolRetCol(retObj, retObj_stdout, retObj_stderr)

    def  __apply_async__(self, func, args=[], kwargs={}):
        if self.pClosedEv.is_set():
            raise myPoolExcpt("Pool is closed, but new process was tried to be added.")
        self.pAvailable.wait()
        # return Obj
        rIdx=len(self.retList)
        retObj=myPoolRet(self.retList, rIdx)
        self.retList.append("")
        # submit the job to the Pool
        self.pList.append( (func, args, kwargs, rIdx) ) 
        if len(self.pList) >= self.maxPoolCnt:
            self.pAvailable.clear()
        # place a Req to worker
        self.rQueue.put("POOL")
        return retObj

    def __pipe__(self):
        self.pipeAvailableEv.clear()
        self.rQueue.put("PIPE")
        # wait until available
        self.pipeAvailableEv.wait()
        # return the fd
        return self.pipeQueue.get()

#********************************************
#  test routines
#********************************************
def PrintSomething(title, n):
    for i in xrange(n):
        print >>sys.stdout, "stdout: %s(%d)" % (title, i); sys.stdout.flush()
        print >>sys.stderr, "stderr:%s(%d)" % (title, i); sys.stderr.flush()
        print >>sys.__stdout__, "__stdout__: %s(%d)" % (title, i); sys.__stdout__.flush()
        print >>sys.__stderr__, "__stderr__:%s(%d)" % (title, i); sys.__stderr__.flush()

def myPoolTest(ID=""):
    pool=myPool(processes=5, maxpoolcnt=1000, ID=ID)
    Nproc=2
    retlst=[]
    tagfunc=lambda i: "%s subproc[%d/%d]" %(ID,i+1,Nproc)
    for i in xrange(Nproc):
        tag=tagfunc(i) 
        retlst.append(pool.apply_async(PrintSomething, [tag,1]))
    pool.close()
    pool.join()
    for i,oo in enumerate(retlst):
        tag=tagfunc(i)
        print "Output to stdout(%s):" % tag
        print oo.get()[1]
    for i,oo in enumerate(retlst):
        tag=tagfunc(i)
        print "Output to stderr(%s):" % tag
        print oo.get()[2]

def wrap_test():
    wrap_output((PrintSomething, ("hello",1), {}), "wrap.out", "wrap.err")
    print "After wrap"
#    Process(target=wrap_output, args=[ (PrintSomething, ("hello",1) ), "wrap.out", "wrap.err" ] ).start()
#    Process(target=wrap_output, args=[ (PrintSomething, ("HELLO",1) ), "WRAP.out", "WRAP.err" ] ).start()

def pool_test():
    Npools=2
    plist=[]
    for i in xrange(Npools):
        plist.append(Process(target=myPoolTest,args=["Pool[%d]" % i]))
        plist[-1].start()
    for proc in plist:
        proc.join()

def evworker(Evt, ID):
    while True:
       Evt.wait()
       print "%s woke up(Evt.is_set=%d)" % (ID, Evt.is_set())
       sys.stdout.flush()
       time.sleep(0.5) 

def ev_test():
   mgr=Manager()
   evt=mgr.Event()
   NN=2
   plist=[]
   for i in xrange(NN):
       plist.append(Process(target=evworker, args=[evt, "Proc[%d]" % i]))
       plist[-1].start()
   while True:
       print "Event set for 1 sec"
       sys.stdout.flush()
       evt.set(); 
       time.sleep(1)
       print "Event cleared for 5 sec"
       sys.stdout.flush()
       evt.clear()
       time.sleep(5)
   for proc in plist:
        proc.join()

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
    wrap_test()
#    pool_test()
#    ev_test()
#    cond_test()
