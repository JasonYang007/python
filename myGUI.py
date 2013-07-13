## @package myGUI
#  Various test routines to learn Python
#
# \f{eqnarray*}{ 
#     I(x) &=& \int_{-\infty}^{\infty}\exp(-x^2)\frac{\mathcal{H}(x)}{1+x^2}\,dx \\
#   \\
#	  \mathcal{E} &=& \mathcal{A}\cdot\mathcal{B} \\
#	\\
#		z &=& \overbrace{\underbrace{x y X Y}_{\mbox{real}}+
#			   \underbrace{i A B C}_{\mbox{imaginary}}}^{\mbox{complex number}} \\
#		\mathcal{B} &=& \mathop{\sum}_{i,j=0 \atop i \le j}^{\infty} 
#				\mathcal{A}_{i,j} 
# \f}
# \f{eqnarray*}{
#       a &\stackrel{abc}{\longrightarrow}& b \\
#       x &\stackrel{def}{\longmapsto}& f(x) \\
#		x &\in& \mathbb{R}\\
#		z &\in& \mathbb{C}\\
#		\mathcal{F}(x) &=& 
#			\cases{
#			   x &, \mbox{ for }x \ge 0 \cr
#			   x^2 &, \mbox{ for }x < 0
#			}
# \f}				
# \f{aligned}
#		\sigma_1 &= x+y	\mbox{ , }&\quad \sigma_2 &= \frac{x}{y} \\
#		\sigma_1' &= 1+y \mbox{ , }&\quad \sigma_2' &= \frac{1}{y}
# \f}

import sys, os, wx
from operator import itemgetter
from mymodule import *

class MainWindow(wx.Frame):
	def __init__(self, parent, title):
		wx.Frame.__init__(self,parent,title=title,size=(200,100))
		# Text control
		self.txtcontrol=wx.TextCtrl(self, style=wx.TE_MULTILINE)

		# Status bar
		self.CreateStatusBar()

		# setting up the menu
		filemenu=wx.Menu()
		
		# Open
		OpenItm=filemenu.Append(0,"&Open","Open a file")

		# wx.ID_ABOUT and wx.ID_EXIT are standard IDs provided by wxWidgets.
		AboutItm=filemenu.Append(wx.ID_ABOUT, "&About",
				"Information about this program")
		
		filemenu.AppendSeparator()
		
		ExitItm=filemenu.Append(wx.ID_EXIT, "E&xit", "Terminate the program")

		# Creating the menubar
		menuBar = wx.MenuBar()
		menuBar.Append(filemenu,"&File")
		self.SetMenuBar(menuBar)
		self.Show(True)

		# Set events
		self.Bind(wx.EVT_MENU, self.OnOpen, OpenItm)
		self.Bind(wx.EVT_MENU, self.OnAbout, AboutItm)
		self.Bind(wx.EVT_MENU, self.OnExit, ExitItm)	
		
	def OnOpen(self,e):
		self.dirname=''
		dlg=wx.FileDialog(self,"Choose a file", self.dirname,"","*.*", 
				wx.OPEN)
		if dlg.ShowModal() == wx.ID_OK:
			self.filename=dlg.GetFilename()
			self.dirname=dlg.GetDirectory()
			f=open(os.path.join(self.dirname,self.filename),'r')
			self.txtcontrol.SetValue(f.read())
			f.close()
		dlg.Destroy()

	def OnExit(self,event):
		self.Close(True)

	def OnAbout(self,event):
		dlg=wx.MessageDialog(self,"A small text editor",
			"About Sample Editor", wx.OK)
		dlg.ShowModal()
		dlg.Destroy()

def test():
	app = wx.App(False)
	frame=MainWindow(None,'Small Editor')
	app.MainLoop()

#######################################################
## matplotlib test

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc


def testplot():
	LL=2
	delta=0.1
	x=np.arange(-LL,LL,delta)
	y=np.arange(-LL,LL,delta)
	(xi,yi)=np.meshgrid(x,y)
	zi=(1+xi**3-2*xi*yi**2+xi*yi-yi**4)*np.exp(-(xi**2+yi**2))
	CS=plt.contour(xi,yi,zi)
	plt.colorbar(CS, shrink=0.8)
	plt.clabel(CS, inline=1, fontsize=9)
#	plt.title(r'$(1+x^3-2\cdot x\cdot y^2+x\cdot y-y^4)\cdot exp(-(x^2+y^2))$')
	plt.title(r'$Time \sim N^{2.33}$')
	plt.xlabel(r'$\mathcal{X}\ Coord$')
	plt.ylabel(r'$\mathcal{Y}\ Coord$')
	plt.show()

## error bars
def testerr():
	x=np.arange(.1, 4, .5)
	y=np.exp(-x)
	yerr=0.1+0.2*np.sqrt(x)
	xerr=0.1+yerr

	plt.figure()
	plt.errorbar(x,y,xerr=0.2, yerr=0.4)
	plt.title('Simplest errorbars, 0.2 in x, 0.4 in y')

	fig,axs=plt.subplots(nrows=2, ncols=2, sharex=True)
	ax=axs[0,0]
	ax.errorbar(x,y,yerr=yerr, fmt='o')
	ax.set_title('Vert. symmetrc')

	ax.locator_params(nbins=4)

	ax=axs[0,1]
	ax.errorbar(x,y,xerr=xerr,fmt='o')
	ax.set_title('Hor. Symmetric')

	ax=axs[1,0]
	ax.errorbar(x,y,yerr=[yerr,2*yerr], xerr=[xerr,2*xerr], fmt='--o')
	ax.set_title('H, V asymmetric')

	ax=axs[1,1]
	ax.set_yscale('log')
	ylower=np.maximum(1e-2, y-yerr)
	yerr_lower=y-ylower

	ax.errorbar(x, y, yerr=[yerr_lower, 2*yerr], xerr=xerr,
		fmt='o', ecolor='g')
	ax.set_title('Mixed sym., log y')
	fig.suptitle('Variable errorbars')
	
	
	plt.show()

def plotPolar():
	fig=plt.figure()
	ax=fig.add_subplot(111, polar=True)
	r=np.arange(0, 1, .001)
	theta = 4*np.pi*r
	line,=ax.plot(theta, r, color='#ee8d18', lw=3)

	ind=800
	thisr, thistheta=r[ind], theta[ind]
	ax.plot([thistheta],[thisr],'o')
	ax.annotate('a polar annotation',
		xy=(thistheta, thisr),
		xytext=(0.05, 0.05),
		textcoords='figure fraction',
		arrowprops=dict(facecolor='k', shrink=0.05),
		horizontalalignment='left',
		verticalalignment='bottom'
	)
	plt.show()

## does not work in windows. don't know why!!!
def testTeX():
	rc('text', usetex=True)
	rc('font', family='serif')
	plt.figure(1, figsize=(6,4))
	ax=plt.axes([0.1, 0.1, 0.8, 0.7])
	t=np.arange(0.0, 1.0+0.01, 0.01)
	s=np.cos(2*2*np.pi*t)+2
	plt.plot(t,s)

	plt.xlabel(r'\textbf{time (s)}')
	plt.ylabel(r'\textit{voltage (mV)', fontsize=16)
	plt.grid(True)
	plt.show()
## Clipping to arbitrary patches and paths
#
import matplotlib.path as path
import matplotlib.patches as patches

def testPatch():
	fig=plt.figure()
	ax=fig.add_subplot(111, frameon=False, xticks=[], yticks=[])
	im=ax.imshow(np.random.rand(10,10))
	patch=patches.Circle((300,300), radius=100)
	im.set_clip_path(patch)
	plt.show()

## Calculate hypotenuse from polar coordinates
#	@param r radius1
#	@param d radius2
#	@param theta angle(radian)
#	@return hypotenuse
#   \f[
#	   \mbox{hypot}(r,d,\theta) = \sqrt{r^2+d^2-2 r d \cos(\theta)}
#	\f]
def hypot(r,d,theta):
	return np.sqrt(r*r+d*d-2*r*d*np.cos(theta))

def makeCap(phi, maxVal):
	SS=np.sign(phi)*maxVal
	Idx=np.abs(phi) > maxVal
	phi[Idx]=SS[Idx]
		
## Draw contour plots for the electric potential for two charge configuation:
# \f[
#     \phi(r,\theta)=\frac{Q_1}{r}+\frac{Q_2}{\sqrt{r^2+{d_2}^2-2 r {d_2} 
#  	  \cos(\theta)}}+\frac{Q_3}{\sqrt{r^2+{d_3}^2-2 r {d_3} \cos(\theta-\theta_3)}}
# \f]
# \f$ Q_1 \f$ is at the origin of the coordinate system, 
# \f$ Q_2 \f$ is on the x-axis apart from \f$ Q_1 \f$ by the distance \f$ d_2 \f$, and
# \f$ Q_3 \f$ is at the location \f$ (r_3\cos(\theta_3), r_3\sin(\theta_3)) \f$
# Below is the contour plot showing the potential for this configuration:
# \image html 3-charge-potential.png  "Potential contour map"
def testPotential():
	Q1, Q2, Q3=3,-1,-0.5
	d2, d3, th3=2.5, 3, np.pi/4.
	LL, delta= 3 , 0.1
	x=np.arange(-LL+d2/2.0, LL+d2/2.0, delta)
	y=np.arange(-LL,LL, delta)
	xi,yi=np.meshgrid(x,y)
	ri=np.sqrt(xi**2+yi**2)
	thi=np.arctan2(yi,xi)
	phi=Q1/ri+Q2/hypot(ri,d2,thi)+Q3/hypot(ri,d3,thi-th3)
	makeCap(phi, 5)

	fig=plt.figure()
	ax=plt.subplot(111)
	CS=plt.contourf(xi,yi,phi, 30, cmap=plt.cm.hot)
	cpoints= [(0,0), (d2,0), (d3*np.cos(th3), d3*np.sin(th3))]
	plt.plot(*zip(*cpoints), linestyle='None', marker='o', markersize=2)
	plt.colorbar(CS)
	ax.set_aspect('equal')

	plt.show()

def makeECap(Ex, Ey, limit):
	EE=Ex*Ex+Ey*Ey
	Idx=(EE > limit*limit)
	Ex[Idx], Ey[Idx] = 0,0
	
def phi(r,theta, xi,qi):
	res=0.0
	for x,q in zip(xi,qi):
		res += q/(hypot(r,x, theta)+1e-20)
	return res

def testConductor():
	#phi(r,theta)=qi*invRi(r,theta)+Q/hypot(r,theta,d)
	#qi=invRi(r,theta)*(phi(r,theta)-Q/hypot(r,theta,d))
	
	phi0, 	r0,  phi1,	r1,		d, 	eps, 	NN = \
	3,		1,	 -1,	0.5,	2,	1e-3,	5
	
	# boundary condition
	phi_i=np.append( np.ones(NN)*phi0, 
				     np.ones(NN)*phi1 )
	# location for yet unknown charges inside conductor
	xi=np.append( np.linspace(-r0, r0, NN+2)[1:-1],
				  np.linspace(-r1+d, r1+d,NN+2)[1:-1] )	
	## spherical boundary condition
	thi=np.linspace(0, np.pi, NN) # angle on the conductor surface for boundary condition
	ri=np.ones(NN)*r0
	th2i=np.arctan2(r1*np.sin(thi),d+r1*np.cos(thi))
	r2i=hypot(d, r1, np.pi-thi)
	thi=np.append(thi, th2i)
	ri=np.append(ri, r2i)

	# potential on the conductor surface	
	r_ix,  x_ix = np.ix_(ri, xi)
	th_ix, x_ix = np.ix_(thi, xi)
	invR_i=np.linalg.inv(1.0/hypot(r_ix, x_ix, th_ix))
	qi=np.dot(invR_i, phi_i)
	
	# Plot region definition
	LL, delta= 4 , 0.02
	xx=np.arange(-LL+d/2.0, LL+d/2.0, delta)
	yy=np.arange(-LL,LL, delta)
	# Calculate potential matrix
	xxi,yyi=np.meshgrid(xx,yy)
	rri=np.sqrt(xxi*xxi+yyi*yyi)
	tthi=np.arctan2(yyi,xxi)
	pphi=phi(rri,tthi,xi,qi)
	## set boundary condition
	pphi[rri <= r0]=phi0
	pphi[hypot(rri,d, tthi) <= r1]=phi1
	
	fig=plt.figure()
	ax=plt.subplot(111) # 211)
	CS=plt.contour(xxi, yyi, pphi, 60, cmap=plt.cm.jet)
	cpoints = [xi,np.zeros(len(xi))]
	plt.plot(*cpoints, linestyle='None', marker='o', ms=2)
#	plt.colorbar(CS)
	ax.set_aspect('equal')

	fig=plt.figure()
#	ax=plt.subplot(212)
	plt.plot(xi[:NN],qi[:NN],'ro-', ms=6, lw=0.5)
	plt.plot(xi[NN:],qi[NN:],'bo-', ms=6, lw=0.5)
	plt.xlim(xx[0],xx[-1])
	plt.axhline(lw=0.5, color='y')
#	ax.set_aspect('equal')
	
	plt.show()
	
	
## Draw vector plot for the electric fields for two charge configuration:
# \f[
#  \begin{eqnarray*}
#	  \vec{E}(r,\theta)&=&E_r \hat{r} + E_{\theta} \hat{\theta} \mbox{ , where}\\
#     E_r &=& \frac{Q_1}{r^2}+\frac{Q_2 (r-d_2 \cos(\theta))}
#                    {(r^2+d_2^2-2 r d_2 \cos(\theta))^{3/2}}+
#			  \frac{Q_3 (r-d_3 \cos(\theta-\theta_3))}
#                    {(r^2+d_3^2-2 r d_3 \cos(\theta-\theta_3))^{3/2}} \\
#	  E_{\theta} &=& \frac{Q_2 d_2 \sin(\theta)} 
#						{(r^2+d_2^2-2 r d_2 \cos(\theta))^{3/2}}+
#	                 \frac{Q_3 d_3 \sin(\theta-\theta_3)} 
#						{(r^2+d_3^2-2 r d_3 \cos(\theta-\theta_3))^{3/2}}
#	\end{eqnarray*}
# \f]
# \f$ Q_1 \f$ is at the origin of the coordinate system, 
# \f$ Q_2 \f$ is on the x-axis apart from \f$ Q_1 \f$ by the distance \f$ d_2 \f$, and
# \f$ Q_3 \f$ is at the location \f$ (r_3\cos(\theta_3), r_3\sin(\theta_3)) \f$
def testEfield():
	Q1, Q2, Q3=3,-1,-0.5
	d2, d3, th3=2.5, 3, np.pi/4.
	LL, delta= 3 , 0.1
	x=np.arange(-LL+d2/2.0, LL+d2/2.0, delta)
	y=np.arange(-LL,LL, delta)
	xi,yi=np.meshgrid(x,y)
	ri=np.sqrt(xi**2+yi**2)
	thi=np.arctan2(yi,xi)
	Ei=Q1/ri**2+Q2*(ri-d2*np.cos(thi))/hypot(ri,d2,thi)**3 + \
			Q3*(ri-d3*np.cos(thi-th3))/hypot(ri,d3,thi-th3)**3
	Eth=Q2*d2*np.sin(thi)/hypot(ri,d2,thi)**3 + \
			Q3*d3*np.sin(thi-th3)/hypot(ri,d3,thi-th3)**3
	ui=(np.cos(thi), np.sin(thi))
	vi=(-np.sin(thi), np.cos(thi))
	Ex,Ey=Ei*ui+Eth*vi
	makeECap(Ex, Ey, 6)
	fig=plt.figure()
	ax=plt.subplot(111)
	plt.plot([0,d2, d3*np.cos(th3)],[0,0, d3*np.sin(th3)], 'ko', markersize=2)
	CS=plt.quiver(xi,yi,Ex,Ey, color='b') #cmap=plt.cm.jet)
	ax.set_aspect('equal')

	plt.show()

import time
def elapsedTime():
	elN=np.array([100,200,400,800,1000])
	elT=[]
	for N in elN:
			mm=np.matrix(np.random.randn(N,N))
			mm=0.5*(mm+mm.T)
			print 'Elapsed time (N=%d):' % N,
			st=time.clock()
			eig=np.linalg.eigh(mm)
			elapsed=time.clock()-st
			print '\t%.3f sec' % elapsed
			elT.append(elapsed)
	elT=np.array(elT)
	xx=np.vstack([np.log(elN), np.ones(len(elN))]).T
	yy=np.log(elT)
	result=np.linalg.lstsq(xx,yy)
	(m,c),SSE=result[0], result[1]
	R2=1-SSE/(len(yy)*yy.var())
	print '-'*40
	print 'Time ~ N^%.2f' %m
	plt.plot(elN,elT,'o', markersize=10, label='raw data')
	plt.plot(elN, np.exp(c)*elN**m, 'r', label='regression')
	plt.legend()
	plt.title(r'$Time \sim N^{%.2f} (R^2=%.1f\%%)$' % (m, 100*R2)) 
	plt.xscale('log'); plt.yscale('log')
#	plt.loglog(elN,elT,'o', markersize=10)
#	plt.loglog(elN, np.exp(c)*elN**m, 'r')
	plt.xlabel('Matrix dimension')
	plt.ylabel('Elapsed time (s)')
	plt.show()

#####
## class diagram generation

class Base:
	bb=1

class Derived_1(Base):
	dd=10

class Derived_2(Base):
	dd=20

class Derived_3(Base):
	dd=30

class Derived_4(Base):
	dd=40

class DDerived(Derived_2, Derived_4):
	dd=2400

#####
##  Caller function No.1
#	@param void
#	@return void
def Caller1():
	func1()
	func2()

## Caller function No.2
#	@param void
#	@return void
def Caller2():
	func1()

## Called function No.1
#	@param void
#	@return void
def func1():
	print "Hello"

## Called function No.2
#	@param void
#	@return void
def func2():
	print "Hi, there!"

def testTuple():
	xx=[1,2,3,4]
	yy=['a','b','c','d']
	zz=['A','B','C','D']
	ll=zip(xx,yy,zz)
	(XX,YY,ZZ)=zip(*ll) # unzip!!
	print XX,YY,ZZ

class Base(object):
    def __init__(self):
        print "Base created"

class ChildA(Base):
    def __init__(self):
        Base.__init__(self)

class ChildB(Base):
    def __init__(self):
        super(ChildB, self).__init__()


def test_super():
	print ChildA(),ChildB()

from multiprocessing import Process, Pool, Manager, current_process, cpu_count
from multiprocessing import Lock as mpLock
import os, time, shelve

mp_lock=mpLock()
shelve_file='mpDB.slv'

def mpUpdateDB(key, val):
	mp_lock.acquire()
	SS=shelve.open(shelve_file)
	SS[key]=val
	SS.close()
	mp_lock.release()

def th_func(nn):
	stime=time.time()
	key='thread(%d)' % nn
	print key,' started at ', time.asctime()
	mm=np.matrix(np.random.randn(nn,nn))
	mm=mm+mm.H
	ee=np.linalg.eigh(mm)[0]
	elapsed=time.time()-stime
	print key, ' ended at %s (total %.2f sec)'% (time.asctime(), elapsed)
	mpUpdateDB(key, (ee[0],ee[nn/2],ee[-1]))

def test_mp():
	thList=[]
	stime=time.time()
	for id in [800, 700, 600]:
		thList.append(Process(target=th_func, args=(id,) ))
		thList[-1].start()
	for th in thList:
		th.join()
#	elapsed=time.strftime('%Hh %Mmin %Ssec', time.gmtime(time.time()-stime))
	elapsed='%.2f sec' % (time.time()-stime)
	print '==> Whole job ended at %s (total %s)'% (time.asctime(), elapsed)

def test_mp2():
	pool=Pool(processes=cpu_count())
	res=[]
	stime=time.time()
	for id in [1000, 1500, 900, 1050, 1400, 1300]:
		res.append(pool.apply_async(th_func,[id]))
	pool.close()
	pool.join()
#	elapsed=time.strftime('%Hh %Mmin %Ssec', time.gmtime(time.time()-stime))
	elapsed='%.2f sec' % (time.time()-stime)
	print '==> Whole job ended at %s (total %s)'% (time.asctime(), elapsed)
	print '-'*5,'Data stored in Shelve:','-'*20
	SS=shelve.open(shelve_file)
	for k,v in SS.iteritems():
		print k,':',v
## Manager test
def mp_func(id, d):
	tkk=current_process().name+('[%d]'%id)
	print '%s launched'%tkk
	for i in xrange(10):
		kk='%s: key%d'%(tkk, i)
		time.sleep(0.1)
#		d[kk]=i*2
		vv=d.get('key',[])
		vv.append(kk)
		d['key']=vv
	print '<<%s ended'%tkk

def test_mgr():
	manager=Manager()
	d=manager.dict()
	pool=Pool(processes=2)
	res=[]
	for i in xrange(3):
		res.append(pool.apply_async(mp_func, [i,d]))
	pool.close()
	pool.join()
	print '*'*40
	slist=sorted([(k,v) for k,v in d.items()], key=itemgetter(0,1))
#	for k,v in slist:
#		print '%s : %d' % (k,v)
	for v in d['key']:
		print v

## Multiprocessing.Queue test
from multiprocessing import Process, Queue
from Queue import Empty, Full

def prod1(q):
	map(q.put, [1,2,3])
#	time.sleep(1)

def prod2(q):
	map(q.put, ['A','B','C'])
#	time.sleep(1)

def consum1(q):
	while q:
		time.sleep(0.1)
		try:
			vv=q.get(True,2)
		except Empty:
			return
		print 'Consum 1: ',vv

def consum2(q):
	while q:
		time.sleep(0.1)
		try:
			vv=q.get(True,2)
		except Empty:
			return
		print 'Consum 2: ',vv

def test_queue():
	q=Queue()
	#procLst=[Process(target=p, args=(q,) ) for p in [prod1, prod2, consum1, consum2]]
	pLst=[Process(target=p, args=(q,) ) for p in [prod1, prod2]]
	cLst=[Process(target=p, args=(q,) ) for p in [consum1, consum2]]
	procLst=pLst+cLst
	for pp in procLst:
		pp.start()
#	for pp in pLst:
#		pp.join()
#	q.put('STOP')
	q.close()
#	print 'Queue is closed'
	q.join_thread()
#	for pp in procLst:
#		pp.join()

## StringIO test
def test_StringIO():
	import cStringIO as StringIO
	from contextlib import closing
	with closing(StringIO.StringIO()) as fd:
		sys.stdout=fd
		print 'This is a message to stdout'
		sys.stdout=sys.__stdout__
		print 'Message captured by StringIO ==> ', fd.getvalue()

def genOgrid(h,w, ext):
	return np.ogrid[ext[2]:ext[3]:h*1j, ext[0]:ext[1]:w*1j]
	
def mandelbrot(h,w, ext, maxit=20):
	'''Returns an image of the Mandelbrot fractal of size (h,w)
	'''
	y,x=genOgrid(h,w,ext)
	c=x+y*1j
	z=c
	divtime = maxit + np.zeros(z.shape, dtype=int)
	
	for i in xrange(maxit):
		z=z**2+c
		diverge=z*np.conj(z) > 2**2
		div_now = diverge & (divtime==maxit)
		divtime[div_now]=i 
		z[diverge]=2
		
	return divtime[::-1,:]

def test_mandelbrot():
	ext=[-2, 0.8,-1.4, 1.4]
	plt.imshow(mandelbrot(400,400, ext), extent=ext)
	plt.colorbar()
	plt.show()
	
def newton(h,w, ext, maxit=20, deg=3):
	y,x=genOgrid(h,w,ext)
	z=x+y*1j
	degm1=deg-1
	for i in xrange(maxit):
		z -= (z**deg-1)/(deg*z**degm1)
	return np.angle(z)[::-1,:]/np.pi

def findRowCol(n):
	h=int(np.round(np.sqrt(n)))
	w=(n+h-1)/h
	return h,w

def test_newton():
	h,w=500,500
	ext=[-3,3,-3,3]
	nplots, max_iter=6,5
	nrows,ncols=findRowCol(nplots)
	for idx,itr in enumerate(np.int_(np.linspace(0, max_iter+1, nplots))):
		plt.subplot(nrows, ncols, idx+1)
		plt.imshow(newton(h,w,ext,itr), extent=ext)
		plt.title('Iter=%d' % itr)
#	plt.colorbar()
	plt.show()

def zTrans(func,h,w, ext):
	y,x=genOgrid(h,w,ext)
	z=x+y*1j
	w=func(z)
	return np.abs(w)[::-1,:]/np.pi

def test_zTrans():
	h,w=500,500
	mm=5
	ext=[-mm,mm,-mm,mm]
	plt.imshow(zTrans(lambda z: z**3+z**2+2*z+50, h,w,ext), extent=ext)
	plt.show()
		
from multiprocessing import Pool, Queue, Manager
eofString=chr(3)

def subProdQ(q):
	print "Producer started"
	for i in xrange(10):
		q.put("Producer %d" % i)
	q.put(eofString)
	print "Producer finished"
	
def subReaderQ(q):
	print "Reader started"
	res=[]
	for ss in iter(q.get, eofString):
		res.append(ss)
	print "\n".join(res)
	print "Reader finished"

def Consumer(rfd):
	print "Consumer(rfd=%d)" % (rfd)
	sys.stdout.flush()
	res=[]
	brkFlag=False
	while not brkFlag:
		ss=os.read(rfd, 1024)
		if ss=="": break
		if ss[-1]==eofStr:
			ss=ss[:-1]
			brkFlag=True
		print "Consumer: <%s>" % ss
		sys.stdout.flush()
		res.append(ss)
	os.close(rfd)
	return "".join(res)
	
def Producer(wfd):
	print "Producer(wfd=%d)" % (wfd)
	sys.stdout.flush()
	for i in xrange(5):
		print "Producer[%d]" % i
	
def PrintSomething(n):
	print "PrintTest(wfd=%d)" % (wfd)
	for i in xrange(n):
		print "PrintTest[%d]" % i
	sys.stdout.flush()

def testIPC():
#	mgr=Manager()
#	q=mgr.Queue()
	rhd,whd=os.pipe()
	pool=myPool(processes=2)
	pool.apply_async(Producer, [whd])
	pool.apply_async(Consumer, [rhd])
	pool.close()
	pool.join()

def testPipe():
	pool=myPool(processes=2)
	ret=pool.apply_async(PrintSomething, [5])
	pool.close()
	pool.join()
	print "Output to stdout: ", ret.get()[1]

def ProcWrite(wfd, ID):
	wfd.write("%s ProcWrite" % ID)
	
def test_fileIO():
	fname="test.out"
	wfd=open(fname,"w")
	wfd.write("Main routine")
	sys.stdout.flush()
	Nproc=2
	procLst=[]
	for i in xrange(Nproc):
		procLst.append(Process(target=ProcWrite, args=[wfd,"Process-%d"%i]))
		procLst[-1].start()
	for proc in procLst:
		proc.join()
	wfd.close()
	
from matplotlib.colors import LinearSegmentedColormap
from mymodule import *
import scipy
import scipy.special
import colorsys
	
def test_cmap():
#	hls=np.random.rand(10,3)
#	for h,l,s in hls:
#		r,g,b=colorsys.hls_to_rgb(h,l,s)
#		r1,g1,b1=HLS_to_RGB(h,l,s)
#		print "H:%.4f L:%.4f S:%.4f ==>" % (h,l,s)
#		print "\tR:%.4f G:%.4f B:%.4f (colorsys)" % (r,g,b)
#		print "\tR:%.4f G:%.4f B:%.4f (HLS_to_RGB)" % (r1,g1,b1)
	LL=3; ext=[-LL,LL,-LL,LL]
	x=np.linspace(-LL,LL, 500)
	y=x
	xx,yy=np.meshgrid(x,y)
	zz=(xx+yy*1j)

	plotLst=[
			(lambda z: z, "$z$", False, None, None),
			(scipy.special.arcsin, "$arcsin(z)$", True, 10, '#ababab'),
#			(scipy.special.arctanh, "$arctanh(z)$", True, 10, '#ababab'),
			(lambda z: np.sin(np.pi/z), "$sin(\pi/z)$", False, None, None),
			(scipy.special.gamma, "$\Gamma(z)$", False, None, None),
			(scipy.special.psi, "$\Psi(z)$", True, 10, '#ababab'),
			(scipy.special.erf, "$erf(z)$", False, None, None),
			(lambda z: scipy.special.hankel1(0,z), "$H_0(z)$",False, 10, '#ababab'),
			(lambda z: scipy.special.airy(np.pi/2*z)[0], "$Ai(\pi z/2)$", False, None,None),
			(lambda z: scipy.special.airy(np.pi/2*z)[2], "$Bi(\pi z/2)$", False, None,None)
			]
	cc,rr=findRowCol(len(plotLst))
	for idx, pitem in enumerate(plotLst):
		plt.subplot(rr,cc,idx+1)
		func, title, contFlag, Ncont, clr = pitem
		plt.title(title, fontsize=18)
		ZZ=func(zz)
		plt.imshow(cmplx_to_rgb(ZZ), extent=ext, origin="lower")
		if contFlag:
			aZZ=np.abs(ZZ)
			plt.contour(xx,yy,aZZ/(aZZ+1),Ncont,colors=clr)
	plt.show()
	
def test_svd():
	NN=1000
	ones=np.ones(NN)
	x0=np.random.randn(NN)
	y0=1+2*x0+3*x0**2+0.2*np.random.randn(NN)
	varY=np.var(y0)
	flist=[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
	for f in flist:
		x=f*x0
		X=np.array([ones, x, x**2])
		X=X.transpose()
		SXX=np.dot(X.T , X)
		SXY=np.dot(X.T , y0)
		coeff=np.dot(np.linalg.inv(SXY), [SXY])
		ymodel=np.dot(X,coeff)
		varYm=np.var(ymodel)
		R2=varYm/varY
		print "factor: %.3f ==> R2: %.3f%%" % (f, 100*R2)

def test_regression():
	nn=1000
	xx=np.linspace(-4, 0.2, nn)
	yy=inv_slog10(-0.3*xx+0.2*xx**2+0.3*np.random.randn(nn))
	#############################
	X=np.array( [np.ones(nn), xx, xx**2]).T
	Sxx=np.dot(X.T, X)
	lyy=slog10(yy)
	Sxy=np.dot(X.T, lyy)
	coeff=np.linalg.lstsq(Sxx, Sxy)[0]
	coeff_str=" ".join(["{0:.3f}".format(cc) for cc in coeff])
	lyy_m=np.dot(X, coeff)
	lR2=np.var(lyy_m)/np.var(lyy)
	
	R2=1.0-np.var(yy-inv_slog10(lyy_m))/np.var(yy)
			
	Sxy=np.dot(X.T, yy)
	coeff=np.linalg.lstsq(Sxx,Sxy)[0]
	y_m=np.dot(X, coeff)
	R2_orig=np.var(y_m)/np.var(yy)
	
	plt.subplot(121)
	plt.plot(xx,lyy,"ko", xx, zip(lyy_m, slog10(y_m)),"-", ms=4, lw=2, mew=0)
	plt.axhline(0, -4 , 5, color="#ababab", ls="dashed")
	plt.title("coeff: {0}\nR2={1:.1f}%".format(coeff_str, R2*100))
	
	plt.subplot(122)
	plt.plot(xx,yy, "ko", xx, zip(inv_slog10(lyy_m),y_m),"-", ms=4, lw=2, mew=0)

	plt.axhline(0, -4 , 5, color="#ababab", ls="dashed")
	plt.title("R2={0:.1f}% (orig: {1:.1f}%)".format(R2*100, R2_orig*100))
	
	plt.show()
		

def test_3d():
	xx=np.linspace(-5,5,50)
	yy=np.linspace(-5,5,50)
	xxi,yyi=np.meshgrid(xx,yy)
	zzi=0.7*xxi**2-0.8*yyi**2+0.5*xxi*yyi
	fig=plt.figure()
	ax=fig.add_subplot(111, projection='3d')
	ax.plot_wireframe(xxi,yyi,zzi,  cstride=1, rstride=1, lw=0.2, color='b')
	ax.plot_wireframe(xxi,yyi,-zzi, cstride=1, rstride=1, lw=0.2, color='r')
	plt.show()
	
	
	
def test_plot():
	fig1=plt.figure()
	fig2=plt.figure()
	xx=np.linspace(-3,3,100)
		
	ax11=fig1.add_subplot(111)
	ax11.plot(xx,np.sin(xx),"r--")
	ax21=fig2.add_subplot(111)
	ax21.plot(xx,np.cos(xx),"b--")
	
	ax11=fig1.add_subplot(111)
	ax11.plot(xx,np.sin(xx+0.2),"r--")
	ax21=fig2.add_subplot(111)
	ax21.plot(xx,np.cos(xx+0.2),"b--")
	
	plt.show()
	
def brownian_test():
	a,b,rho = 1,2, 0.95
	Npaths=500
	Nsteps=100
	bm=[np.cumsum(np.random.randn(Nsteps,Npaths),axis=0) for i in xrange(3)]
	fig=plt.figure()
	nrows, ncols, pidx=2,3,0
	axlist=[ fig.add_subplot(nrows, ncols, idx) for idx in [1,2,3,4,5,6]]
	
	ax=axlist[pidx]; pidx += 1
	ax.plot(bm[0],"k-",bm[1],"b-",bm[2],"r-")
	
	ax=axlist[pidx]; pidx += 1
	for paths,cc in zip(bm,["k","b","r"]):
#		for vals in paths:
		normplot(paths,"-", axes=ax, ms=3, mew=0, color=cc)

	ax=axlist[pidx]; pidx += 1
	for BV,cc in zip(bm,["k","b","r"]):
		xylst=[]
		for vals in BV:
			xylst.append([np.average(vals), np.std(vals)])
		ax.plot(range(Nsteps), xylst,'-', color=cc,ms=2, mew=0)			
	
	ax=axlist[pidx]; pidx += 1
	A, B = (a+rho*b), b*np.sqrt(1-rho**2)
	BV1=A*bm[0]+B*bm[1]
	ax.plot(BV1,"-")
	
	
	ax=axlist[pidx]; pidx += 1
	for vals in BV1:
		normplot(vals,"k-", axes=ax, ms=3, mew=0)
	BV2= np.sqrt(A**2+B**2)*bm[2]                  # W=aX+bY ==> Var(W)=(a*Stdev(X))**2+(b*Stdev(Y))**2+2*a*b*rho*Std(X)*Std(Y)=(a**2+b**2+2*a*b)*Var(Z)
	for vals in BV2:
		normplot(vals,"r-", axes=ax, ms=3, mew=0)
		
	ax=axlist[pidx]; pidx += 1
	xylst=[]
	for vals in BV1:
		xylst.append([np.average(vals), np.std(vals)])
	ax.plot(range(Nsteps), xylst,'-')
	
	xylst=[]
	for vals in BV2:
		xylst.append([np.average(vals), np.std(vals)])
	ax.plot(range(Nsteps), xylst,'o',ms=2, mew=0)	
	
	plt.show()
	
def Hull_Whilte_1f():
	"""
	dr(t)=(theta(t)-alpha*r(t))dt+sigma*dWt
	"""
	dT, 	theta, 	alpha, 	sigma, 	r0 = \
	0.1, 	0.04, 	0.7, 	0.009, 	0.04
	
	Npaths, Nsteps = 500, 100
	
	fig=plt.figure()
	Nplots=5
	rr,cc=findRowCol(Nplots)
	axlist=[ fig.add_subplot(rr,cc, idx) for idx in [1,2,3,4,6,5]]
	pidx=0
	
	## brownian motion
	ti=np.cumsum(np.ones(Nsteps)*dT)
	dWt=np.random.randn(Nsteps,Npaths)*np.sqrt(dT); dWt[0,:]=0
	Wt=np.cumsum(dWt, axis=0)
	
	# Std BM
	ax=axlist[pidx]; pidx += 1
	ax.plot(ti, Wt)
	ax.set_title("Standard BM")

	# Normality check
	ax=axlist[pidx]; pidx += 1
#	for vals in Wt:
#		normplot(vals,  "-", axes=ax)
	normplot(Wt,"-", axes=ax)
	ax.set_title("Normality Check")
	
	## Spot rate
	rt_avg, rt_stdev=[r0],[0.]
	rt=dWt
	rt[0,:]=r0
	for tt in xrange(1,Nsteps):
		rt[tt,:]=rt[tt-1,:]+(theta-alpha*rt[tt-1,:])*dT+sigma*dWt[tt,:]
		rt_avg.append(np.average(rt[tt,:]))
		rt_stdev.append(np.std(rt[tt,:]))
	ax=axlist[pidx]; pidx += 1
	ax.plot(ti, rt, "-")
	ax.plot(ti, rt_avg, "yo-", ms=3, mew=0.1, mec="#ababab")
	ax.set_title("Spot rate: paths & sample average")

	# Spot rate: Avg vs. Theory
	ax=axlist[pidx]; pidx += 1
	rt_avg_th=np.exp(-alpha*ti)*r0+theta/alpha*(1-np.exp(-alpha*ti))
	rt_stdev_th=sigma*np.sqrt((1-np.exp(-2.*alpha*ti))/(2.*alpha))
	ax.plot(ti, rt_avg, "o", ms=3, mew=0)
	ax.plot(ti, rt_avg_th,"-")
	ax.set_title("Average vs. Theory")
	
	# Spot rate: Stdev vs. Theory
	ax=axlist[pidx]; pidx += 1
	ax.plot(ti, rt_stdev, "o", ms=3, mew=0)
	ax.plot(ti, rt_stdev_th,"-")
	ax.set_title("Stdev vs. Theory")
	
	# Bond price
	lnP=np.copy(rt)*dT
	lnP=np.cumsum(lnP[::-1,:], axis=0)[::-1,:]
	lnP -= (rt+rt[-1,:])*dT/2.
	P=np.exp(-lnP)
	aP=np.average(P,axis=1)
	ax=axlist[pidx]; pidx += 1
	ax.plot(ti, P, "-")
	ax.plot(ti, aP, "yo", ms=3, mew=0)
	##>> bond price in theory
	Bi=(1.-np.exp(-alpha*(ti[-1]-ti)))/alpha
	ax.set_title("Bond price")
	ax.set_ylim(0,1.)
	
	
	## show the plot	
	plt.show()   

import threading

th_global=[]

def print_num(wfd):
	sys.stdout.flush()
	o_wfd=os.dup(1)
	os.dup2(wfd, 1)
	for i in xrange(5):
		ss=str(i)+","
		os.write(1,ss)
		print ss
#		time.sleep(0.1)
#	print eofString
#	sys.stdout.flush()
	os.dup2(o_wfd,1)
	os.close(wfd) 
	os.close(o_wfd)
		
def collect_num(rfd):
	th_str=""
	while True:
		ss=os.read(rfd,1024)
		if ss=="": break
#		if ss[-1]==eofString: break
		th_global.append(ss)
	
def test_thread():
	rfd,wfd=os.pipe()
	th=threading.Thread(target=print_num, args=(wfd,))
	th.start()
	thc=threading.Thread(target=collect_num, args=(rfd,))
	thc.start()
	
	th.join(); thc.join()
	
	th_str="".join(th_global)
	print "Collected string <%s>" % th_str

import threading
def thr_wrapper(func, *args, **kwargs):
    rfd, wfd=os.pipe()
    rfd2, wfd2=os.pipe()
    th_res=[]
    th2_res=[]
    thlst=[]
    pc=threading.Thread(target=__pipe_collector__, args=(rfd, th_res)); pc.start(); thlst.append(pc)
    pc2=threading.Thread(target=__pipe_collector__, args=(rfd2, th2_res)); pc2.start(); thlst.append(pc2)
    __wrapper__( (wfd,wfd2), func, args, {} ); 
    for th in thlst:
        th.join()
    print "*** Output ***"
    print "".join(th_res)
    print "*** Err ***"
    print "".join(th2_res)

def thr_test():
   thr_wrapper(PrintSomething, "Thread", 1)

    
def new_MC():
    """
    dr(t)=(theta(t)-alpha*r(t))dt+sigma*dWt
    """
    dT,     theta,     alpha,     sigma,     r0 = \
    0.1,     0.04,     0.7,     0.009,     0.04
    
    Npaths, Nsteps = 500, 30
    
    fig=plt.figure()
    plot_lst=[1,2,3,4]; Nplots=len(plot_lst)
    rr,cc=findRowCol(Nplots)
    axlist=[ fig.add_subplot(rr,cc, idx) for idx in plot_lst]
    pidx=0
    
    ## brownian motion
    ti=np.cumsum(np.ones(Nsteps)*dT)
    dWt=np.random.randn(Nsteps,Npaths)*np.sqrt(dT); dWt[0,:]=0
    Wt=np.cumsum(dWt, axis=0)
    
    # Std BM
    ax=axlist[pidx]; pidx += 1
    ax.plot(ti, Wt)
    ax.set_title("Standard BM", fontsize=12)

    # Normality check
    ax=axlist[pidx]; pidx += 1
    normplot(Wt,"-", axes=ax)
    ax.set_title("Normality Check", fontsize=12)
    
    # Feynmann paths
    ax=axlist[pidx]; pidx += 1
    
    FWt=2*(np.random.rand(Nsteps, Npaths)-0.5); FWt[0,:]=0.0
    for ii in xrange(Nsteps):
        FWt[ii,:] *= 5*np.sqrt(ti[ii])
    
    dFWt=np.diff(FWt, axis=0)
#    Wpath=np.exp(-0.5*np.sum(dFWt**2/dT, axis=0))
    Wpath=-0.5*np.sum(dFWt**2/dT, axis=0)
    Wpath -= np.average(Wpath)
    Wpath = np.exp(Wpath)
    Wpath /= np.sum(Wpath)
    
    ax.plot(ti, FWt)
    ax.set_title("Feynman Paths", fontsize=12)
    
    # Reproduction of BM
    ax=axlist[pidx]; pidx += 1
#    normplot(FWt[-1,:], "-", axes=ax, freq=Wpath)
    ax.plot(FWt[-1,:], Wpath,"o", ms=3, mew=0, axes=ax)
#    ax.plot(FWt[-1,:],"o-", axes=ax)
    ax.set_title("Normal reproduction", fontsize=12)
    
    # Show plots
    plt.show()

class RegRandn:
    def __init__(self, npaths, nfactors, siglevel):
        self.npaths = npaths
        self.nfactors = nfactors
        self.siglevel = siglevel
        self.cum = np.zeros( (nfactors,npaths) )
        self.tidx = 0
    def next(self):
        self.tidx += 1
        arr = np.random.randn(self.nfactors,self.npaths)
#        cum_t = self.cum + arr
#        msk = ((np.abs(cum_t) / np.sqrt(self.tidx)) >= self.siglevel)
        self.cum += arr
        msk = ((np.abs(self.cum) / np.sqrt(self.tidx)) >= self.siglevel)
        arr[msk] *= -1.0
        self.cum[msk] += 2*arr[msk]
        return arr
        
def test_regrandn():
    npaths = 500
    ntimes = 200
    nfactors = 3
    siglevel = 3.0
    rndG = RegRandn(npaths, nfactors, siglevel)
    
    arr = np.zeros( (nfactors,ntimes, npaths) )
    for tidx in xrange(ntimes):
        arr[:,tidx,:] = rndG.next()
    
    sigma = 0.6
    dT = 0.1
    times = (np.arange(ntimes)+1.0)*dT
    sigDT = sigma*np.sqrt(dT)
    diff = np.cumsum(arr*sigDT, axis=1)
    
    nrows=nfactors
    ncols=3
    
    plt.figure()
    pCnt=0
    
    for fidx in xrange(nfactors):
        darr = diff[fidx,:,:]
        pCnt += 1
        plt.subplot(nrows,ncols,pCnt)
        plt.plot(times, darr, "o-", ms=2, mew=0.1)
        
        pCnt += 1
        plt.subplot(nrows,ncols,pCnt)
        plt.plot(times, (darr.T/(sigma*np.sqrt(times))).T, "o-", ms=2, mew=0.1)
        plt.ylim(-4,4)
        
        pCnt += 1
        plt.subplot(nrows,ncols,pCnt)
        normplot(darr,"o-", ms=2, mew=0.1)
    
    		
if __name__ == '__main__':
    pass
    test_regrandn()
#    thr_test()
#	test_thread()
#	test_cmap()
#	testPipe()
#	testIPC()	
#	testplot()
#	plotPolar()
#	testerr()
#	testTeX()
#	testPatch()
#	testPotential()
#	testEfield()
#	testConductor()
#    test_mandelbrot()
#    test_newton()
#    test_zTrans()
#	elapsedTime()
#	testTuple()
#	test_super()
#	test_mp()
#	test_mp2()
#	test_StringIO()
#	test_mgr()
#	test_queue()
#	test_fileIO()
#	test_svd()
#	test_regression()
#	test_3d()
#	test_plot()
#	brownian_test()
#	Hull_Whilte_1f()
#    new_MC()
    plt.show()
