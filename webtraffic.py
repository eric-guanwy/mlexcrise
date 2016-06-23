#!/usr/bin/python

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
#import timeit

data = sp.genfromtxt("/home/eric/machinelearningwork/BuildingMachineLearningSystemswithPython/web_traffic.tsv")
x0 = data[:,0]
y0 = data[:,1]

x=x0[~np.isnan(y0)]
y=y0[~np.isnan(y0)]
plt.scatter(x,y)
ad = np.arange(x[-1]+1,1000)
xx = np.append(x,ad)

def error(f,x,y):
	return sp.sum((f(x)-y)**2)

def errorprint(f,x,y):
	print "error for %dd fit:\t%f" %(f.order,error(f,x,y))

inflection = 3.5*7*24


fx = sp.linspace(0,xx[-1],1000)
#legendtxt = []

x=x[inflection:]
y=y[inflection:]

fp1,resdiuals,rank,sv,rcond=sp.polyfit(x,y,1,full=True)
f1 = sp.poly1d(fp1)

plt.plot(fx,f1(fx),linewidth=3,color='green',ls=':',label='d=%i' %f1.order)
#plt.legend(["d=%i" %f1.order],loc="upper left")
#legendtxt.append("d=%i" %f1.order)


f2p=sp.polyfit(x,y,2)
f2=sp.poly1d(f2p)

plt.plot(fx,f2(fx),linewidth=3,color='brown',ls='-',label='d=%i' %f2.order)
#plt.legend(["d=%i" %f2.order],loc="upper left")
#legendtxt.append("d=%i" %f2.order)


f3p = sp.polyfit(x,y,3)
f3 = sp.poly1d(f3p)

plt.plot(fx,f3(fx),linewidth=3,color='red', linestyle='dashed',label='d=%i' %f3.order)
#plt.legend(['r'],["d=%i" %f3.order],loc="upper left")
#legendtxt.append("d=%i" %f3.order)
f10p = sp.polyfit(x,y,10)
f10 = sp.poly1d(f10p)

plt.plot(fx,f10(fx),linewidth=3,color='blue', linestyle='dashed',label='d=%i' %f10.order)


f100p,resdiuals,rank,sv,rcond=sp.polyfit(x,y,100,full=True)
f100 = sp.poly1d(f100p)

plt.plot(fx,f100(fx),linewidth=3,color='red',label='d=%i' %f100.order)

x=x0[~np.isnan(y0)]
y=y0[~np.isnan(y0)]
x=x[:inflection-1]
y=y[:inflection-1]

errorprint(f1,x,y)
errorprint(f2,x,y)
errorprint(f3,x,y)
errorprint(f10,x,y)
errorprint(f100,x,y)


plt.legend(loc="upper left")
#plt.scatter(x,y)
plt.title("Web traffic over the last month")
plt.xlabel("time")
plt.ylabel("hits/hour")
plt.xticks([w*7*24 for w in range(7)],["week %i" %w for w in range(7)])
#plt.autoscale()
plt.ylim(0,10000)
plt.grid()
plt.show()
"""

inflection = 3.5*7*24

xa = x[:inflection]
ya = y[:inflection]

fap=sp.polyfit(xa,ya,1)
fa = sp.poly1d(fap)

xb = x[inflection:]
yb = y[inflection:]
fbp = sp.polyfit(xb,yb,1)
fb  = sp.poly1d(fbp)

dota = sp.linspace(0,xa[-1],1000)
dotb = sp.linspace(xa[-1],xb[-1],1000)

plt.plot(dota,fa(dota),linewidth=4,color='red',ls='dashed',label='d=%i' %fa.order)
plt.plot(dotb,fb(dotb),linewidth=4,color='blue',ls='-',label='d=%i' %fb.order)

plt.legend(loc='upper left')
plt.scatter(x,y)
plt.title("Web traffic over the last month")
plt.xlabel("time")
plt.ylabel("hits/hour")
plt.xticks([w*7*24 for w in range(10)],["Week %i" %w for w in range(10)])
plt.autoscale()
plt.grid()
plt.show()

"""