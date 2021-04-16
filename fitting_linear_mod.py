#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 00:10:53 2020

@author: jude
"""

import numpy as np
import matplotlib.pyplot as plt





def data_set():
    x=np.logspace(0,4,num=10000)
    b=-2.0
    a=3.0
           
    y=10.**a * x**b     # <---- Correct function
    x=np.log10(x)
    y=np.log10(y)
    
    sigma=1.
    y=np.random.normal(y,sigma)
    
    return x,y
xs,ys=data_set()


N=len(xs)

      
# method one
       
def lSM(xs,ys):
    num=0.0
    deno=0.0
  
    for i in range(len(xs)):
        num+=ys[i]*(xs[i]-(np.mean(xs)))
        deno+=xs[i]*(xs[i]-np.mean(xs))
    B=num/deno   
    A=np.mean(ys)-(np.mean(xs)*B)
    
    return B,A


B,A=lSM(xs, ys)  

print(B,A)  

#method two

def leastsquaremethod(xs,ys):
    B = (((np.mean(xs)*np.mean(ys)) - np.mean(xs*ys)) /
         ((np.mean(xs)*np.mean(xs)) - np.mean(xs*xs)))
    
    A = np.mean(ys) - B*np.mean(xs)
    
    return B, A

B1,A1 = leastsquaremethod(xs,ys)

print(B1,A1)

def std(xs,ys):
    dy=0.0
    for i in range(1,N):
        dy+=(ys[i]-A1-(B1*xs[i]))**2
    dy=np.sqrt(dy/(N-1))    
    
    return dy

dy=std(xs,ys)

print('std',dy)

#regression_line = [(B*x)+A for x in xs]
regression_line = []
regression_line1 = []
for x in xs:
    regression_line.append((B*x)+A)
    regression_line1.append((B1*x)+A1)


plt.scatter(xs,ys,color='b',label='data')
plt.scatter(xs, regression_line, color='r',marker='x',label='regression line')
plt.plot(xs, regression_line1, color='black',label='regression line1')
plt.legend(loc=3)
plt.savefig('regressin.jpg')
plt.show()



