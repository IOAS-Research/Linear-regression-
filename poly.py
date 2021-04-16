#examples/fit/poly.py
#performs least-square fit with polynomial function - applied to a star evolution example
import numpy as np
import numpy.random as rnd
from numpy.linalg import solve
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 15})

def data_set(): 
    fname="./evol_120msun_scattered.dat"
    time, mass=np.genfromtxt(fname,dtype=float, comments="#", usecols=(0,1),unpack=True)
    return time, mass

# a values in the slide
def poly_fit(x,y,m):
    A=np.zeros([m+1,m+1],float)
    b=np.zeros([m+1],float)

    for j in range(0,m+1):
        #this is the loop like i+=i
        #bk=sum(xy) slide 18
        b[j]=sum(x**j*y)
        for k in range(0,m+1):
            A[j][k]=sum(x**(j+k))
# this is a making matrix
#using matrix solving we can find aj values as mentioned in slide 18
#A(jk)a(j)=b(k) need to find a(j)
    k=solve(A,b)
   
    return k


def least_square(x,y,k,m):

    N=len(x)
    ynew=np.zeros([N],float)

    for i in range(N):
        temp=0.0
        for j in range(len(k)):
            
            # we use minimizatin value as mentioned in slide 17
            temp+=(k[j]*x[i]**j)
        ynew[i]=temp
            

    res = sum((y-ynew)*(y-ynew))

    sy = np.sqrt(res/(N-m))
   # sa= sy * np.sqrt(sx2/D)
   # sb= sy * np.sqrt(N/D)

    return ynew, res



#main
m=3 #order of polynomial
x,y=data_set()

k=poly_fit(x,y,m)
ynew3,res3=least_square(x,y,k,m)

m=7 #order of polynomial

k=poly_fit(x,y,m)
ynew7,res7=least_square(x,y,k,m)


a=plt.scatter(x,y,marker="*",color="blue")
b,=plt.plot(x,ynew3,linestyle="-",color="red",linewidth="1")
c,=plt.plot(x,ynew7,linestyle="-",color="black",linewidth="1")
plt.ylabel("Mass [M$_\odot$]")
plt.xlabel("time [Myr]")
plt.legend([a,b,c],["Data","Poly Fit $3$", "Poly Fit $7$"])

plt.tight_layout()
plt.show()
