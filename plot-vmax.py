#!/usr/bin/env python3
# Plot the vmax
# @author patrickdiehl@lsu.edu
# @author serge.prudhomme@polymtl.ca
# @date 03/02/2021
import numpy as np
import sys 
import matplotlib.pyplot as plt
import matplotlib

pgf_with_latex = {"text.usetex": True, "font.size" : 12, "pgf.preamble" : [r'\usepackage{xfrac}'] }

a = 1
b = 2
c = 3
delta = 1/8 
lam = 24

def vm(x):
    if x < a :
        return (lam*delta*delta)/24 *(b-a) * x
    if x < b :
        return   (lam*delta*delta)/48 * (b*b-a*a-(b-x)*(b-x))
    if x < c :
        return   (lam*delta*delta)/48 * (b*b-a*a)

def vd(x):
    if x < a :
        return (lam*delta*delta/48) * (b-a)/c * (2*c-(a+b)) *x
    if x < b :
        return  (lam*delta*delta/48) * ( (b*b-a*a ) * (c-x) /c -(b-x)*(b-x) )
    if x < c :
        return (lam*delta*delta/48) * (b*b-a*a) * (c-x) / c



x = np.linspace(0,3,300)

y = []

for xi in x:
    y.append(vm(xi))

plt.plot(x,y,c="black")

plt.ylabel("v(x)")
       
plt.title("")
#plt.legend()
plt.grid()
plt.xlabel("$x$")

plt.savefig("vmax-neumann.pdf",bbox_inches='tight')

plt.clf()

y = []

lam=128/7
for xi in x:
    y.append(vd(xi))

plt.plot(x,y,c="black")

plt.ylabel("v(x)")
       
plt.title("")
#plt.legend()
plt.grid()
plt.xlabel("$x$")

plt.savefig("vmax-dirichlet.pdf",bbox_inches='tight')



