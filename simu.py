import numpy as np
import scipy.stats as sps
from scipy.optimize import fsolve

class Pricing(object):
    def __init__(self, l1, l2, c1, c2):
        self.l1 = l1
        self.l2 = l2
        self.c1 = c1
        self.c2 = c2
        self.p1 = 0.01
        self.p2 = 0.01

        self.meanc = (c1+c2)/2

    def F1(self, c):
        return (1-np.exp(-100*c))
        # return sps.norm(self.meanc, 2).cdf(c)

    def F2(self, c):
        return (1-np.exp(-10*c))
        # return sps.norm(self.meanc, 5).cdf(c)

    def L(self):
        return (self.l1 * self.F1(self.c1+self.c2) - self.l2 * self.F2(self.c1+self.c2))

    def Phi1(self, q):
        x = q
        return ( -self.L() + ( 2*x / (self.c1+self.c2) -1 )* self.l1 * self.F1(x) )

    def Phi2(self, q):
        x = q
        return ( self.L() + ( 2*x / (self.c1+self.c2) )* self.l2 * self.F2(x) - self.l1 * self.F1(self.p1) )

    def Trival(self, q):
        x = q
        return ( (self.c1 + self.c2) * self.l1 * self.F1(x) / (self.l1 * self.F1(x) + self.l2 * self.F2(x)) - x )

    def solver(self):
        self.p1 = fsolve(self.Phi1, self.c1+self.c2)
        self.p2 = fsolve(self.Phi2, self.c1+self.c2)
        print(self.p1, self.p2)

        print( fsolve(self.Trival, self.c1+self.c2) )
        


l1 = 8
l2 = 5
c1 = 60
c2 = 40
p = Pricing(l1, l2, c1, c2)
p.solver()