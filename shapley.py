import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

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
        self.varc = 10

    def F1(self, c):
        # return (np.exp(-c))
        # f1 = 1-sps.norm(self.c1, 2).cdf(c)
        # return (1-sps.norm(self.meanc, self.varc).cdf(c))
        return (1 - c/(self.c1+self.c2))

    def F2(self, c):
        # return (np.exp(-c))
        # f2 = 1-sps.norm(self.c2, 2).cdf(c)
        # return (1-sps.norm(self.meanc, self.varc).cdf(c))
        return (1 - c/(self.c1+self.c2))

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
        self.p1 = fsolve(self.Phi1, self.meanc)
        self.p2 = fsolve(self.Phi2, self.meanc)

        if (self.F1(self.p1) == 0) or (self.F2(self.p2) == 0):
            return np.nan, np.nan, np.nan
        else:
            # t = fsolve(self.Trival, self.c1+self.c2)
            t = 0
            return self.p1, self.p2, t 

        
        

def main():
    c1 = 15
    c2 = 10
    l1 = 10
    l2 = 5

    print('Input cost 1 and cost 2')
    c = input()
    c1 = int(c.split()[0])
    c2 = int(c.split()[1])

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Make data.
    X = np.arange(1, 100, 5)
    Y = np.arange(1, 100, 5)
    # X = np.arange(10, 20, 0.9)
    # Y = np.arange(10, 20, 0.9)
    
    X, Y = np.meshgrid(X, Y)
    samplesize = len(X)

    Zl1 = np.zeros([samplesize, samplesize])
    Zl2 = np.zeros([samplesize, samplesize])
    Ztri = np.zeros([samplesize, samplesize])

    for x in range(samplesize):
        for y in range(samplesize):
            
            l1 = X[0][x]
            l2 = X[0][y]
            if (l1 >= l2):
                p = Pricing(l1, l2, c1, c2)
                Zl1[x][y], Zl2[x][y], Ztri[x][y] = p.solver()
            else:
                Zl1[x][y], Zl2[x][y], Ztri[x][y] = np.nan, np.nan, np.nan
            '''
            c1 = X[0][x]
            c2 = X[0][y]
            p = Pricing(l1, l2, c1, c2)
            Zl1[x][y], Zl2[x][y], Ztri[x][y] = p.solver()
            '''
            a1 = p.l1 * p.F1(p.p1)
            a2 = p.l2 * p.F2(p.p2)
            cc = p.p1 * a1 + p.p2 * a2
            
            print('l1={}, l2={}, c1={}, c2={}'.format(l1,l2,c1,c2))
            print('total={}, a1={}, a2={}'.format(cc, a1, a2))
            print('P1={}, P2={}'.format(p.p1, p.p2))

            print('T=', fsolve(p.Trival, p.meanc) )
            print(".")

    # Plot the surface.
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Zl1, rstride=1, cstride=1, alpha=0.7,
                    cmap='Reds', linewidth=0.1, edgecolor='r', vmin=np.nanmin(Zl1), vmax=np.nanmax(Zl1))
    
    ax.plot_surface(X, Y, Zl2, rstride=1, cstride=1, alpha=0.7,
                    cmap='Blues', linewidth=0.1, edgecolor='b', vmin=(c1+c2)/3, vmax=(c1+c2)/2)

    '''
    ax.plot_surface(X, Y, Ztri, rstride=1, cstride=1, alpha=0.2,
                    cmap='Greens', linewidth=0.1, edgecolor='g', vmin=np.nanmin(Zl2), vmax=c1+c2)
    '''
    ax.text(0, 100, np.nanmax(Zl1), 'High', color='red')
    ax.text(0, 100, np.nanmax(Zl2), 'Low', color='blue')

    # Customize the z axis.
    # ax.set_zlim(np.nanmin(Zl2), c1+c2)
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    ax.set_xlabel('High')
    ax.set_ylabel('Low')
    ax.set_zlabel('Price')
    plt.title('Pricing Policy Distribution')

    plt.show()
    plt.savefig('{}{}.png'.format(c1, c2), dpi=600)


if __name__ == "__main__":
    main()
