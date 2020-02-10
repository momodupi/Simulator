# import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

# from matplotlib import cm
# from matplotlib.ticker import LinearLocator, FormatStrFormatter
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
# from matplotlib.pyplot import figure

import numpy as np
import scipy.stats as sps
from scipy.optimize import fsolve

import scipy.io as sio

import time

class Pricing(object):
    def __init__(self, para):
        self.p1 = 0.01
        self.p2 = 0.01

        self.l1, self.l2 = para.get('l')
        self.c1, self.c2 = para.get('c')
        self.m1, self.m2 = para.get('m')
        self.v1, self.v2 = para.get('v')

    def F1(self, c):
        # return (np.exp(-c))
        # f1 = 1-sps.norm(self.c1, 2).cdf(c)
        return (1-sps.norm(self.m1, self.v1).cdf(c))
        # return (1 - c/(self.c1+self.c2))

    def F2(self, c):
        # return (np.exp(-c))
        # f2 = 1-sps.norm(self.c2, 2).cdf(c)
        return (1-sps.norm(self.m2, self.v2).cdf(c))
        # return (1 - c/(self.c1+self.c2))

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

    def solver(self, lvl):
        self.p1 = fsolve(self.Phi1, lvl)
        self.p2 = fsolve(self.Phi2, lvl)

        c = self.c1 + self.c2

        if (1-self.F1(c) - (1-self.F1(self.p1)) >= self.l2/self.l1*(self.F2(c))):
            self.p1 = np.nan
            self.p2 = np.nan

        if (self.F1(self.p1) == 0) or (self.F2(self.p2) == 0):
            return np.nan, np.nan
        else:
            return self.p1, self.p2

        
        

def simu(fnum, para, s, lvl):
    '''
    print('Input cost 1 and cost 2')
    c = input()
    c1 = int(c.split()[0])
    c2 = int(c.split()[1])
    '''
    l1, l2 = para.get('l')
    c1, c2 = para.get('c')
    m1, m2 = para.get('m')
    v1, v2 = para.get('v')

    # Make data.
    X = np.arange(1, 100, s)
    Y = np.arange(1, 100, s)    
    X, Y = np.meshgrid(X, Y)
    samplesize = len(X)

    Zl1 = np.zeros([samplesize, samplesize])
    Zl2 = np.zeros([samplesize, samplesize])
    # Ztri = np.zeros([samplesize, samplesize])

    Zm1 = np.zeros(Zl1.shape, dtype=bool)
    Zm2 = np.zeros(Zl2.shape, dtype=bool)

    for x in range(samplesize):
        for y in range(samplesize):
            
            para['l'] = (X[0][x], X[0][y])
            if (X[0][x] >= X[0][y]):
                p = Pricing(para)
                Zl1[y][x], Zl2[y][x] = p.solver(lvl)

                a1 = p.l1 * p.F1(p.p1)
                a2 = p.l2 * p.F2(p.p2)
                cc = p.p1 * a1 + p.p2 * a2
                
                print('l1={}, l2={}, c1={}, c2={}, m1={}, m2={}, v1={}, v2={}'.format(p.l1,p.l2,p.c1,p.c2,p.m1,p.m2,p.v1,p.v2))
                print('total={}, a1={}, a2={}, F1={}, F2={}'.format(cc, a1, a2, p.F1(p.p1), p.F2(p.p2)))
                print('P1={}, P2={}'.format(p.p1, p.p2))
                print(".")

            else:
                Zl1[y][x], Zl2[y][x] = np.nan, np.nan
                # Zl1[y][x], Zl2[y][x] = 0, 0
                Zm1[y][x], Zm2[y][x] = np.ma, np.ma

            



    # Plot the surface.
    fig = plt.figure(figsize=(12, 6))
    grid = ImageGrid(fig, 110,
                 nrows_ncols=(1,2),
                 axes_pad=0.3,
                 share_all=True
                 )
    
    # im = grid[0].imshow(Zl1, cmap='gray')
    # im = grid[0].imshow(Zl1, cmap='gray', vmin=np.nanmin(Zl1)-2, vmax=np.nanmax(Zl1)+2)
    
    im = grid[0].contour(X, Y, Zm1, 0, colors='black', linewidths=0.75)
    im = grid[0].contour(X, Y, Zl1, colors='black', linewidths=0.75)
    plt.clabel(im, inline=True, fontsize=8)

    grid[0].set_xlabel('High')
    grid[0].set_ylabel('Low')
    grid[0].set_title('Price: High Demand Area')
    
    # im = grid[1].imshow(Zl2, cmap='gray')
    # im = grid[1].imshow(Zl2, cmap='gray', vmin=np.nanmin(Zl1)-2, vmax=np.nanmax(Zl1)+2)

    im = grid[1].contour(X, Y, Zm2, 0, colors='black', linewidths=0.75)
    im = grid[1].contour(X, Y, Zl2, colors='black', linewidths=0.75)
    plt.clabel(im, inline=1, fontsize=10)

    grid[1].set_xlabel('High')
    grid[1].set_ylabel('Low')
    grid[1].set_title('Price: Low Demand Area')

    plt.clabel(im, inline=1, fontsize=10)

    # grid[1].cax.colorbar(im)
    # grid[1].cax.toggle_label(True)

    
    # fig = plt.figure(num = fnum, figsize=(8,6), dpi=300)
    # x, y = X.flatten(), Y.flatten()
    
    # ct1 = ax1.contourf(X, Y, Zl1, cmap='gray', alpha=0.7)
    # fig.colorbar(ct1)
    # ct2 = ax2.contourf(X, Y, Zl2, cmap='gray', alpha=0.7)
    # fig.colorbar(ct2)

    # im = ax1.imshow(Zl1, cmap='gray', vmin=np.nanmin(Zl1), vmax=np.nanmax(Zl1))
    # im = ax2.imshow(Zl2, cmap='gray', vmin=np.nanmin(Zl1), vmax=np.nanmax(Zl1))
    # fig.subplots_adjust(right=1)
    # fig.colorbar(im)

    '''
    ax.plot_surface(X, Y, Zl1, rstride=1, cstride=1, alpha=0.7,
                    cmap='Reds', linewidth=0.1, edgecolor='r', vmin=np.nanmin(Zl1), vmax=np.nanmax(Zl1))
    
    ax.plot_surface(X, Y, Zl2, rstride=1, cstride=1, alpha=0.7,
                    cmap='Blues', linewidth=0.1, edgecolor='b', vmin=(c1+c2)/3, vmax=(c1+c2)/2)

    
    ax.plot_surface(X, Y, Ztri, rstride=1, cstride=1, alpha=0.2,
                    cmap='Greens', linewidth=0.1, edgecolor='g', vmin=np.nanmin(Zl2), vmax=c1+c2)
    '''
    # ax.text(0, 0, np.nanmax(Zl1), 'High', color='red')
    # ax.text(0, 0, np.nanmax(Zl2), 'Low', color='blue')

    # Customize the z axis.
    # ax.set_zlim(np.nanmin(Zl2), c1+c2)
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    '''
    ax1.set_xlabel('High')
    ax1.set_ylabel('Low')
    ax1.set_title('Price at high demand area')
    ax2.set_xlabel('High')
    ax2.set_ylabel('Low')
    ax2.set_title('Price at low demand area')
    '''

    # ax.set_zlabel('Price')
    # plt.title('Pricing Policy Distribution')

    # plt.show()
    # ax.view_init(90, 120)
    # plt.draw()
    # plt.imshow(Zl1, cmap='gray')
    
    plt.savefig('{}.png'.format(fnum), dpi=300)

    data = {
        "l": (l1,l2),
        "c": (c1,c2),
        "m": (m1,m2),
        "v": (v1,v2),
        "x": X,
        "y": Y,
        "Zl1": Zl1,
        "Zl2": Zl2,
        "Zm1": Zm1,
        "Zm2": Zm2
    }
    sio.savemat('{}.mat'.format(fnum), data)


def main():
    para = {
        'l': (0, 0),
        'c': (15, 15),
        'v': (15, 15),
        'm': (25, 25)
    }

    lvl = 0
    filename = 0
    
    para = {
        'l': (0, 0),
        'c': (15, 15),
        'v': (15, 15),
        'm': (25, 25)
    }


    start_time = time.time()
    # lvl = para.get('m')[0]
    lvl = 0
    simu(filename, para, 1, lvl)

    print('--{} seconds--'.format(time.time()-start_time))
    
    '''
    for i in range(-5, 5, 2):
        para_b = para
        para_b['m'] = (para['m'][0]+i,para['m'][0]+i)
        simu(filename, para_b, 1, lvl)
        filename = filename + 1
    

    for i in range(0, 20, 5):
        para_b = para
        para_b['v'] = (para_b['v'][0]+i,para_b['v'][1]+i)
        simu(filename, para, 1, lvl)
        filename = filename + 1
    
    for i in range(0, 20, 5):
        para_b = para
        para_b['c'] = (para_b['c'][0]+i,para_b['c'][1]+i)
        # m = (para['c'][0]+para['c'][1])/2
        # para['m'] = (m,m)
        simu(filename, para, 1, lvl)
        filename = filename + 1
    '''
    
    

if __name__ == "__main__":
    main()
