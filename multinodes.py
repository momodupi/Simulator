import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape
from scipy.stats import halfnorm, norm
from scipy.special import binom
from scipy.optimize import fsolve, brentq, least_squares
from cvxopt import matrix, solvers
from sympy import solve, Poly, Eq, Function, exp 

import resource, sys
import pickle

from itertools import chain, combinations, product
from copy import copy, deepcopy
from multiprocessing import Pool, Queue, Manager


def powerset(iterable):
    s = list(iterable)
    return list( chain.from_iterable(combinations(s, r) for r in range(len(s)+1)) )
# SS = list(powerset(road_set))
# print(list(SS))


def psi(a, c, t, N):
    beq = np.zeros(N)
    Aeq = np.zeros(shape=(N,N**2))
    for i in range(N):
        beq[i] = np.sum( a[i,:] ) - np.sum( a[:,i] )
        
        Aeq_i = np.zeros(shape=(N,N))
        Aeq_i[:,i] = -1
        Aeq_i[i,:] = 1
        Aeq_i[i,i] = 0
        Aeq[i,:] = Aeq_i.flatten('F')
            
    Aub = -np.eye(N**2)
    bub = np.zeros(N**2)

    # solvers.options['show_progress'] = False
    result = solvers.lp(
        # c=matrix(c.flatten('F')*t.flatten('F')), 
        c=matrix((t+2*np.eye(N)).flatten('F')), 
        G=matrix(Aub), 
        h=matrix(bub), 
        A=matrix(Aeq), 
        b=matrix(beq), 
        solver='glpk',
        options={'glpk':{'msg_lev':'GLP_MSG_OFF'}}
    )
    
    res = np.asarray(result['x'])
    return res.reshape((N,N))


def service_rate(R, a, c, t, sigma, N):
    def F(c,r):
        # return r/(self.C+0.01)
        rv = halfnorm(c, sigma)
        # rv = norm( c+5, sigma )
        return rv.cdf(r)
    # print(R,a)
    a = (1-np.vectorize(F)(c, R))*a
    # return a + psi(a, c, t)
    res = a + psi(a, c, t, N)
    # print(res)
    return res

# MVA
def Thpt(M, mu_r, mu_n, pi_r, pi_n, pi):
    L_n = np.zeros(len(mu_n))

    D_r = 1/(mu_r+10e-6)
    TH = 1
    for m in range(M+1):
        D_n = (1 + L_n)/(mu_n+10e-6)
        TH = m/( np.sum(pi_n*D_n) + np.sum(pi_r*D_r) )
        # print(m, np.sum(pi_n*D_n) + np.sum(pi_r*D_r))
        L_n = pi_n*TH*D_n
        # L_r = pi_r*TH*D_r
    
    return TH*pi


def Net_sol(m, R, w, N):
    a = w['a']
    c = w['c']
    t = w['t']
    sigma = w['f']

    mu_ij_n = service_rate( R, a, c, t, sigma, N )
    # print('mu', mu_ij_n)
    if mu_ij_n.sum() == 0:
        res = {
            'TH': np.zeros(N**2),
            'mu_r': np.zeros(N*(N-1)),
            'mu_n': np.zeros(N),
            'pi': np.zeros(N**2)
        }
        return res

    pi_n = mu_ij_n.dot(np.ones(N))
    # p = mu_ij_n / pi_n[:, np.newaxis]
    pi_r = mu_ij_n[t!=-1]
    pi = np.concatenate( (pi_n,pi_r), axis=None )
    pi /= np.sum(pi)
    pi_n = pi[:N] 
    pi_r = pi[N:]
    # print(pi)
    mu_r = 1/t[t!=-1]
    mu_n = mu_ij_n.dot(np.ones(N))
    # print(mu_n, mu_r)


    res = {
        'TH': Thpt(m, mu_r, mu_n, pi_r, pi_n, pi),
        'mu_r': mu_r,
        'mu_n': mu_n,
        'pi': pi
    }
    return res



def cost(m, w, R, N):
    res = Net_sol(m, R, w, N)

    TH = res['TH'][N:]
    return TH.dot( w['c'][w['c']!=-1] )

def v(U, m, w, R, N):
    w_u = deepcopy(w)
    a_u = np.zeros(shape=np.shape(w['a']))
    for u in U:
        i = int(u/10)-1
        j = int(u%10)-1
        a_u[i,j] = w['a'][i,j]
    w_u['a'] = a_u
    # print(U)
    return cost(m, w_u, R, N)





def Sh_ij(args):
    p_set = args['s']
    p = args['p']
    w = args['w']
    R = args['R']
    m = args['m']
    N = args['n']
    
    p_size = len(p_set)
    p_set.remove(p)
    p_p = powerset(p_set)
    # SS_size = len(SS)

    # print(SS, p_set, p)

    Sh = 0
    for u in p_p:
        u_l = list(u)
        coeff = binom(p_size-1, len(u_l))
        Sh += (v(u_l+[p], m, w, R, N) - v(u_l, m, w, R, N))/coeff

    # print('v', v(p_set+[p], m, w, R, N), Sh)
    
    res = {'p': p, 'v': Sh/p_size}
    return res


# res = Net_sol(m, R, w)
# thpt = res['TH']
# thpt_ij = thpt[ node_set.index(13) ]
# print(thpt)
# print(thpt[4:].dot(c[c!=-1].flatten('F')))
# price = Sh/thpt_ij

# print(price)

# buff = {
#     'sh' : [],
#     'th': []
# }



if __name__ == '__main__':

    a = np.array([ [0,0.6,2.1,1.4], [0,0,3.6,1.2], [0,0,0,0], [0,0,1,0] ])
    t = np.array([
            [-1,15,21,20],
            [14,-1,12,15],
            [21,12,-1,12],
            [19,15,12,-1]
        ])

    c = np.array([
            [-1,12.4,16.5,16.9],
            [8.5,-1,5.2,8.5],
            [16.2,5.1,-1,5],
            [17,9.6,3.9,-1]
        ])

    # print(arr)
    nei_set = [1,2,3,4]
    # node_com = combinations(node_set, 2)
    node_product = product(nei_set, nei_set)
    road_set = [ 10*n[0]+n[1] for n in node_product if n[1]!=n[0] ]
    node_set = nei_set+road_set


    N = len(nei_set)
    players = [12,13,14,23,24,43]
    m = 100
    sigma = 10

    w = {
        'a': a,
        'f': sigma,
        'c': c,
        't': t
    }
    R_init = np.ones(len(players))

    def phi(m, w, R, N):
        R_s = np.zeros(shape=(N,N))
        for i,p in enumerate(players):
            R_s[int(p/10)-1, int(p%10)-1] = R[i]

        process_pool = [ {'s': players, 'p': i, 'w': w, 'm': m, 'n': N, 'R': R_s} for i in players ]

        with Pool() as pool:
            res = pool.map( Sh_ij, process_pool )
        
        res_SS = Net_sol(m, R_s, w, N)
        Th_SS = res_SS['TH']

        # for i in res:
        #     if i['p'] == 13:
        #         buff['sh'].append(i['v'])
        # buff['th'].append(Th_SS[node_set.index(13)])


        prices =  [ p['v']/(Th_SS[ node_set.index(p['p']) ]+10e-6) for p in res ]
        return np.array(prices)


    def T(R):
        return phi(m, w, R, N)-R

    # players = [12,13,14,23,24,43]


    # x = np.arange(0,10,0.1)

    # for l in x:
    #     w['a'][0,2] = l
    #     res = phi(m, w, R_init, N)
    #     # print(res)

    # y1 = np.array(buff['sh'])
    # y2 = np.array(buff['th'])

    # # ax = plt.plot(x,y1,'-r', x,y2,'-b', x,y1/y2,'-g')
    # # plt.show()

    # res = {
    #     'x': x,
    #     'y1': y1,
    #     'y2': y2
    # }
    # with open(f'res_R1_x20_s10.pickle', 'wb') as pickle_file:
    #     pickle.dump(res, pickle_file, protocol=pickle.HIGHEST_PROTOCOL) 


    # R_f = fsolve(T, R_init)
    # R_f = least_squares(T, R_init, bounds = (0, 100))

    # R_f = phi( w, np.array( [22.14597448, 32.21572539, 31.6702842 , 10.42220426, 13.39419726, 4.51399027] ) )

    # print(R_f)

    sigma_set = [10,15,20,25]
    a_range = np.arange(0,11,0.2)

    for sig in sigma_set:    
        res_a = {}
        w['f'] = sig
        for a_r in a_range:
            w['a'][0,2] = a_r
            R_f = fsolve(T, R_init)
            res_a[a_r] = R_f

        with open(f'p_sigma_{sig}.pickle', 'wb') as pickle_file:
            pickle.dump(res_a, pickle_file, protocol=pickle.HIGHEST_PROTOCOL) 
    
    w['a'] = a

    lambda_set = [2, 4, 6, 8]
    s_range = np.arange(10,26,0.5)

    for lam in lambda_set:
        res_s = {}
        w['a'][0,2] = lam
        for s_r in s_range:
            w['f'] = s_r
            R_f = fsolve(T, R_init)
            res_s[s_r] = R_f

        with open(f'p_lambda_{lam}.pickle', 'wb') as pickle_file:
            pickle.dump(res_s, pickle_file, protocol=pickle.HIGHEST_PROTOCOL) 
        

