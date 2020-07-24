from __future__ import print_function
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

from ngsolve import *
from netgen.geom2d import unit_square


import numpy as np
import scipy
import math
import time
import textwrap 

from ngsolve.webgui import Draw

np.random.seed(42)

class ReducedBasis:
    
    def __init__(self, fes, blf, rhs, snap = None):

        self.logging = True    
        
        self.fes = fes
        
        # set (bi-)linear forms and matrices
        self.omega = Parameter(0)
        self.a = BilinearForm(self.fes)
        keys = ['k', 'r', 'm']
        for j in range(len(keys)):
            if blf[j]:
                self.a += self.omega**j*blf[j][0].coef * blf[j][0].symbol
                exec(textwrap.dedent("""
                self.{} = BilinearForm(self.fes)
                self.{} += blf[j]
                self.{}.Assemble()
                """.format(keys[j],keys[j],keys[j])))
                key_tmp = keys[j]

        self.a.Assemble()
        
        self.f = LinearForm(self.fes)
        self.f += rhs
        self.f.Assemble()
                
        # store ainv for better performance
        self.ainv = self.a.mat.Inverse(self.fes.FreeDofs(), inverse="sparsecholesky")

        # initialize grid functions
        self.gfu = GridFunction(self.fes)
        self.drawu = GridFunction(self.fes)

        # temporary ngsolve vectors
        self.__bv_tmp = eval('self.{}.mat.CreateColVector()'.format(key_tmp))
        self.__bv_tmp2 = eval('self.{}.mat.CreateColVector()'.format(key_tmp))
        self.__gf_tmp = GridFunction(self.fes)
        
        
        self.proj = Projector(self.fes.FreeDofs(), True)

        # compute norm of f
        self.__bv_tmp.data = self.proj*self.f.vec
        self.f.vec.data = self.__bv_tmp.data
        self.__normf = Norm(self.__bv_tmp)**2
        
        # initialize reduced matrices
        self.k_red = None
        self.m_red = None
        self.r_red = None

        try:
            self.addSnapshots(snap)
        except:
            if self.logging: print("no snapshots given")
            
            

    def addSnapshots(self, new_snapshots):
        
        # set snapshots and check that they do not already exist
        try:
            new_snapshots = np.unique(new_snapshots)
            new_snapshots = new_snapshots[np.array([s not in self.__snapshots for s in new_snapshots])]
            if self.logging: print("append snapshots with {}".format(new_snapshots))
            self.__snapshots = np.append(self.__snapshots, new_snapshots)
            
        except:
            if self.logging: print("set snapshots and reset basis")
            self.__snapshots = np.array(new_snapshots)
            self.V = None

        # store smallest and biggest snapshot parameter
        self.omega_min = min(self.__snapshots)
        self.omega_max = max(self.__snapshots)
        
        # compute reduced basis
        
        if self.logging: print("compute Reduced Basis")

        # extend basis if it already exists
        try: 
            existing_basis_len = len(self.V)
            if self.logging: print("extend reduced basis")
        except:
            existing_basis_len = 0

        with TaskManager():

            for n in range(0+existing_basis_len, len(self.__snapshots)):
                omega = self.__snapshots[n]
                
                # compute FEM solution for parameter omega
                self.omega.Set(omega)
                self.a.Assemble()

                self.ainv.Update()
                self.gfu.vec.data = self.ainv * self.f.vec
                
                try:
                    self.V.AppendOrthogonalize(self.gfu.vec)
                except:
                    self.V = MultiVector(self.gfu.vec, 1)
                    self.V[0] = self.gfu.vec
                    
            # compute matrices in reduced space
            mv = MultiVector(self.gfu.vec, len(self.__snapshots))
            
            self.not_zero = []
            for key in ['k', 'r', 'm']:
                try:
                    mv[0:len(self.__snapshots)] = eval('self.{}.mat * self.V'.format(key))
                    exec('self.{}_red = InnerProduct(self.V, mv)'.format(key))
                    self.not_zero += [key]
                except:
                    exec('self.{}_red = Matrix(len(self.V), len(self.V), self.fes.is_complex)'.format(key))
                    exec('self.{}_red[:] = 0'.format(key))

            mv = MultiVector(self.f.vec, 1)
            mv[0] = self.f.vec
            self.f_red = InnerProduct(self.V, mv)
            
            
            if self.logging: print("finished computing Reduced Basis")
        
        
            dim = eval('(self.{}_red.h, self.{}_red.w)'.format(
                self.not_zero[0], self.not_zero[0]))
            bv_tmp = self.__bv_tmp # needed to be able to use exec(''' .. ''')
            tmp = MultiVector(self.__bv_tmp, dim[0])
            keys = []

            for i in range(len(self.not_zero)):
                # set multivectors
                exec(textwrap.dedent('''
                {}_zeta = MultiVector(bv_tmp, dim[0])
                tmp.data = self.{}.mat * self.V
                {}_zeta.data = self.proj * tmp
                '''.format(self.not_zero[i], self.not_zero[i], self.not_zero[i])))
                # set keys
                for k in range(i, len(self.not_zero)):
                    keys += [self.not_zero[i]+self.not_zero[k]]

            self.__res_mat = {} # available keys: 'kk','kr','km','rr','rm','mm'

            # calculate scalar products
            for key in keys:
                self.__res_mat[key] = eval(
                    "InnerProduct ({}_zeta, {}_zeta, conjugate=False)".format(list(key)[0], list(key)[1]))
                # calculate inner products with right hand side
                if list(key)[0] == list(key)[1]:
                    self.__res_mat['{}f'.format(list(key)[0])] = Vector(dim[0], 
                                                            self.fes.is_complex)
                    for j in range(dim[0]): 
                        self.__res_mat['{}f'.format(list(key)[0])][j] = eval(
                            "InnerProduct ({}_zeta[{}], self.f.vec.data, conjugate=False)".format(list(key)[0], j))

            # set other matrices to zero
            for key in ['kk', 'kr', 'km', 'rr', 'rm', 'mm']:
                if key not in keys:
                    self.__res_mat[key] = Matrix(dim[0], dim[1], 
                                                 self.fes.is_complex)
                    self.__res_mat[key][:] = 0

            for key in ['k', 'r', 'm']:
                if key not in self.not_zero:
                    self.__res_mat['{}f'.format(key)] = Vector(dim[0],
                                                 self.fes.is_complex)
                    self.__res_mat['{}f'.format(key)][:] = 0

    def getSnapshots(self):
        return self.__snapshots
    
    
    def draw(self, omega, redraw=False):
        
        # compute reduced solution

        ## TODO: updatable a_red.inv (Base Matrix instead of bla-Matrix?) and omega as parameter
        A = self.k_red+self.m_red*omega*omega+self.r_red*omega
        v = A.I * self.f_red
        
        self.drawu.vec.data = self.V * v[:,0]
        
        if self.logging: print("omega: {}, norm of solution: {}".format(omega, 
            np.real(Integrate ( Conj(self.drawu)*(self.drawu), self.fes.mesh))))
        
        # draw solution
        if not redraw:
            self.scene = Draw(self.drawu)
        else:
            self.scene.Redraw()
            

    def computeValues(self, param, residual=True, norm=True, cheap = True):
        
        if residual and norm: 
            if self.logging: print("compute residual and norm")
        elif residual: 
            if self.logging: print("compute residual")
        elif norm: 
            if self.logging: print("compute norm")
        else: return
        
        norm_ret = []
        residual_ret = []
        
        with TaskManager():

            for omega in param:

                # compute reduced solution
                A = self.k_red+self.m_red*omega*omega+self.r_red*omega
                v = A.I * self.f_red
                red_sol_vec = v[:,0]
            
                if norm:

                    self.__gf_tmp.vec.data = self.V * red_sol_vec
                    # imaginary part is not exactly 0 due to numerical errors
                    nof = np.real(Integrate(self.__gf_tmp*Conj(self.__gf_tmp), self.fes.mesh))
                    norm_ret += [nof]

                if residual:   
                    
                    if cheap:
                            
                        A = (self.__res_mat['kk']
                             + self.__res_mat['km']*2*omega**2
                             + self.__res_mat['mm']*omega**4 
                             + self.__res_mat['kr']*2*omega
                             + self.__res_mat['rm']*2*omega**3
                             + self.__res_mat['rr'] *omega**2)
                            
                        A_F = self.__res_mat['kf']+self.__res_mat['mf']*omega**2+self.__res_mat['rf'] *omega
      
                        res = (InnerProduct(red_sol_vec, A * red_sol_vec) 
                               - 2*np.real( InnerProduct(red_sol_vec, A_F)) 
                               + self.__normf)
                        
                        residual_ret += [abs(res)]

                    else:
                        
                        if not norm: self.__gf_tmp.vec.data = self.V * red_sol_vec
                        self.omega.Set(omega)
                        self.a.Assemble()
                        self.__bv_tmp.data = self.a.mat*self.__gf_tmp.vec - self.f.vec
                        self.__bv_tmp2.data = self.proj*self.__bv_tmp
                        res = Norm(self.__bv_tmp2)**2
                    
                        residual_ret += [res]
                    
                    
        if self.logging: print("finished computing values")
        if norm and (not residual): return norm_ret
        if residual and (not norm): return residual_ret
        if norm and residual: return norm_ret, residual_ret
    
