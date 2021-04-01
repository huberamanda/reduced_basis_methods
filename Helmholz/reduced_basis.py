from ngsolve import *
from netgen.geom2d import unit_square


import numpy as np
import scipy
import math
import time

try:
    from ngsolve.webgui import Draw
except:
    import netgen.gui

np.random.seed(42)

class ReducedBasis:
    
    def __init__(self, fes, blf, rhs, snap = None):

        self.logging = True    
        
        self.fes = fes
        
        # set (bi-)linear forms and matrices
        self.omega = Parameter(0)

        self.bfs = {} # dictionary for bilinear forms
        self.a = BilinearForm(self.fes)
        for j in range(3):
            if blf[j]:
                self.a += sum([self.omega**j*igl.coef * igl.symbol for igl in blf[j]])
                self.bfs[j] = BilinearForm(blf[j]).Assemble()
        self.a.Assemble()
        
        self.f = LinearForm(self.fes).Add(rhs).Assemble()
        
        # set to zero at dirichlet boundaries
        self.proj = Projector(self.fes.FreeDofs(), True)
        self.proj.Project(self.f.vec)
        
        # store ainv for better performance
        self.ainv = self.a.mat.Inverse(self.fes.FreeDofs(), inverse="sparsecholesky")

        # initialize grid functions
        self.gfu = GridFunction(self.fes)
        self.drawu = GridFunction(self.fes)

        # temporary ngsolve vectors
        self.__bv_tmp = self.bfs[[*self.bfs][0]].mat.CreateColVector()
        self.__gf_tmp = GridFunction(self.fes)
        
        # store snapshots and reduced basis space
        self.__snapshots = []
        self.V = None

        if snap is not None:
            self.addSnapshots(snap)
                    

    def addSnapshots(self, new_snapshots):
        
        if self.logging: print("compute Reduced Basis for snapshots ", new_snapshots)
    
        with TaskManager():
            
            for omega in new_snapshots:
                if omega not in self.__snapshots:
                    
                    self.__snapshots.append(omega)

                    # compute FEM solution for parameter omega
                    self.omega.Set(omega)
                    self.a.Assemble()

                    self.ainv.Update()
                    self.gfu.vec.data = self.ainv * self.f.vec

                    if self.V is None:
                        self.V = MultiVector(self.gfu.vec, 1)
                        self.V[0] = self.gfu.vec
                    else:
                        self.V.AppendOrthogonalize(self.gfu.vec)
                        
            # compute matrices in reduced space
            mv = MultiVector(self.gfu.vec, len(self.__snapshots))
            self.red = {} # dictionary of matrices in reduced space
            for key in self.bfs.keys():
                mv[0:len(self.__snapshots)] = self.bfs[key].mat * self.V
                self.red[key] = InnerProduct(self.V, mv, conjugate=False)

            # right hand side in reduced space
            mv = MultiVector(self.f.vec, 1)
            mv[0] = self.f.vec
            self.f_red = InnerProduct(self.V, mv, conjugate=False)
            
            
            if self.logging: print("finished computing Reduced Basis")
                
        # store smallest and biggest snapshot parameter
        self.omega_min = min(self.__snapshots)
        self.omega_max = max(self.__snapshots)
        
        self.__computeResMat()


    def getSnapshots(self):
        return self.__snapshots
    
    def draw(self, omega,minval = -0.0001,maxval = 0.0001, autoscale = True,redraw=False):
        
        # compute reduced solution
        A = Matrix(len(self.__snapshots), len(self.__snapshots), self.fes.is_complex)
        A[:] = 0
        for key in self.red.keys():
            A += self.red[key]*omega**key
        
        v = A.I * self.f_red
        
        self.drawu.vec.data = self.V * v[:,0]
        
        if self.logging: print("omega: {}, norm of solution: {}".format(omega, 
            np.real(Integrate ( Conj(self.drawu)*(self.drawu), self.fes.mesh))))
        
        # draw solution
        if not redraw:
            self.scene = Draw(self.drawu,min = minval, max = maxval,autoscale = autoscale)
        else:
            self.scene.Redraw()

            
    def __computeResMat(self):

        self.__update_res_mat = False
        dim = (self.red[[*self.red][0]].h, self.red[[*self.red][0]].w)
        tmp = MultiVector(self.__bv_tmp, dim[0])
        
        names = ['k', 'r', 'm']
        keys = []
        zeta = {}
        with TaskManager():
            for i in self.bfs.keys():
                # set multivectors
                zeta[names[i]] = self.proj.Project((self.bfs[i].mat*self.V).Evaluate())
                # set keys
                for k in range(i, 3):
                    if k in self.bfs.keys(): keys += [names[i]+names[k]]
            
            self.__res_mat = {} # available keys: 'kk', 'kr', 'km', 'rr', 'rm', 'mm'

            # calculate scalar products
            for key in keys:
                self.__res_mat[key] = InnerProduct (zeta[key[0]], zeta[key[1]]).T
                # calculate inner products with right hand side
                if key[0] == key[1]:
                    self.__res_mat['{}f'.format(key[0])] = Vector(dim[0], self.fes.is_complex)
                    for j in range(dim[0]): 
                        self.__res_mat['{}f'.format(list(key)[0])][j] = InnerProduct (zeta[key[0]][j], self.f.vec.data)

            # set other matrices to zero
            for key in ['kk', 'kr', 'km', 'rr', 'rm', 'mm']:
                if key not in keys:
                    self.__res_mat[key] = Matrix(dim[0], dim[1], self.fes.is_complex)
                    self.__res_mat[key][:] = 0

            for i in range(3):
                if i not in self.bfs.keys():
                    self.__res_mat['{}f'.format(names[i])] = Vector(dim[0], self.fes.is_complex)
                    self.__res_mat['{}f'.format(names[i])][:] = 0


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
        
        # needed only in this jupyter notebook
        if self.__update_res_mat:
            self.__computeResMat()
        self.__update_res_mat = False

        
        with TaskManager():

            for omega in param:
                
                # compute reduced solution
                A = Matrix(len(self.__snapshots), len(self.__snapshots), self.fes.is_complex)
                A[:] = 0
                for key in self.red.keys():
                    A += self.red[key]*omega**key
                v = A.I * self.f_red
                red_sol_vec = v[:,0]
            
                if norm:

                    self.__gf_tmp.vec.data = self.V * red_sol_vec
                    # imaginary part is not exactly 0 due to numerical errors
                    nof = np.real(Integrate(self.__gf_tmp*Conj(self.__gf_tmp), self.fes.mesh))
                    norm_ret += [nof]

                if residual:   
                    
                    if cheap:
                        
                        if self.__update_res_mat: self.__computeResMat()
                            
                        A_F = self.__res_mat['kf']+self.__res_mat['mf']*omega**2+self.__res_mat['rf'] *omega
                        
                        # compute cheap residual for complex spaces 
                        if self.fes.is_complex:
                            A = (self.__res_mat['kk']
                                 + (self.__res_mat['kr']+self.__res_mat['kr'].H)*omega 
                                 + (self.__res_mat['km']+self.__res_mat['km'].H+self.__res_mat['rr'])*omega**2
                                 + (self.__res_mat['rm']+self.__res_mat['rm'].H)*omega**3
                                 + self.__res_mat['mm']*omega**4) 
                        else: # compute cheap residual for real spaces
                            A = (self.__res_mat['kk']
                                 + self.__res_mat['km']*2*omega**2
                                 + self.__res_mat['mm']*omega**4 
                                 + self.__res_mat['kr']*2*omega
                                 + self.__res_mat['rm']*2*omega**3)
                            
                        res = (InnerProduct(red_sol_vec, A * red_sol_vec) 
                               - 2*np.real( InnerProduct(red_sol_vec, A_F, conjugate = False)) 
                               + InnerProduct(self.f.vec, self.f.vec))
                        
                        residual_ret += [abs(res)]

                    else:
                        
                        if not norm: self.__gf_tmp.vec.data = self.V * red_sol_vec
                        self.omega.Set(omega)
                        self.a.Assemble()
                        res = Norm(self.proj.Project((self.a.mat*self.__gf_tmp.vec - self.f.vec).Evaluate()))
                        
                        residual_ret += [res]
                    
                    
        if self.logging: print("finished computing values")
        if norm and (not residual): return norm_ret
        if residual and (not norm): return residual_ret
        if norm and residual: return norm_ret, residual_ret
    
    