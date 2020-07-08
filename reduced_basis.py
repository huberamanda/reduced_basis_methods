from ngsolve import *
from netgen.geom2d import unit_square

import numpy as np
import scipy
import math

from ngsolve.webgui import Draw

class ReducedBasis:
    
    def __init__(self, fes, f, cfn = 1):
        self.setSpace(fes, f, cfn)
        self.logging = True
    
    def setSpace(self, fes, func, cfn = 1):
        

        self.fes = fes
        
        # assume robin bnd if fes is complex
        if fes.is_complex:
            self.robin = True
        else:
            self.robin = False
        

        # set (bi-)linear forms and matrices
        self.omega = Parameter(0)
                
        ##########################################
        ## TODO: implement the boundary thing in a better way
        u,v =self.fes.TnT()
        self.a = BilinearForm(self.fes)
        self.a += (grad(u)*grad(v) - self.omega*self.omega*cfn*cfn*u*v) * dx
        # what happens per default if "robin" doesn't exist as boundary?
        if self.robin:
            if 'robin' in self.fes.mesh.GetBoundaries():
                self.a += -1j*self.omega*u*v*ds (definedon=self.fes.mesh.Boundaries("robin"))
            else:
                self.a += -1j*self.omega*u*v*ds
        self.a.Assemble()
        self.a.Assemble()
        ##########################################
        
        # store ainv for better performance
        self.ainv = self.a.mat.Inverse(self.fes.FreeDofs(), inverse="sparsecholesky")

        self.f = LinearForm(self.fes)
        self.f += func * v * dx
        self.f.Assemble()
         

        self.k = BilinearForm(self.fes)
        self.k += grad(u)*grad(v)*dx
        self.k.Assemble()
        rows,cols,vals = self.k.mat.COO()
        self.K_orig = scipy.sparse.csr_matrix((vals,(rows,cols)))

        self.m = BilinearForm(self.fes)
        self.m += cfn*cfn*u*v * dx
        self.m.Assemble()
        rows,cols,vals = self.m.mat.COO()
        self.M_orig = scipy.sparse.csr_matrix((vals,(rows,cols)))

        if self.robin:
            self.r = BilinearForm(self.fes)
            if 'robin' in self.fes.mesh.GetBoundaries():
                self.r += u*v*ds (definedon=self.fes.mesh.Boundaries("robin"))
            else:
                self.r += u*v*ds
            self.r.Assemble()
            rows,cols,vals = self.r.mat.COO()
            self.R_orig = scipy.sparse.csr_matrix((vals,(rows,cols)))
    

        # initialize grid functions
        self.gfu = GridFunction(self.fes)
        self.drawu = GridFunction(self.fes)

        # temporary ngsolve vectors
        self.__bv_tmp = self.k.mat.CreateColVector()
        self.__bv_tmp2 = self.k.mat.CreateColVector()
        self.__gf_tmp = GridFunction(self.fes)
        
        
        self.__proj = Projector(self.fes.FreeDofs(), True)

        # compute norm of f
        self.__bv_tmp.data = self.__proj*self.f.vec
        self.f.vec.data = self.__bv_tmp.data
        self.normf = Norm(self.__bv_tmp)**2
        
        
        self.reset()

    
    def reset(self):
        # reset dynamically updated parameters
        self.K_red = None
        self.M_red = None
        self.F_red = None
        self.R_red = None
        self.__V = None  # snapshot solutions
        self.__snapshots_updated = True
        self.__snapshots = [] # snapshot parameters
        self.__indices = []


    def setSnapshots(self, new_snapshots, reset = True, compute_RB = True):
        
        # TODO: check that snapshots do not already exist
        if len(self.__snapshots) > 0 and not reset:
            new_snapshots = np.unique(new_snapshots)
            new_snapshots = new_snapshots[np.array([s not in self.__snapshots for s in new_snapshots])]
            if self.logging: print("append snapshots with {}".format(new_snapshots))
            self.__snapshots = np.append(self.__snapshots, new_snapshots)
        else:
            if self.logging: print("set snapshots and reset basis")
            self.__snapshots = np.array(new_snapshots)
            self.__V = None

        self.__snapshots_updated = True
        self.__update_res_mat = True

        # store smallest and biggest snapshot parameter
        self.omega_min = min(self.__snapshots)
        self.omega_max = max(self.__snapshots)

        # store indices of snapshots in ascending order
        tmp = self.__snapshots
        zip_to_sort = list(zip(tmp, range(len(tmp))))
        sorted_zip = sorted(zip_to_sort, key=lambda x: x[0], reverse=False)
        self.__indices = [tup[1] for tup in sorted_zip]

        if compute_RB:
            self.__computeRB()

    def getSnapshots(self):
        return self.__snapshots[self.__indices]
    
class ReducedBasis(ReducedBasis):
    
    def __computeRB(self):


        if self.logging: print("compute Reduced Basis")

        if len(self.__snapshots) == 0:
            if self.logging: print(""" no snapshots given, please call 'instance.setSnapshots' first""")
            return
        
        dim_orig = len(self.gfu.vec)
        dim_red = len(self.__snapshots)

        if self.robin:
            npdtype = "complex"
            
        else:
            npdtype = "float"
            
        V_tmp = np.zeros((dim_orig, dim_red), dtype=npdtype)


        # extend basis if it already exists
        try: 
            existing_basis_len = len(self.__V)
            ## TODO: implement Numpy interface instead of for loop
            for i in range(existing_basis_len):
                V_tmp[:,i] = self.__V[i].FV().NumPy()
#             V_tmp[:,0:existing_basis_len] = self.__V
            self.__V.Expand(dim_red-existing_basis_len)
            if self.logging: print("extend reduced basis")

        except:
            existing_basis_len = 0
            self.__V = MultiVector(self.__bv_tmp, dim_red)


        with TaskManager():

            for n in range(0+existing_basis_len, dim_red):
                _omega = self.__snapshots[n]
                
                # compute FEM solution for parameter _omega
                self.omega.Set(_omega)
                self.a.Assemble()

                self.ainv.Update()
                self.gfu.vec.data = self.ainv * self.f.vec

                V_tmp[:,n] = self.gfu.vec.FV().NumPy()        


            if self.logging: print("Calculate QR_Decomposition")
            dim = V_tmp.shape[1]
            tmp = np.zeros(V_tmp.shape[0], dtype=npdtype)
            tmp2 = np.zeros(V_tmp.shape[0], dtype=npdtype)
            r = np.zeros([V_tmp.shape[1],V_tmp.shape[1]], dtype=npdtype)
            for j in range(dim):
                r[j,j] = np.linalg.norm(V_tmp[:,j])
                tmp[:] = V_tmp[:,j]/r[j,j]
                for k in range(j+1,dim):
                    r[j,k] = np.vdot(tmp,V_tmp[:,k])
                    tmp2[:] = V_tmp[:,k]-r[j,k]*tmp
                    V_tmp[:,k] = tmp2/np.linalg.norm(tmp2)
            

            # rearange V and snapshots due to the order of the snapshots
            ## TODO: if Numpy interface working get rid of loop
            for i in range(dim_red):
                self.__V[i].FV().NumPy()[:] = V_tmp[:, self.__indices[i]]

            # set system in reduced basis space
            
             ## TODO: ngsolve instead of numpy
            V_tmp = V_tmp[:, self.__indices]
            self.K_red = np.transpose(V_tmp).dot(self.K_orig.dot(V_tmp))
            self.M_red = np.transpose(V_tmp).dot(self.M_orig.dot(V_tmp))
            self.F_red = np.transpose(V_tmp).dot(self.f.vec.data)

            if self.robin:
                self.R_red = np.transpose(V_tmp).dot(self.R_orig.dot(V_tmp))

            self.__snapshots_updated = False
            self.__snapshots = self.__snapshots[self.__indices]
            self.__indices = range(dim_red)
            
            if self.logging: print("finished computing Reduced Basis")
                
                    
    def draw(self, omega, redraw=False):
        
        if self.__snapshots_updated:
            self.__computeRB()
        
        # compute reduced basis
        
        ## TODO: can a updateable inverse used in that case?
        if self.robin:
            A = self.K_red-omega*omega*self.M_red-1j*omega*self.R_red
#             Ainv = np.linalg.inv(self.K_red-omega*omega*self.M_red-1j*omega*self.R_red)
        else:
            A = self.K_red-omega*omega*self.M_red
#             Ainv = np.linalg.inv(self.K_red-omega*omega*self.M_red)

#         red_sol_vec = Ainv.dot(self.F_red)
        red_sol_vec = np.linalg.solve(A, self.F_red)
        
        v = Vector(red_sol_vec.tolist()) 
        
        self.drawu.vec.data = self.__V * v
        if self.logging: print("omega: {}, norm of solution: {}".format(omega, Integrate ( Conj(self.drawu)*(self.drawu), self.fes.mesh)))
        # draw solution
        if not redraw:
            self.scene = Draw(self.drawu)
        else:
            self.scene.Redraw()
    
    def __computeResMat(self):
        
        self.__update_res_mat = False
        dim = self.K_red.shape
        

        k_zeta = MultiVector(self.__bv_tmp, dim[0])
        m_zeta = MultiVector(self.__bv_tmp, dim[0])
        tmp = MultiVector(self.__bv_tmp, dim[0])
        
        # enforce dirichlet boundaries
        tmp.data = self.k.mat * self.__V
        k_zeta.data = self.__proj * tmp
        
        tmp.data = self.m.mat * self.__V
        m_zeta.data = self.__proj * tmp
        
        
        keys = ['kk', 'mm', 'mk']
        
        
        if self.robin:
            r_zeta = MultiVector(self.__bv_tmp, dim[0])
            tmp.data = self.r.mat * self.__V
            r_zeta.data = self.__proj * tmp

            keys += ['rr', 'rm', 'rk']


        self.__res_mat = {} # kk, mm, km, rr, rm, rk
        
        for key in keys:
            self.__res_mat[key] = Matrix(dim[0], dim[1], self.robin)

            if list(key)[0] == list(key)[1]:
                self.__res_mat['{}f'.format(list(key)[0])] = Vector(dim[0], self.robin)
        
        
        # calculate scalar products
        for key in keys:
            self.__res_mat[key] = eval(
                "InnerProduct ({}_zeta, {}_zeta)".format(list(key)[0], list(key)[1]))
            # f
            if list(key)[0] == list(key)[1]:
                for j in range(dim[0]): 
                    self.__res_mat['{}f'.format(list(key)[0])][j] = eval(
                        "InnerProduct ({}_zeta[{}], self.f.vec.data)".format(list(key)[0], j))
    
    def computeValues(self, param, residual=True, norm=True, cheap = True):
        
        ret_val = []

        if self.__snapshots_updated:
            self.__computeRB()
        
        
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

            for _omega in param:
                
                ## TODO: can a updateable inverse used in that case?
                if self.robin:
                    A = self.K_red-_omega*_omega*self.M_red-1j*_omega*self.R_red
        #             Ainv = np.linalg.inv(self.K_red-omega*omega*self.M_red-1j*omega*self.R_red)
                else:
                    A = self.K_red-_omega*_omega*self.M_red
        #             Ainv = np.linalg.inv(self.K_red-omega*omega*self.M_red)

        #         red_sol_vec = Ainv.dot(self.F_red)
                red_sol_vec = Vector(np.linalg.solve(A, self.F_red).tolist())
            
                
                if norm:

                    self.__gf_tmp.vec.data = self.__V * red_sol_vec

                    # imaginary part is not exactly 0 due to numerical errors
                    nof = np.real(Integrate(self.__gf_tmp*Conj(self.__gf_tmp), self.fes.mesh))
                    norm_ret += [nof]

                if residual:   
                    
                    if cheap:
                        if self.__update_res_mat:
                            self.__computeResMat()
                        ## TODO: wrapper for ".C" for real matrices?
                        if self.robin:
                            A = (self.__res_mat['kk']-(self.__res_mat['mk']+
                                self.__res_mat['mk'].C)*_omega**2+ self.__res_mat['mm']*_omega**4)
                        else:
                            A = (self.__res_mat['kk']- self.__res_mat['mk']*_omega**2*2+self.__res_mat['mm']*_omega**4)
                            
                        A_F = self.__res_mat['kf']-self.__res_mat['mf']*_omega**2
        
                        if self.robin:
                            A += ((self.__res_mat['rk']+self.__res_mat['rk'].C)*-1j*_omega
                                 + (self.__res_mat['rm']+self.__res_mat['rm'].C) * 1j*_omega**3
                                 + self.__res_mat['rr'] * 1j*_omega**2)
                            A_F -= self.__res_mat['rf'] * 1j*_omega
                            
                        res = InnerProduct(red_sol_vec, A * red_sol_vec) - 2*np.real( InnerProduct(red_sol_vec, A_F)) + self.normf
                        residual_ret += [abs(res)]

                    else:
                        if not norm: self.__gf_tmp.vec.data = self.__V * red_sol_vec
                    
                        self.__bv_tmp.data = self.k.mat*self.__gf_tmp.vec - _omega*_omega*self.m.mat*self.__gf_tmp.vec - self.f.vec
                        
                        if self.robin:
                            self.__bv_tmp.data += -1j*_omega*self.r.mat*self.__gf_tmp.vec

                        self.__bv_tmp2.data = self.__proj*self.__bv_tmp
                        res = Norm(self.__bv_tmp2)**2
                    
                        residual_ret += [res]
                    
                    
        if self.logging: print("finished computing values")
        if norm and (not residual): return norm_ret
        if residual and (not norm): return residual_ret
        if norm and residual: return norm_ret, residual_ret
    