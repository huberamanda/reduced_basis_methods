from ngsolve import *
from netgen.geom2d import unit_square
import numpy as np
import scipy.sparse as sparse
import cProfile

class ReducedBasis:

    def __init__(self):

        self.setSpace(np.pi, 0.4, 0.6)    
        self.setInterval(0, 5)

    def reset(self):
        
        self.V = None
        self.K_red = None
        self.M_red = None
        self.P_red = None
        self.sol = []
        self.norm_of_solutions_red = [] 
        self.norm_of_solutions_orig = [] # l2 norm 
        self.residual = []
        self.dim_orig= len(self.gfu.vec)# dim of original space
        self.dim_red  = None
        self.__snapshots_updated = True
        self.__snapshots = []
        self.indices = []
        self.__first_draw = True
        self.__drawmode = 'default'
        
    def setDrawmode(self, mode):
        allowed_modes = ['webgui', 'default']
        
        if mode in allowed_modes:
            self.__drawmode = mode
        else:
            self.__drawmode = 'default'
        self.__first_draw = True
        

    def setSpace(self, a, x_0, y_0, rb='dirichlet'):
        ## generate mesh
        geo = netgen.geom2d.SplineGeometry()
        # p1 = geo.AppendPoint (0,0)
        # p2 = geo.AppendPoint (np.pi,0)
        # p3 = geo.AppendPoint (np.pi, np.pi)
        # p4 = geo.AppendPoint (0,np.pi)

        p1 = geo.AppendPoint (0,0)
        p2 = geo.AppendPoint (a,0)
        p3 = geo.AppendPoint (a,a)
        p4 = geo.AppendPoint (0,a)

        geo.Append (["line", p1, p2], bc = "bottom")
        geo.Append (["line", p2, p3], bc = "right")
        geo.Append (["line", p3, p4], bc = "top")
        geo.Append (["line", p4, p1], bc = "left")

        self.mesh = Mesh(geo.GenerateMesh(maxh=0.1))

        if rb == 'dirichlet':
            self.fes = H1(self.mesh, order=5, dirichlet='top|bottom|left|right')
        else:
            self.fes = H1(self.mesh, order=5)
        u,v =self.fes.TnT()

        factor = 25

        func = exp(-factor*((x-x_0)**2 + (y-y_0)**2))

        self.f = LinearForm(self.fes)
        self.f += func * v * dx
        self.f.Assemble()

        self.omega = Parameter(0)
        self.a = BilinearForm(self.fes)
        self.a += (grad(u)*grad(v) - self.omega*self.omega *u*v) * dx
        self.a.Assemble()

        self.gfu = GridFunction(self.fes)
        self.drawu = GridFunction(self.fes)
        

        k_blf = BilinearForm(self.fes)
        k_blf += grad(u)*grad(v)*dx
        k_blf.Assemble()
        rows,cols,vals = k_blf.mat.COO()
        self.K_orig = sparse.csr_matrix((vals,(rows,cols)))

        m_blf = BilinearForm(self.fes)
        m_blf += u*v * dx
        m_blf.Assemble()
        rows,cols,vals = m_blf.mat.COO()
        self.M_orig = sparse.csr_matrix((vals,(rows,cols)))

        self.reset()


    def setInterval(self, beginning, end):
        self.beginning = beginning
        self.end = end

        self.reset()
        
    def save(self, out_dir):
        
        print("save in folder {}", out_dir)
        
        changeable = [self.V, self.K_red, self.M_red, self.P_red, self.sol, self.norm_of_solutions_red,                               self.norm_of_solutions_orig, self.residual, self.beginning, self.end, self.dim_orig, self.dim_red, 
         self.__snapshots, self.__snapshots_updated, self.indices]

        for j in range(len(changeable)):
            np.save(out_dir+str(j), changeable[j], allow_pickle=True)
            
    def load(self, in_dir):
        
        print("load from folder {}", in_dir)

        ending = ".npy"
        self.V = np.load(in_dir+str(1)+ending, allow_pickle=True)
        self.K_red = np.load(in_dir+str(2)+ending, allow_pickle=True)
        self.M_red = np.load(in_dir+str(3)+ending, allow_pickle=True)
        self.P_red = np.load(in_dir+str(4)+ending, allow_pickle=True)
        self.sol = np.load(in_dir+str(5)+ending).tolist()
        self.norm_of_solutions_red = np.load(in_dir+str(6)+ending).tolist()
        self.norm_of_solutions_orig = np.load(in_dir+str(7)+ending).tolist()
        self.residual = np.load(in_dir+str(8)+ending).tolist()
        self.beginning = np.load(in_dir+str(9)+ending)
        self.end = np.load(in_dir+str(10)+ending)
        self.dim_orig= np.load(in_dir+str(11)+ending)
        self.dim_red  = np.load(in_dir+str(12)+ending)
        self.__snapshots = np.load(in_dir+str(13)+ending)
        self.__snapshots_updated = np.load(in_dir+str(14)+ending)
        self.indices = np.load(in_dir+str(15)+ending).tolist()
        
        
    def setSnapshots(self, new_snapshots, reset = False):
        ## TODO: check format of snapshots
        if len(self.__snapshots) > 0 and not reset:
            print("append snapshots with {}".format(new_snapshots))
            self.__snapshots = np.append(self.__snapshots, np.array(new_snapshots))
        else:
            print("set snapshots and reset basis")
            self.__snapshots = np.array(new_snapshots)
            self.V = None
        self.__snapshots_updated = True
        tmp = self.__snapshots
        zip_to_sort = list(zip(tmp, range(len(tmp))))
        sorted_zip = sorted(zip_to_sort, key=lambda x: x[0], reverse=False)
        self.indices = [tup[1] for tup in sorted_zip]
    
    def getSnapshots(self, sorted = True):
        return self.__snapshots[self.indices]
          
                
    def __computeRB(self):

        print("compute Reduced Basis")


        if len(self.__snapshots) == 0:
            print(""" no snapshots given, please call 'instance.setSnapshots' first""")
            return
        
        _visual = False
        if _visual:
            Draw(self.gfu)

        self.dim_red = len(self.__snapshots)
        V_tmp = np.zeros((self.dim_orig, self.dim_red))

        try: 
            existing_basis_len = self.V.shape[1]
            V_tmp[:,0:existing_basis_len] = self.V
            print("extending basis")
        except:
            existing_basis_len = 0


        for n in range(0+existing_basis_len, self.dim_red):
            _omega = self.__snapshots[n]
            ## compute FEM solution for parameter _omega
            self.omega.Set(_omega)
            self.a.Assemble()
            self.gfu.vec.data = self.a.mat.Inverse(self.fes.FreeDofs(), inverse="sparsecholesky") * self.f.vec
            nof = Integrate(self.gfu*self.gfu, self.mesh)
            self.norm_of_solutions_orig += [nof]
            V_tmp[:,n] = np.array(self.gfu.vec.data) 
            self.sol += [self.gfu.vec]

            if _visual:
                Redraw()
                input("norm of solution {}: {}".format(_omega, nof))
                
        try:
            q, r = np.linalg.qr(V_tmp)
            self.V = V_tmp.dot(np.linalg.inv(r))
        except:
            print("Matrix is singular")

        self.K_red = np.transpose(self.V).dot(self.K_orig.dot(self.V))
        self.M_red = np.transpose(self.V).dot(self.M_orig.dot(self.V))
        self.P_red = np.transpose(self.V).dot(self.f.vec.data)
        
        self.__snapshots_updated = False
        print("finished computing Reduced Basis")
        
        
    def computeValues(self, param, residual=True, norm=True):

        if self.__snapshots_updated:
            self.__computeRB()

                    
        self.norm_of_solutions_red = [] 
        self.residual = []

        ured = GridFunction(self.fes)
        tmp = GridFunction(self.fes)

        j = 0
        norm_diff = []
        for _omega in param:

            if norm:
                ## TODO: solve lgs
                # with TaskManager(pajetrace = 100*1000*10000):
                Ainv = np.linalg.inv(self.K_red-_omega*_omega*self.M_red)
                red_sol_vec = np.matmul(Ainv, self.P_red)

                ured.vec.FV().NumPy()[:] = self.V.dot(red_sol_vec)[:]
                nof = Integrate(ured*ured, self.mesh)
                self.norm_of_solutions_red += [nof]
            
            if residual:
                ## TODO: efficient version 
                # print("compute Residual")
                A = self.K_orig-_omega*_omega*self.M_orig
                res = np.linalg.norm((A.dot(np.array(ured.vec.data))-self.f.vec.data)[self.fes.FreeDofs()])

                self.residual += [res]
            #print("norm of solution {}: {}".format(_omega, nof))
            j += 1

                
    def draw(self, omega):
        
        
        if self.__snapshots_updated:
            self.__computeRB()
        
        if self.__drawmode == 'webgui':
            from ngsolve.webgui import Draw
            
            if self.__first_draw:
                Draw(self.drawu)
        elif self.__drawmode == 'default':
            import netgen.gui
            from ngsolve import Draw
            if self.__first_draw:
                Draw(self.drawu)
        else: 
            print("unknown drawmode use instance.setDrawmode")
            return
            self.__first_draw = False
        

        ## TODO: solve lgs
        Ainv = np.linalg.inv(self.K_red-omega*omega*self.M_red)
        red_sol_vec = np.matmul(Ainv, self.P_red)
        
        self.drawu.vec.FV().NumPy()[:] = self.V.dot(red_sol_vec)[:]
        
        Redraw()


        
def func():
    RBinst = ReducedBasis()
    RBinst.setInterval(0,5)
    params = np.arange(5)
    RBinst.setSnapshots(params)
    RBinst.computeValues(params)



