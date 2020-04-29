from ngsolve import *
from netgen.geom2d import unit_square
import numpy as np

# Hi 
#import matplotlib.pyplot as plt ## error if included

def NormOfSolutions(parameters):
    visual = False
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.2))

    fes = H1(mesh, order=3, dirichlet='top|bottom|left|right')
    u,v =fes.TnT()


    x_0 = 0.5
    y_0 = 0.5
    sigma_sqared = 0.05
    factor = 10

    f = LinearForm(fes)
    f += exp(-factor*((x-x_0)**2/(2*sigma_sqared) + (y-y_0)**2/(2*sigma_sqared)))* v * dx
    f.Assemble()

    norm_of_solutions = [] # sqared l2 norm actually
    V = [] # reduced basis
    mu = [] # corresponding values of parameters    

    gfu = GridFunction(fes)
    if visual: Draw(gfu)


    for w in parameters:
        ## compute FEM solution for parameter w
        a = BilinearForm(fes)
        a += (grad(u)*grad(v) - w*w *u*v) * dx
        a.Assemble()
        gfu.vec.data = a.mat.Inverse(fes.FreeDofs(), inverse="sparsecholesky") * f.vec
        nof = sqrt(Integrate(gfu*gfu, mesh))
        norm_of_solutions += [nof]
        
        if visual:
            Redraw()
            input("norm of solution {}: {}".format(w, nof))
    return norm_of_solutions







def Do(red_param, comp_param):
    visual = False
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.2))

    fes = H1(mesh, order=3, dirichlet='top|bottom|left|right')
    u,v =fes.TnT()

    k_blf = BilinearForm(fes)
    k_blf += grad(u)*grad(v)*dx
    k_blf.Assemble()
    K_orig = k_blf.mat

    m_blf = BilinearForm(fes)
    m_blf += u*v * dx
    m_blf.Assemble()
    M_orig = m_blf.mat

    x_0 = 0.5
    y_0 = 0.5
    sigma_sqared = 0.05
    factor = 10

    func = exp(-factor*((x-x_0)**2/(2*sigma_sqared) + (y-y_0)**2/(2*sigma_sqared)))

    f = LinearForm(fes)
    f += func * v * dx
    f.Assemble()

    omega = Parameter(0)
    a = BilinearForm(fes)
    a += (grad(u)*grad(v) - omega*omega *u*v) * dx
    a.Assemble()

    gfu = GridFunction(fes)
    if visual: Draw(gfu)

    dim_orig= len(gfu.vec)# dim of original space
    dim_red = len(red_param)  # dim of reduced space
    norm_of_solutions_orig = [] # l2 norm 
    V = np.zeros((dim_orig,dim_red)) # reduced basis

    basis = GridFunction(fes, multidim=dim_red)


    for n in range(0, dim_red):
        _omega = red_param[n]
        ## compute FEM solution for parameter _omega
        omega.Set(_omega)
        a.Assemble()
        gfu.vec.data = a.mat.Inverse(fes.FreeDofs(), inverse="sparsecholesky") * f.vec
        nof = sqrt(Integrate(gfu*gfu, mesh))
        norm_of_solutions_orig += [nof]
        z = np.array(gfu.vec.data)
        z = np.zeros(dim_orig)
        if n == 0:
            z += np.array(gfu.vec.data)
        else:
            z += np.matmul(V, np.matmul(np.transpose(V),
                        np.array(gfu.vec.data)))
            z = np.array(gfu.vec.data) - z
        
        V[:,n] = z/np.linalg.norm(z)
        # basis.vecs[n].FV().NumPy()[:] = z/np.linalg.norm(z)

        if visual:
            Redraw()
            input("norm of solution {}: {}".format(_omega, nof))
    
    ## test computation of basis
    visual = False
    # print(basis)
    # help(basis)
    if visual:

        for j in range(0, dim_red):
            no = sqrt(Integrate(basis.components[j]*basis.components[j], mesh))
            # Redraw()
            input("norm of solution {}: {}".format(red_param[j], no))
        

    np.savetxt('V.txt', V)


    K_red = np.zeros((dim_red, dim_red))
    M_red = np.zeros((dim_red, dim_red))
    P_red = np.zeros(dim_red)

    # TODO: make more efficient!!
    # for i in range(0, dim_red):
    #     tmp_p = 0
    #     for j in range(0, dim_red):
    #         tmp_k = 0
    #         tmp_m = 0
    #         for n in range(0, dim_orig):
    #             if j == 0:
    #                 tmp_p += f.vec.data[n]*V[n,i]
    #                 #print("V[{},{}]: {}".format(n,i, V[n,i]))
    #             for k in range(0, n):
    #                 tmp_k += V[k,j]*V[n-k,i]*K_orig[k,n-k]
    #                 tmp_m += V[k,j]*V[n-k,i]*M_orig[k,n-k]
    #         K_red[i,j] = tmp_k
    #         M_red[i,j] = tmp_m
    #     print('finished for i={}'.format(i))
    #     P_red[i] = tmp_p
    
    tmpi = GridFunction(fes)
    tmpj = GridFunction(fes)
    for i in range(0, dim_red):
        for j in range(0, dim_red):
            tmpi.vec.FV().NumPy()[:] = V[:,i]
            tmpj.vec.FV().NumPy()[:] = V[:,j]
            M_red[i,j] = Integrate(tmpi*tmpj, mesh)
            K_red[i,j] = Integrate(grad(tmpi)*grad(tmpj), mesh)
        P_red[i] = Integrate(func*tmpi, mesh)

    np.savetxt('K_red.txt', K_red)
    np.savetxt('M_red.txt', M_red)
    np.savetxt('P_red.txt', P_red)



    ## norms with reduced basis
    norm_of_solutions_red = [] # l2 norm

    ured = GridFunction(fes)
    visual = True

    Draw(ured)
    for _omega in comp_param:
        ## compute reduced solution for parameter _omega
        Ainv = np.linalg.inv(K_red-_omega*_omega*M_red)
        red_sol_vec = np.matmul(Ainv, P_red)
        ured.vec.FV().NumPy()[:] = np.matmul(V, red_sol_vec)[:]
        nof = sqrt(Integrate(ured*ured, mesh))
        norm_of_solutions_red += [nof]
        # norm_diff = Integrate(ured-) # Diff echte Loesg 

        if visual:
            Redraw()
            input("norm of solution {}: {}".format(_omega, nof))
    
    return  [norm_of_solutions_orig, norm_of_solutions_red]


if __name__ == "__main__":
    stepsize= 0.2
    reduced_stepsize = 0.01
    beginning = 3
    end = 20
    amount_of_pi = int(float(end-beginning)/np.pi)

    x_pi = [k*np.pi+beginning for k in range(amount_of_pi)]
    y_pi = [1]*amount_of_pi
    param = np.arange(beginning,end, stepsize)
    test_param = np.arange(beginning,end, reduced_stepsize)
    sol_red = test_param
    # sol_orig = NormOfSolutions(param)
    [sol_orig, sol_red] = Do(param, param)
    ## import here because ngsolve returns an error otherwise
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(param, sol_orig)
    ax.plot(x_pi, y_pi, 'ro')
    plt.yscale('log')
    ax.set_title("original space with stepsize {}".format(stepsize))
    fig, ax = plt.subplots()
    ax.plot(test_param, sol_red)
    ax.plot(x_pi, y_pi, 'ro')
    plt.yscale('log')
    ax.set_title("reduced space with stepsize {}".format(reduced_stepsize))
    plt.show()

