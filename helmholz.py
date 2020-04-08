


from ngsolve import *
from netgen.geom2d import unit_square
import numpy as np
ngsglobals.msg_level = 0

#import matplotlib.pyplot as plt ## error if included

def compute_solution(parameters):
    TOL = 1e-10

    mesh = Mesh(unit_square.GenerateMesh(maxh=0.2))

    fes = H1(mesh, order=3, dirichlet='t|b|l|r')

    u,v =fes.TnT()

    x_0 = 0.5
    y_0 = 0.5
    sigma_sqared = 0.01
    factor = 10

    f = LinearForm(fes)
    f += exp(-factor*((x-x_0)**2/(2*sigma_sqared) + (y-y_0)**2/(2*sigma_sqared)))* v * dx
    f.Assemble()

    norm_of_solutions = [] # sqared l2 norm actually
    V = []
    z = GridFunction(fes)
    tmp = GridFunction(fes)
    for w in parameters:
        a = BilinearForm(fes)
        a += (grad(u)*grad(v) - w*w *u*v) * dx
        a.Assemble()
        gfu = GridFunction(fes)
        gfu.vec.data = a.mat.Inverse(fes.FreeDofs(), inverse="sparsecholesky") * f.vec
        nof = Integrate(gfu*gfu, mesh)
        norm_of_solutions += [nof]
        if nof  > TOL:
            ## inline implementation of Gram-Schmidt 
            ## TODO: make more efficient
            if V==[]:
                z.vec.data = gfu.vec.data
            else:
                tmp.Set(0)
                for zeta in V:   
                    a = Integrate(zeta*zeta+grad(zeta)*grad(zeta), mesh)
                    b = Integrate(gfu*zeta + grad(gfu)*grad(zeta), mesh)
                    ## attention: multiplication with scalar only allowed from left side
                    tmp.vec.data += float(Integrate(gfu*zeta + grad(gfu)*grad(zeta), mesh)/
                            Integrate(zeta*zeta+grad(zeta)*grad(zeta), mesh))*zeta.vec.data 
                z.vec.data = gfu.vec.data-tmp.vec.data
            z.vec.data = float(1./sqrt(Integrate(z*z+grad(z)*grad(z), mesh)))*z.vec.data
            V += [z] 

        #Draw(gfu)
        #Redraw()
        # input("press key")

    ## Test if all vectors in V are normed
    for zeta in V:
        print(Integrate(zeta*zeta+grad(zeta)*grad(zeta), mesh))
    return norm_of_solutions



if __name__ == "__main__":
    sol = compute_solution(np.arange(20,50))
    # import here because ngsolve returns an error otherwise
    import matplotlib.pyplot as plt
    plt.plot(sol)
    plt.show()

