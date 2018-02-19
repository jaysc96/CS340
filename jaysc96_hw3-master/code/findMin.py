import numpy as np
from numpy.linalg import inv,norm

def findMin(funObj,w,maxEvals,*args):
    # Parameters of the Optimization
    optTol = 1e-2
    gamma = 1e-4

    # Evaluate the initial function value and gradient

    f,g = funObj(w,*args)
    funEvals = 1

    alpha = 1
    while True:
        # Line-search to find an acceptable value of alpha
        w_new = w - alpha*g
        [f_new,g_new] = funObj(w_new,*args)
        funEvals += 1

        gg = g.T.dot(g)

        while f_new > f - gamma * alpha*gg:
            print("Backtracking...")
            alpha = (alpha**2) * gg/(2*(f_new - f + alpha*gg))
            w_new = w - alpha * g
            (f_new, g_new) = funObj(w_new, *args)
            funEvals += 1


        # Update step-size for next iteration
        y = g_new - g
        alpha = -alpha*np.dot(y.T,g)/np.dot(y.T,y)

        # Update parameters/function/gradient
        w = w_new
        f = f_new
        g = g_new

        # Test termination conditions

        optCond = norm(g,float('inf'))
        print("{0} {1} {2} {3}".format(funEvals,alpha,f,optCond))

        if optCond < optTol:
            print("Problem solved up to optimality tolerance")
            break

        if funEvals >= maxEvals:
            print("At maximum number of function evaluations")
            break

    return w,f
