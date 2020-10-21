

##LBFGS with obj and grad_j provided by autograd
from collections import deque

def lbfgs(f, gradf, n, ITERS=20, x0 = None):
    '''
        f: function for evaluating objective
        gradf: function which outputs gradient of f
        n: number of degrees of freedom
    '''
    #x0 = np.zeros(n)
    
    x0 = np.random.rand(n)
    
    
    g0 = np.zeros(n)
    H0 = np.identity(n); #first approximation of the hessian
    history = deque(maxlen = m);


    gradient_history = deque(maxlen = m)
    sk_history = deque(maxlen = m)
    yk_history = deque(maxlen = m)

    Sm = np.zeros((n,m))
    Ym = np.zeros((n,m))
    ## outer index i

    gradToler = 1e-10; #% tolerance for the norm of the slope
    XToler = 1e-10;    #% tolerance for the variables' refinement

    ## RUN 1 ITERATION PRIOR to entering the loop
    # assumes function is differentiable
    [f0,g0]=f(x0), gradf(x0); # f0 is evaluation, g0 is the gradient
                            # we want to move the gradient eval as a seperate function
    #print(f0, g0)

    # line search
    # usually line search method only return step size alpha
    # we return 3 variables to save caculation time.
    [alpha,f1,g1, eval_count] = strongwolfe(f, gradf,-g0,x0,f0,g0);
    print('initial ls count: ', eval_count)
    #print(alpha, f1, g1)
    x1 = x0 - alpha*g0;
    k =0;

    for _ in range(ITERS):

        ## use norm tolerance...this prevents the algorithm from having divide by zero errors
        fnorm = np.linalg.norm(g0);
        if(fnorm < gradToler):
            break;

        s0 = x1-x0; #nx1
        y0 = g1-g0; #nxn

        ## should this be a vector or not a vector?
        gamma_k = np.dot(s0.T,y0)/np.dot(y0,y0)
        hdiag = gamma_k*np.identity(n); #diagonal of the hessian, dneominator is scalar, numerator is a vector
        p = np.zeros((len(g0),1));
        #print(hdiag)
        ## UPDATE SEARCH DIRECTION
        if(k<m):  ##Cache isn't full
            # update S,Y
            Sm[:,k] = s0;
            Ym[:,k] = y0;
            # never forget the minus sign
            ## this doesn't seem to recompute any new g0 
            p = -getHg_lbfgs(g1,Sm[:,:k],Ym[:,:k],hdiag); 

        elif(k>=m): # the cache is full, reupdate the cache
            Sm[:,:m-1]=Sm[:,1:m];
            Ym[:,:m-1]=Ym[:,1:m];
            Sm[:,m-1] = s0;
            Ym[:,m-1] = y0;    
            # never forget the minus sign
            p = -getHg_lbfgs(g1,Sm,Ym,hdiag);

        ## LINE SEARCH:
        [alpha ,fs,gs, eval_count]= strongwolfe(f, gradf,p,x1,f1,g1);
        print('evals for line search: ', eval_count)
        x0 = x1; ## update x0 with x1 the delta
        g0 = g1;

        ## make a step
        x1 = x1 + alpha*p;
        f1 = fs;
        g1 = gs;
        # save caculation
        # [f1,g1]=feval(myFx,x1);
        k = k + 1;
        
def strongwolfe(f,gradf, d, x0, fx0, gx0):
    '''
        f: function 
        gradf: gradient of function (autograd)
        d: search direction
        x0: initial start
        fx0: initial function value at x0
        gx0: initial gradient at x0; should be a vector
        
        OUTPUTS:
        alphas: step size
        fs:     the function value at x0+alphas*d
        gs:     the gradient value at x0+alphas*d  
    '''
    ## only up to 3 iterations allowed?
    
    maxIter = 3; # how many iterations to run 
    alpham = 20;
    alphap = 0;
    c1 = 1e-4;
    c2 = 0.9;
    
    alphax = 1;
    gx0 = np.dot(gx0.T,d); ## what's this 
    fxp = fx0;
    gxp = gx0;
    i=1;
    eval_count = 0;
    while(True):
        xx = x0 +alphax*d; #proposed new val;
        ## evaluate function and gradient
        fxx, gxx = f(xx), gradf(xx);
        eval_count+=1;
        fs = fxx;
        gs = gxx;
        gxx = np.dot(gxx.T,d); ## project on the direction of search?
        if(fxx > fx0 + c1*alphax*gx0 or ((i>1) and (fxx >= fxp))): #evaluate wolfe condition;
            [alphas,fs,gs, subc] = Zoom(f,gradf,x0,d,alphap,alphax,fx0,gx0);
            eval_count+=subc
            break;
        if(np.abs(gxx) <= -c2*gx0): #not sure what this condition is
            alphas = alphax;
            break;
        if(gxx >= 0):
            [alphas,fs,gs, subc] = Zoom(f,gradf,x0,d,alphax,alphap,fx0,gx0);
            eval_count+=subc

            break;
        ## all breaking conditions were not satisfied
        alphap = alphax;
        fxp = fxx;
        gxp = gxx;
        if(i > maxIter):
            alphas = alphax;
            break
        ## r = rand(1);%randomly choose alphax from interval (alphap,alpham)
        r = 0.8;
        alphax = alphax + (alpham-alphax)*r;
        i+=1;
    
    return alphas, fs, gs, eval_count
            
def Zoom(f, gradf, x0,d,alphal,alphah,fx0,gx0):
    '''
        myFx:
        x0:
        d:
        alphal:
        alphah:
        fx0:
        gx0:
        
        returns
            alphas
            fs
            gs 
            eval_counts
        
    '''
    c1 = 1e-4;
    c2 = 0.9;
    i =0;
    maxIter = 3
    while(True):
        ## bisection
        alphax = 0.5*(alphal+alphah);
        alphas = alphax;
        xx = x0 + alphax*d;
        ## this is highly inefficient if we need to evaluate grad
        [fxx,gxx] = f(xx), gradf(xx);
        fs = fxx;
        gs = gxx;
        gxx = np.dot(gxx.T,d);
        xl = x0 + alphal*d;
        
        fxl = f(xl); #no gradient required
        
        if((fxx > fx0 + c1*alphax*gx0) or (fxx >= fxl)):
            alphah = alphax;
        else:
            if(abs(gxx) <= -c2*gx0):
                alphas = alphax;
                break;
            if(gxx*(alphah-alphal) >= 0):
                alphah = alphal;
            alphal = alphax;
       
        i = i+1;
        if(i > maxIter):
            alphas = alphax;
        
    return alphas,fs,gs, i
    
    
def weak_wolfe(f, d, x0, fx0):
    '''
        backtracking line search
        does not evaluate gradient
    '''
    alphaMax     = 1; #this is the maximum step length
    alpha        = alphaMax;
    fac          = 1/2; #< 1 reduction factor of alpha
    c_1          = 1e-1;

    while(f(x+alpha*dir) > fx0 + c_1*alpha*(dir'*dx):

        alpha = fac*alpha;

        if alpha < 10*eps:
            error('Error in Line search - alpha close to working precision');
      
    return alpha;
    
        
        
        

print(f1, f(x1))