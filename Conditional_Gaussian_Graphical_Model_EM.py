##########################################################################################################################
# author: Thomas Lartigue <thomas.lartigue@polytechnique.edu>
#
#
# This is the companion code to the article:
#
#
# Mixture of Conditional Gaussian Graphical Models for unlabelled heterogeneous populations in the presence of co-factors. 
# Thomas Lartigue, Stanley Durrleman, Stéphanie Allassonnière.
# https://hal.inria.fr/hal-02874192/document
#
#
# If you use elements of this code in your work, please cite this article as reference.
##########################################################################################################################


############## functions used to define the GGL penalised EM for a Conditional GGM model (uses cofeatures) ###############


import numpy as np
import xarray as xr
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


##########################################################################################################################
#                         Conditional GGM with GGL penalty optimisation scheme (Proximal Gradient Descent)
#
#                                  with given class labels or class weights (supervised)
#
#                                               used for the M step of the EM 
##########################################################################################################################


# likelihood part of the supervised loss
def CGGM_neg_log_likelihood(Lambda, Sigma, Theta, Syy, Sxx, Sxy, n_xr):
    # determinant of the covariance
    # normalised by n_k/n, our renormalisation convention
    determinants= np.sum(-0.5*n_xr*np.log(np.linalg.det(Lambda)))/n_xr.sum()
    
    # dotproduct in the exponent
    # the Sxx... are already renormalised by 1/n
    dotproduct_yy = xr.dot(Lambda, Syy, dims=["component_y", "component_yT"])
    dotproduct_xy = 2*xr.dot(Theta, Sxy, dims=["component_y", "component_x"])
    dotproduct_xx = xr.dot(
        xr.dot(Theta, 
               xr.dot(Sigma, 
                      Theta, 
                      dims = ["component_y"]).rename(
                   {"component_yT":"component_y", 
                    "component_x":"component_xT"}), 
               dims = ["component_y"]), 
        Sxx, 
        dims = ["component_x", "component_xT"])


    dotproducts = 0.5*(dotproduct_yy+dotproduct_xx+dotproduct_xy).sum(dim=["label"])
    return float(determinants + dotproducts) # + 0.5*np.log(2*np.pi)*len(Lambda.component_y)

# penalty part of the supervised loss
def GGL_penalty(Lambda, Theta, l1, l2,  l1_Theta, l2_Theta):
    ## Lambda
    
    # make a copy and remove the diagonal in lambda that we won't penalise
    Lambda_prime = Lambda.copy()
    p = len(Lambda.component_y)
    for k in Lambda_prime.label:
        Lambda_prime.loc[dict(label=k)] = Lambda_prime.sel(label=k) -\
            np.eye(p)*np.diag(Lambda_prime.sel(label=k))
            
    pen_Lambda = np.float(l1*np.abs(Lambda_prime).sum() + 
                l2*((Lambda_prime**2).sum(dim = 'label')**0.5).sum())
    
    ## Theta
    pen_Theta = np.float(l1_Theta*np.abs(Theta).sum() + 
                l2_Theta*((Theta**2).sum(dim = 'label')**0.5).sum())
    
    return pen_Lambda + pen_Theta

# total supervised loss
def CGGM_full_loss(Lambda, Sigma, Theta, Syy, Sxx, Sxy, n_xr, l1, l2, l1_Theta, l2_Theta):
    # I divide by the "number of terms" in the sums (not actually the same for every sum) to get a clearer number
    return (CGGM_neg_log_likelihood(Lambda, Sigma, Theta, Syy, Sxx, Sxy, n_xr) + GGL_penalty(Lambda, Theta, l1, l2, l1_Theta, l2_Theta))/\
        (len(Lambda.label)*len(Lambda.component_y)**2)

def soft_thresholding(x, l):
    return np.sign(x)*np.maximum(np.abs(x)-l, 0)

# explicit, coefficient-wise solution of the approximate quadratic + penalised problem
def GGL_solution(G, l1, l2, alpha, is_Lambda=True):
    S = soft_thresholding(G, l1*alpha)
    # soft thresholding formula for thr off diagonal
    solution = S*np.maximum(1 - alpha*l2/((S**2).sum(dim=["label"]))**0.5, 0)
    # for the diagonal, same as G
    if is_Lambda:
        for k in solution.label.values:
            np.fill_diagonal(solution.loc[dict(label = k)].values, np.diag(G.sel(label=k)))
    return solution

# solve the proximal problem (quadratic approx + penalty)
def solve_proximal_problem(
    # current parameters
    Lambda_t, Theta_t, 
    # current gradients
    grad_Lambda, grad_Theta, 
    # initial step size
    alpha, 
    # penalty intensity
    l1, l2, l1_Theta, l2_Theta):
    
    # likelihood gradient step
    G_Lambda = Lambda_t - alpha*grad_Lambda
    G_Theta = Theta_t - alpha*grad_Theta
    # compute the GGL solutions with this step size
    Lambda_GGL = GGL_solution(G_Lambda, l1, l2, alpha, is_Lambda=True)
    Theta_GGL = GGL_solution(G_Theta, l1_Theta, l2_Theta, alpha, is_Lambda=False)
    return Lambda_GGL, Theta_GGL
    

# make sure that the solution of the proximal is SPD
def proximal_GD_step( # current parameters
    Lambda_t, Theta_t, 
    # current gradients
    grad_Lambda, grad_Theta, 
    # initial step size
    alpha, 
    # step size decreasing rate (<1)
    beta,
    # penalty intensity
    l1, l2, l1_Theta, l2_Theta):

    Lambda_GGL, Theta_GGL = solve_proximal_problem( 
        # current parameters
        Lambda_t, Theta_t, 
        # current gradients
        grad_Lambda, grad_Theta, 
        # initial step size
        alpha, 
        # penalty intensity
        l1, l2, l1_Theta, l2_Theta)

    # if the step leaves the matrix Lambda non positive, then make a smaller one
    while (np.linalg.eigvals(Lambda_GGL) <= 0).any():
        alpha = beta*alpha
        Lambda_GGL, Theta_GGL = solve_proximal_problem( 
            # current parameters
            Lambda_t, Theta_t, 
            # current gradients
            grad_Lambda, grad_Theta, 
            # initial step size
            alpha, 
            # penalty intensity
            l1, l2, l1_Theta, l2_Theta)

    # compute the covariance matrix once the solution is validated
    Sigma_GGL = xr.DataArray(
        data = np.linalg.inv(Lambda_GGL), 
        coords= Lambda_GGL.coords, 
        dims=Lambda_GGL.dims)
    return Lambda_GGL, Sigma_GGL, Theta_GGL, alpha

# choose proximal step size with a decreasing line search and an heuristic
def proximal_line_search( 
    # exhaustive statistics
    Syy, Sxx, Sxy, n_xr,
    # current parameters 
    Lambda_t, Sigma_t, Theta_t, 
    # current gradients
    grad_Lambda, grad_Theta, 
    # initial step size
    alpha,  
    # step size decreasing rate (<1) 
    beta,
    # penalty intensity
    l1, l2,  l1_Theta, l2_Theta,
    # for the line search criterion: exact order 2 term in the benchmark? Or approximation?
    exact_order_2=False, 
    # increase the order 2 (not exact anyways) term in the quadratic approx
    order_2_coeff = 1):

    ## 1st proximal solution

    # get SPD proximal solution
    Lambda_GGL, Sigma_GGL, Theta_GGL, alpha = proximal_GD_step( 
        # current parameters
        Lambda_t, Theta_t, 
        # current gradients
        grad_Lambda, grad_Theta, 
        # initial step size
        alpha, 
        # step size decreasing rate (<1)
        beta,
        # penalty intensity
        l1, l2,  l1_Theta, l2_Theta)

    # likelihood of said SPD proximal solution
    neg_log_likelihood = CGGM_neg_log_likelihood(Lambda_GGL, Sigma_GGL, Theta_GGL, 
                                                 Syy, Sxx, Sxy, n_xr)

    ## quadratic approximation benchmark of the solution

    # full loss "gradient"
    G_full_loss_Lambda = (Lambda_t - Lambda_GGL)/alpha
    G_full_loss_Theta = (Theta_t - Theta_GGL)/alpha

    # three terms of the quadratic approximation
    order_0 = CGGM_neg_log_likelihood(Lambda_t, Sigma_t, Theta_t, Syy, Sxx, Sxy, n_xr) 
    order_1 = - alpha*float(xr.dot(G_full_loss_Lambda,grad_Lambda)+
                            xr.dot(G_full_loss_Theta, grad_Theta))
    order_2 = 0.5*alpha*float((G_full_loss_Lambda**2).sum() + 
                              (G_full_loss_Theta**2).sum())
    if exact_order_2:
        order_2 = 0.5*alpha**2*exact_order_2_term(Sigma_t, Theta_t, Sxx, G_full_loss_Lambda, G_full_loss_Theta)
    else:    
        order_2 = 0.5*alpha*float((G_full_loss_Lambda**2).sum() + 
                                  (G_full_loss_Theta**2).sum())
    quadratic_approximation = order_0 + order_1 + order_2_coeff*order_2


    # if the negative log-likelihood of the proximal solution is not below the quadratic approximation 
    # of the likelihood evaluated at the solution, then start again with a smaller step size
    # 
    # why? This is the heuristic of the Proximal GD algorithm
    #counter=0
    while (neg_log_likelihood>quadratic_approximation):# and(alpha>1e-3):
        alpha = beta*alpha
        """        
        #counter+=1
        if alpha<1e-3:
            neg_log_likelihood_init = CGGM_neg_log_likelihood(
                Lambda_t, Sigma_t, Theta_t, 
                Syy, Sxx, Sxy, n_xr)
            if neg_log_likelihood<=neg_log_likelihood_init:
                print("cannot improve quadratic approx")
                return Lambda_GGL, Sigma_GGL, Theta_GGL, alpha
            else:
                print("cannot improve likelihood")
                return Lambda_t, Sigma_t, Theta_t, alpha
        """
        # get SPD proximal solution
        Lambda_GGL, Sigma_GGL, Theta_GGL, alpha = proximal_GD_step( 
            # current parameters
            Lambda_t, Theta_t, 
            # current gradients
            grad_Lambda, grad_Theta, 
            # initial step size
            alpha, 
            # step size decreasing rate (<1)
            beta,
            # penalty intensity
            l1, l2,  l1_Theta, l2_Theta)

        # likelihood of the solution
        neg_log_likelihood = CGGM_neg_log_likelihood(Lambda_GGL, Sigma_GGL, Theta_GGL, 
                                                     Syy, Sxx, Sxy, n_xr)

        # quadratic approximation benchmark of the solution

        #  "gradient" of the full loss
        G_full_loss_Lambda = (Lambda_t- Lambda_GGL)/alpha
        G_full_loss_Theta = (Theta_t-Theta_GGL)/alpha

        # three terms of the quadratic approximation
        # order_0 is fixed once and for, not dependent on the GD step
        #order_0 = CGGM_neg_log_likelihood(Lambda_t, Sigma_t, Theta_t, Syy, Sxx, Sxy, n) 
        order_1 = - alpha*float(xr.dot(G_full_loss_Lambda,grad_Lambda)+
                                xr.dot(G_full_loss_Theta, grad_Theta))
        if exact_order_2:
            order_2 = 0.5*alpha**2*exact_order_2_term(Sigma_t, Theta_t, Sxx, G_full_loss_Lambda, G_full_loss_Theta)
        else:    
            order_2 = 0.5*alpha*float((G_full_loss_Lambda**2).sum() + 
                                      (G_full_loss_Theta**2).sum())
        quadratic_approximation = order_0 + order_1 + order_2_coeff*order_2

    return Lambda_GGL, Sigma_GGL, Theta_GGL, alpha

# the exact value of the order 2 term in the quadratic approximation
def exact_order_2_term(Sigma_t, Theta_t, Sxx, G_full_loss_Lambda, G_full_loss_Theta):
    # intermediate terms
    B = xr.dot(Sxx, xr.dot(Theta_t, Sigma_t, dims=["component_y"]), 
               dims= ["component_x"]).rename({"component_yT":"component_y", 
                                              "component_xT":"component_x"})
    A = xr.dot(xr.dot(Theta_t, Sigma_t, dims=["component_y"]), B, 
               dims=["component_x"])
    
    # computer order 2 term with real hessian
    order_2_SigSig = 0
    order_2_TheThe = 0
    order_2_SigThe = 0
    for k in Sigma_t.label.values:
        order_2_SigSig += np.trace(xr.dot(
            xr.dot(
                xr.dot(Sigma_t, 
                       G_full_loss_Lambda.rename({"component_y":"component_yTT"}), 
                       dims= ["component_yT"]),
                (2*A+Sigma_t).rename({"component_yT":"component_yTT", 
                          "component_y":"component_yTTT"}),
                dims = ["component_yTT"]),
            G_full_loss_Lambda.rename({"component_y":"component_yTTT"}),
            dims = ["component_yTTT"]).sel(label=k))

        order_2_TheThe += 2*np.trace(xr.dot(
            xr.dot(
                xr.dot(Sigma_t, 
                       G_full_loss_Theta, 
                       dims= ["component_y"]),
                Sxx,
                dims = ["component_x"]).rename({"component_xT":"component_x"}), 
            G_full_loss_Theta,
            dims = ["component_x"]).sel(label=k))

        order_2_SigThe -= 4*np.trace(xr.dot(
            xr.dot(
                xr.dot(Sigma_t.rename({"component_y":"component_yTT"}), 
                       G_full_loss_Lambda.rename({"component_yT":"component_yTT"}), 
                       dims= ["component_yTT"]), 
                B, 
                dims=["component_y"]), 
            G_full_loss_Theta, 
            dims = ["component_x"]).sel(label=k))
    return order_2_SigSig +order_2_TheThe+order_2_SigThe

# solve a hierarchical CGGM with given labels (supervised problem) or class weights (M step)
def fit_supervised_CGGM(Lambda_t0, Sigma_t0, Theta_t0, 
                        Syy, Sxx, Sxy, n_xr,         
                        # penalty parameters
                        l1=10,
                        l2=10,
                         l1_Theta=0.1, l2_Theta=0.1,
                        # initial stepsize
                        alpha_init = 10,
                        # step size decreasing rate (<1) for line search (or when looking for SPD matrix)
                        beta = 0.66,
                        # step size decreasing rate (<1) along the algorithm (useful only if no line search)
                        rho = 1., #0.99
                        # stopping thresholds
                        loss_shift_threshold = 5*1e-4,
                        Lambda_shift_threshold = 5*1e-4,
                        Theta_shift_threshold = 5*1e-4,
                        # do the line search with the heuristic or just the step?
                        skip_line_search = False,
                        # order 2 term in the line search criterion exact or approximate?
                        exact_order_2=False,
                        order_2_coeff =2,
                        # prints or not
                        verbose= False):
    # loop
    loss_t0 = CGGM_full_loss(Lambda_t0, Sigma_t0, Theta_t0, Syy, Sxx, Sxy, n_xr, 
                             l1, l2,  l1_Theta, l2_Theta)
    if verbose:
        print(loss_t0)
        print()
    t=-1
    loss_shift=loss_shift_threshold
    Lambda_shift=Lambda_shift_threshold
    Theta_shift_threshold=Theta_shift_threshold
    while loss_shift >= loss_shift_threshold or Lambda_shift >= Lambda_shift_threshold or Theta_shift >= Theta_shift_threshold:
        t+=1
        # intermediate terms
        B = xr.dot(Sxx, xr.dot(Theta_t0, Sigma_t0, dims=["component_y"]), 
                   dims= ["component_x"]).rename({"component_yT":"component_y", 
                                                  "component_xT":"component_x"})
        A = xr.dot(xr.dot(Theta_t0, Sigma_t0, dims=["component_y"]), B, 
                   dims=["component_x"])

        # gradient of the likelihood
        grad_Lambda = 0.5*(Syy - n_xr*Sigma_t0/n_xr.sum()  - A)
        grad_Theta = 0.5*(2*Sxy + 2*B)

        # find the direction of one gradient step

        ## ""fixed"" step size (not actually fixed because we have to be SPD)

        # initial step size
        alpha=alpha_init*rho**t
        if skip_line_search:
            Lambda_t1, Sigma_t1, Theta_t1, alpha = proximal_GD_step( # current parameters
                Lambda_t0, Theta_t0, 
                # current gradients
                grad_Lambda, grad_Theta, 
                # initial step size
                alpha, 
                # step size decreasing rate (<1)
                beta,
                # penalty intensity
                l1, l2, l1_Theta, l2_Theta)

        else:
            #print("launch line search")
            Lambda_t1, Sigma_t1, Theta_t1, alpha = proximal_line_search( 
                # exhaustive statistics
                Syy, Sxx, Sxy, n_xr,
                # current parameters 
                Lambda_t0, Sigma_t0, Theta_t0, 
                # current gradients
                grad_Lambda, grad_Theta, 
                # initial step size
                alpha,  
                # step size decreasing rate (<1) 
                beta,
                # penalty intensity
                l1, l2, l1_Theta, l2_Theta,
                # exact order 2?
                exact_order_2, order_2_coeff)
            #print("line search done")

        loss_t1 = CGGM_full_loss(Lambda_t1, Sigma_t1, Theta_t1, 
                                 Syy, Sxx, Sxy, n_xr, l1, l2,  l1_Theta, l2_Theta)

        # loss shift after one step
        loss_shift = np.abs(loss_t1 - loss_t0)/np.abs(loss_t0)
        # Lambda shift after this one step
        Lambda_shift = np.float(np.sum(np.abs(Lambda_t1 - Lambda_t0))/np.sum(np.abs(Lambda_t0)))
        # Theta shift after this one step
        Theta_shift = np.float(np.sum(np.abs(Theta_t1 - Theta_t0))/np.sum(np.abs(Theta_t0)))
        if verbose:
            print(alpha)
            print(loss_t1)
            print(float(np.sum((Lambda_t1-Lambda_t0)**2)**0.5),
                  float(np.sum((Sigma_t1-Sigma_t0)**2)**0.5),
                  float(np.sum((Theta_t1-Theta_t0)**2)**0.5)
                  )
            print(loss_shift, Lambda_shift, Theta_shift)
            print()
        # DANGEROUS
        if loss_t1 > loss_t0:
            loss_shift=0
            Lambda_shift=0
            Theta_shift=0
        
        else:
            Lambda_t0, Sigma_t0, Theta_t0, loss_t0 = Lambda_t1, Sigma_t1, Theta_t1, loss_t1


    return Lambda_t0, Sigma_t0, Theta_t0



##########################################################################################################################
#                         Unsupervised (Mixture) likelihood and other unsupervised objective function
#
#                                               used for the E step of the EM 
##########################################################################################################################


## unsupervised (EM)

### model

# likelihood of each data point, under the parameters of each class 
# output : K x n
def by_label_log_likelihood(X, Y, Lambda, Sigma, Theta):
    # products of all the components together (no sum over the observations yet)
    # such that: Sxx = sum_{observations} xx / n_total
    xx = X * X.rename({"component_x":"component_xT"})
    yy = Y * Y.rename({"component_y":"component_yT"})
    xy = X * Y
    
    # determinant of the covariance
    determinants= xr.DataArray(
        data=-0.5*np.log(np.linalg.det(Lambda)), 
        coords=[Lambda.label.values],
        dims=["label"])
    
    # dotproduct in the exponent
    dotproduct_yy = xr.dot(Lambda, yy, dims=["component_y", "component_yT"])
    dotproduct_xy = 2*xr.dot(Theta, xy, dims=["component_y", "component_x"])
    dotproduct_xx = xr.dot(
        xr.dot(Theta, 
               xr.dot(Sigma, 
                      Theta, 
                      dims = ["component_y"]).rename(
                   {"component_yT":"component_y", 
                    "component_x":"component_xT"}), 
               dims = ["component_y"]), 
        xx, 
        dims = ["component_x", "component_xT"])


    dotproducts = 0.5*(dotproduct_yy+dotproduct_xx+dotproduct_xy)
    return -1*(determinants + dotproducts) # - 0.5*np.log(2*np.pi)*len(Lambda.component_y)

# Unsupervised, Mixture of CGGM negative log likelihood
def UCGGM_neg_log_likelihood(X, Y, pi, Lambda, Sigma, Theta):
    # likelihood by label for each observation
    # size log_likelihood: n x K
    log_likelihood = by_label_log_likelihood(X, Y, Lambda, Sigma, Theta) + np.log(pi)

    # direct way
    #return np.float(np.sum(np.log(np.exp(log_likelihood).sum(dim = "label"))))

    # dodge computation errors way
    # there should be no computation errors, the smaller values are put to 0,
    # that is all, there is no infinity here
    
    # we also divide this number by n_total to make it more readable 
    return -1.*np.float(np.sum(
        log_likelihood.max(dim= 'label') + 
        np.log(np.exp(log_likelihood - log_likelihood.max(dim= 'label')).sum(dim='label'))
    ))/len(X.observation_id)

def UCGGM_full_loss(X, Y, pi, Lambda, Sigma, Theta, l1, l2, l1_Theta, l2_Theta):
    # we also divide this number by K*p**2 to make it more readable 
    return (UCGGM_neg_log_likelihood(X, Y, pi, Lambda, Sigma, Theta) +GGL_penalty(Lambda, Theta, l1, l2, l1_Theta, l2_Theta))/\
        (len(Lambda.label)*len(Lambda.component_y)**2)

# weights p_i,k^t = p(y_i | x_i, theta_t) in the Q function (E step)
def label_weights_E_step(
    # observations
    Y, X, 
    # current parameter values
    pi_t, Lambda_t, Sigma_t, Theta_t):
    # return the p_i,k^t = p(y_i | x_i, theta_t)
    # size of the output: n x K
    K = len(pi_t.label)
    # first the gaussian likelihood for each observation for each class
    # size: n x K
    log_p = by_label_log_likelihood(X, Y, Lambda_t, Sigma_t, Theta_t)
    # to avoid log(0) and give a small chance to clusters that have vanished
    if (pi_t<=1e-8).any():
        pi_t = pi_t+1e-7
        pi_t = pi_t / pi_t.sum(dim="label")
    # we have: pi_t.sum(dim="label") = 1 + K*1e-7
    
    # Add the proba by class to get the numerator of the Bayes formula defining p_i,k^t
    log_p = log_p + np.log(pi_t)
    
    # we define this one to work with at least one 1, and 0 in case of pathologies
    log_p_tilde = log_p - log_p.max(dim="label")
    
    # because this one can produce nans:
    # np.exp(log_p)  / np.exp(log_p).sum(dim = "label")
    
    # finally, p_i,k^t :
    p_t = np.exp(log_p_tilde)  / np.exp(log_p_tilde).sum(dim = "label")
    
    if np.isnan(p_t.sum(dim="label", skipna=False).values).any():
        print("Nan at E step, resetting proba for those observations")
        print("{} Nans".format(np.sum(np.isnan(p_t.sum(dim="label", skipna=False).values))))
        # trying to preserve code integrity by resetting the proba for obs. where there
        # are NANs
        p_t.T[np.isnan(p_t.sum(dim="label", skipna=False))] = 1./K
    elif not (np.abs(p_t.sum(dim='label').values -1) <= 1e-10).all():
        print("Computational error w/ low probabilities at E step")        
    if  (p_t.sum(dim='observation_id').values == 0).any():
        print("Classes w/ 0 mass assigned to them at E step")
        
    return p_t

# function Q(theta|theta_t) no tempering
def Q(
    # observations
    Y, X, 
    # new parameter values
    pi, Lambda, Sigma, Theta, 
    # current parameter values
    pi_t, Lambda_t, Sigma_t, Theta_t):
    
    
    # label weights (E step)
    p_t = label_weights_E_step(
        # observations
        Y,X,
        # current parameter values
        pi_t, Lambda_t, Sigma_t, Theta_t)

    # Exhaustive statistics weighted by the E step
    # careful here: remember we divide by n_total (p_t.sum()) not by n_k (p_t.sum(dim=["observation_id"]))
    # this is just by convention, so that the penalty intensity does not have to scale with n, it's done everywhere

    Sxx_unsupervised = xr.dot(p_t*X, X.rename({"component_x":"component_xT"}), dims= "observation_id")/p_t.sum()
    Syy_unsupervised = xr.dot(p_t*Y, Y.rename({"component_y":"component_yT"}), dims= "observation_id")/p_t.sum()
    Sxy_unsupervised = xr.dot(p_t*X, Y, dims= "observation_id")/p_t.sum()
    
    
    # get the log likelihood by class w/ param theta
    # log P_theta(x_i, z_i = k) = log P_theta(x_i | z_i = k) + log(pi_k)
    log_likelihood = -1.*CGGM_neg_log_likelihood(Lambda, Sigma, Theta, Syy_unsupervised, Sxx_unsupervised, 
                                                 Sxy_unsupervised, n_xr=p_t.sum(dim=["observation_id"])) 
    # the result is divided by n_total (p_t.sum() )
    return np.float(log_likelihood + (np.log(pi)*p_t.sum(dim=["observation_id"])).sum()/p_t.sum() )


##########################################################################################################################
#                                       E step and M step together: the EM algorithm 
#
##########################################################################################################################


##### EM algorithm

# one EM step
def EM_step(Y, X, 
            # penalty intensities
            l1, l2, l1_Theta, l2_Theta, 
            # current parameters
            pi_t0, Lambda_t0, Sigma_t0, Theta_t0, 
            # stopping criteria of the proximal gradient descent
            loss_shift_threshold=1e-3, 
            Lambda_shift_threshold=1e-3, 
            Theta_shift_threshold=1e-3,
            # additional parameters
            verbose=False):

    # one EM step from _0 parameters

    # get dimensions of the problem
    #p = len(Lambda_t0.component_y)
    K = len(pi_t0)
    n = len(X.observation_id)
    # proba p_i,k^t
    p_t0 = label_weights_E_step(
        # observations
        Y,X,
        # current parameter values
        pi_t0, Lambda_t0, Sigma_t0, Theta_t0)

    # weight of each class at time t
    N_t0 = p_t0.sum(dim='observation_id')

    # optimise in (pi, Lambda and Theta ) jointly

    # pi is always free
    # update pi
    pi_t1 = N_t0/n
    # log penaly on pi to prevent it from being too small
    #coeff_pen_pi = n/1. # pi > 1/(K+1) 
    #pi_t1 = (N_t0 + coeff_pen_pi)/(n + coeff_pen_pi*len(pi_t1.label))
    #pi_t1 = pi_t1/pi_t1.sum(dim="label")
    
    # (supervised) CGGM optimisation
    # exhaustive statistics (division by "p_t0.sum()=n", not "p_t0.sum(dim="observation_id")" as is our convention)
    Sxx = xr.dot(p_t0*X, X.rename({"component_x":"component_xT"}), dims= "observation_id")/n
    Syy = xr.dot(p_t0*Y, Y.rename({"component_y":"component_yT"}), dims= "observation_id")/n
    Sxy = xr.dot(p_t0*X, Y, dims= "observation_id")/n
    if verbose:
        print(pi_t1.values)
    ### if not penalised: just take the unpenalised explicit formula
    if (l1 == 0) and (l2 ==0) and (l1_Theta==0) and (l2_Theta==0):
        # initialise
        l=1e-7
        q = len(Sxx.component_x)
        p = len(Sigma_t0.component_y)
        # regularisation term, not 0 only if needed
        regul = l*xr.DataArray(
            data=np.array([(np.linalg.matrix_rank(Sxx.sel(label=k))<q)*np.eye(q) for k in range(K) ]),
            coords=Sxx.coords, 
            dims=Sxx.dims)
        Sxx_inv = xr.DataArray(
            data=np.linalg.inv(Sxx + regul), 
            coords=Sxx.coords, 
            dims=Sxx.dims)
        # compute the matrix product
        Theta_Sigma = -1.*xr.dot(
            Sxx_inv, 
            Sxy,
            dims=["component_x"]).rename({"component_xT":"component_x", "component_y":"component_yT"})

        # need to multiply by n/p_t0.sum(dim='observation_id') here to compensate for our extra division in Syy and the rest
        Sigma_t1 = (Syy  + xr.dot(Sxy, Theta_Sigma, dims="component_x"))*n/p_t0.sum(dim='observation_id')
        # regularisation term, not 0 only if needed
        regul = l*xr.DataArray(
            data=np.array([(np.linalg.matrix_rank(Sigma_t1.sel(label=k))<p)*np.eye(p) for k in range(K) ]),
            coords=Sigma_t1.coords, 
            dims=Sigma_t1.dims)
        Lambda_t1 = xr.DataArray(
            data=np.linalg.inv(Sigma_t1  +  regul),
            coords=Sigma_t1.coords, 
            dims=Sigma_t1.dims)
        
        Theta_t1 = xr.dot(Theta_Sigma, Lambda_t1, dims="component_yT")
        
    ### if penalised: we need proximal GD
    else:
        # optimisation in Lambda and Theta (no pi)
        Lambda_t1, Sigma_t1, Theta_t1 = fit_supervised_CGGM(
            Lambda_t0, Sigma_t0, Theta_t0, 
            Syy, Sxx, Sxy, N_t0, l1, l2, l1_Theta, l2_Theta, 
            loss_shift_threshold=loss_shift_threshold, 
            Lambda_shift_threshold=Lambda_shift_threshold, 
            Theta_shift_threshold=Theta_shift_threshold,
            verbose=False)
        
    if verbose:
        print("Q before : {}".format((Q(Y, X,  
                                        # new parameter values
                                        pi_t0, Lambda_t0, Sigma_t0, Theta_t0, 
                                        # current parameter values
                                        pi_t0, Lambda_t0, Sigma_t0, Theta_t0)
                                      - GGL_penalty(Lambda_t0, Theta_t0, l1, l2, l1_Theta, l2_Theta))/
                                     (len(Lambda.label)*len(Lambda.component_y)**2)))
        print("Q after : {}".format((Q(Y, X, 
                                       # new parameter values
                                       pi_t1, Lambda_t1, Sigma_t1, Theta_t1,
                                       # current parameter values
                                       pi_t0, Lambda_t0, Sigma_t0, Theta_t0)    
                                     - GGL_penalty(Lambda_t1, Theta_t1, l1, l2, l1_Theta, l2_Theta))/
                                    (len(Lambda.label)*len(Lambda.component_y)**2)))
    return pi_t1, Lambda_t1, Sigma_t1, Theta_t1

def EM(Y, X, 
       # number of classes desired
       K, 
       # penalty on the precision matrix
       l1, l2, 
       # penalty on the transition matrix
       l1_Theta, l2_Theta,
       # initial values of the parameters
       pi_0, Lambda_0, Sigma_0, Theta_0, 
       # stopping criteria of the EM
       loss_shift_threshold_EM=1e-4, Lambda_shift_threshold_EM=1e-4, Theta_shift_threshold_EM=1e-4,
       # stopping criteria of the proximal gradient descent in the M step
       loss_shift_threshold_PGD=1e-4, Lambda_shift_threshold_PGD=1e-4, Theta_shift_threshold_PGD=1e-4,
       # do we want the prints?
       verbose = False, 
       frequence_print = 1, 
       # maximum number of steps
       max_steps=200):

    # initial values
    pi_t, Lambda_t, Sigma_t, Theta_t = pi_0.copy(), Lambda_0.copy(), Sigma_0.copy(), Theta_0.copy()
    
    # initialise number of steps
    step = 0
    print_counter = 0

    # make sure that we enter the loop
    loss_shift = loss_shift_threshold_EM+1
    Lambda_shift = Lambda_shift_threshold_EM+1
    Theta_shift = Theta_shift_threshold_EM+1
    
    while (loss_shift > loss_shift_threshold_EM or Lambda_shift > Lambda_shift_threshold_EM or Theta_shift > Theta_shift_threshold_EM) & (step<max_steps):
        # current loss
        loss = UCGGM_full_loss(X, Y, pi_t, Lambda_t, Sigma_t, Theta_t, l1, l2, l1_Theta, l2_Theta)
        # if wanted, display regularly the progress in log likelihood
        if verbose &(step*frequence_print >= print_counter):
            print(f"negative log likelihood at step {step} = {loss}")
            print_counter += 1
        
        # make an EM step
        pi_t, Lambda_t1, Sigma_t, Theta_t1 = EM_step(Y, X, l1, l2, l1_Theta, l2_Theta, 
                                                     # current parameters
                                                     pi_t, Lambda_t, Sigma_t, Theta_t, 
                                                     # stopping criteria of the proximal gradient descent
                                                     loss_shift_threshold=loss_shift_threshold_PGD, 
                                                     Lambda_shift_threshold=Lambda_shift_threshold_PGD, 
                                                     Theta_shift_threshold=Theta_shift_threshold_PGD)
        # new loss
        loss_prime = UCGGM_full_loss(X, Y, pi_t, Lambda_t1, Sigma_t, Theta_t1, l1, l2, l1_Theta, l2_Theta)
        
        # Theta shift after the step
        Theta_shift = np.float(np.sum(np.abs(Theta_t1 - Theta_t))/np.sum(np.abs(Theta_t)))
        # update Theta
        Theta_t = Theta_t1
        # Lambda shift after the step
        Lambda_shift = np.float(np.sum(np.abs(Lambda_t1 - Lambda_t))/np.sum(np.abs(Lambda_t)))
        # update Lambda
        Lambda_t = Lambda_t1
        # loss shift after the step
        loss_shift = np.abs(loss - loss_prime)/np.abs(loss)
        # update current loss
        loss = loss_prime
        
        # step counter
        step += 1

    return pi_t, Lambda_t, Sigma_t, Theta_t



##########################################################################################################################
#                                       Utility functions
#
##########################################################################################################################

# get Mixture of Gaussian parameter (no co features) from labelled data
def param_from_clusters(X, proposed_labels, centroides, l=0.1):
    # get dimensions of the problem
    K = len(centroides.centroid_id)
    p = len(centroides.component_y)
    n = len(X)
    
    # the xarray centroides is mu_0
    mu_0 = xr.DataArray(
        data=centroides,
        coords=[range(K), range(p)],    
        dims=['label', 'component_y'])

    # the proposed labels define the probability by class
    pi_0 = xr.DataArray(
        data = [float(np.sum(proposed_labels==k))/n for k in range(K)],
        coords=[range(K)], 
        dims=['label'])
    
    # get Sigma_0 and Lambda_0 from the initial clustering
    Sigma_0 = xr.DataArray(np.zeros((K, p, p)), 
                           coords=[range(K), range(p), range(p)], 
                           dims=['label', 'component_y', 'component_yT'])
    Lambda_0 = xr.DataArray(np.zeros((K, p, p)), 
                           coords=[range(K), range(p), range(p)], 
                           dims=['label', 'component_y', 'component_yT'])

    for k in range(K):
        # check that this is mean 0
        centered_observations = X[proposed_labels==k] - centroides.sel(centroid_id = k)
        if np.abs(float(centered_observations.mean())) > 1e-10:
            pass
            #print("Centroides not mean in clusters at the end of KMeans")
            #print("Centroides not mean in clusters, were they not from KMeans?")
            
        # build the transposition with new dim name to get a square matrix 
        #n_observations_cluster = len(centered_observations.coords["observation_id"])
        centered_observations_T = xr.DataArray(
            data = centered_observations,        
            coords = [centered_observations.observation_id, range(p)],        
            dims = ["observation_id", "component_yT"])

        # empirical covariance
        Sigma_0_cluster = xr.dot(centered_observations, centered_observations_T, 
                                 dims = "observation_id")/len(centered_observations.observation_id)

        # add random diagonal to make it invertible
        if len(centered_observations.observation_id)<p:
            Sigma_0_cluster = Sigma_0_cluster + l*np.diag(np.random.rand(p))
        # For computational reasons, n => p may not be enough 
        # for the matrix to be invertible
        if np.linalg.matrix_rank(Sigma_0_cluster) < p:
            Sigma_0_cluster = Sigma_0_cluster + l*np.diag(np.random.rand(p))
            
        Sigma_0.loc[dict(label=k)] = Sigma_0_cluster
        Lambda_0.loc[dict(label=k)] = np.linalg.inv(Sigma_0_cluster)

    return pi_0, mu_0, Sigma_0, Lambda_0

# initialisae class centroids as randomly sampled data points
def random_2D_parameters(Y, K, q):
    # get dimension
    p = len(Y.component_y)
    # random sample
    centroides = pd.DataFrame(Y.values).sample(n=K)
    # convert to data array
    centroides = xr.DataArray(np.float_(centroides), 
                              coords=[range(K), range(p)], 
                              dims=['centroid_id', 'component_y'])
    # get mean distance between each observation and centroide
    squared_distances = ((Y-centroides)**2).mean(dim="component_y")
    # get minimum distance for each observation
    proposed_labels = squared_distances.argmin(dim= "centroid_id")

    # get inital parameters from the KMeans
    pi_0, mu_0, Sigma_0, Lambda_0 = param_from_clusters(Y, proposed_labels, centroides)

    # Theta_0, just a fixed, neutral initialisation
    Theta_0 = xr.DataArray(
        data = np.array([[0,0], [0,0]]).reshape(K, q, p), 
        coords = [range(K),
                  range(q), 
                  range(p)], 
        dims = ["label",
                "component_x", 
                "component_y"] )
    return pi_0, mu_0, Sigma_0, Lambda_0, Theta_0

# plot confidence ellipses from gaussian covariance
def plot_ellipse(mu, Sigma, confidence_level = 0.95, unique_color = None, linewidth = 1):
    K = len(Sigma.label)
    current_palette = sns.color_palette()
    for k in range(K):
        D, V = np.linalg.eig(-2 * np.log(1 - confidence_level)*Sigma.sel(label=k).values)
        x = np.linspace(0, 2*np.pi)
        coordinates = (np.sqrt(D)*V).dot(np.array([np.cos(x), np.sin(x)]))
        if unique_color is None:
            plt.plot(mu.sel(label=k, component_y=0).values + coordinates[0], 
                     mu.sel(label=k, component_y=1).values + coordinates[1], 
                     color= current_palette[k])
        else:
            plt.plot(mu.sel(label=k, component_y=0).values + coordinates[0], 
                     mu.sel(label=k, component_y=1).values + coordinates[1], 
                     color= unique_color, linewidth = linewidth, zorder=0)