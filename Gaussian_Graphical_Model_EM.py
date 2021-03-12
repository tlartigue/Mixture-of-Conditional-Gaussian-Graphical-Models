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


################## functions used to define the GGL penalised EM for a regular GGM model (no cofeature) ##################

# imports
import numpy as np
import pandas as pd
import xarray as xr


##########################################################################################################################
#                          GGM with GGL penalty optimisation scheme (Alternating Direction Method of Multipliers )
#
#                                  with given class labels or class weights (supervised)
#
#                                               used for the M step of the EM 
##########################################################################################################################


# #### Optimisation functions

def soft_threshold(x, l):
    return np.maximum(x-l, 0) - np.maximum(-x-l,0)

def reduced_Q(Lambda, S, N_t):
    # simply log det - trace, no 2pi, no pi_t
    
    K = len(Lambda.label)
    # det(Lambda)
    determinants = xr.DataArray([np.linalg.det(Lambda[k]) for k in range(K)], 
                                coords=[range(K)], 
                                dims=['label'])

    # trace( Lambda.S)
    traces = xr.dot(Lambda, S, dims=["component_y", "component_yT" ])

    # reduced Q function (part depending of Lambda only)
    return np.float(np.sum(0.5*(np.log(determinants) - traces)*N_t))
    

def GGL_penalty(Lambda, l1, l2):
    # make a copy and remove the diagonal that we won't penalise
    Lambda_prime = Lambda.copy()
    p = len(Lambda.component_y)
    for k in Lambda_prime.label:
        Lambda_prime.loc[dict(label=k)] = Lambda_prime.sel(label=k) - np.eye(p)*np.diag(Lambda_prime.sel(label=k))
    return np.float(l1*np.abs(Lambda_prime).sum() + 
                    l2*((Lambda_prime**2).sum(dim = 'label')**0.5).sum())

def GGL_loss(Lambda, S, N_t, l1, l2):
    # - log likelihood w/ GGL penalty
    return -2.*reduced_Q(Lambda, S, N_t) + GGL_penalty(Lambda, l1, l2)

# augmented lagrangian of the GGL Loss (ADMM)
# Careful: this function guides the ADMM procedure, but is not minimised by it
def augmented_lagrangian(Lambda, Z, U, S, N_t, l1, l2, rho):
    return float(-2*reduced_Q(Lambda, S, N_t) + GGL_penalty(Z, l1, l2) + 
                 rho/2 * ((Lambda-Z+U)**2).sum() - rho/2 * (U**2).sum())

# The specific part of the lagrange loss that Z optimises
# used for debugging purposes, to check that the function was indeed decreasing
def loss_Z_opti(Z, A, l1, l2, rho):
    return float(rho/2 *((Z-A)**2).sum() + GGL_penalty(Z, l1, l2) 
           )

# the ADMM update with GGL penalty
def update_admm_GGL(S, N_t, Lambda, Z, U, rho, l1, l2):
    K =  len(S.label)
    Lambda_prime = Lambda.copy()
    
    # update 1: Lambda
    for k in range(K):
        # if the cluster k has not vanished
        if N_t.sel(label=k)!=0:
            # eigen decomposition
            u, V = np.linalg.eigh((S - rho*Z/N_t + rho*U/N_t).sel(label = k))
            n_k = np.float(N_t.sel(label=k))

            # update diagonl
            u_tilde = n_k/(2*rho) * ( - u + (u**2 + 4*rho/n_k)**0.5  )

            # update rule:
            Lambda_prime.loc[dict(label = k)] =  V.dot(np.diag(u_tilde)).dot(V.T)

        # if the cluster k has vanished
        # Lambda_prime.loc[dict(label = k)] = Lambda.sel(label=k)
        # which is already done by the initialisation

    # update 2: Z
    # GGL penalty formula
    A = Lambda_prime + U
    SftThr = soft_threshold(A, l1/rho)
    # Z_ij: formula
    Z_prime = SftThr * np.maximum(1 - l2/(rho * (SftThr**2).sum(dim="label")**0.5 ), 0)
    # Z_ii = A_ii
    for k in range(K):
        np.fill_diagonal(Z_prime.loc[dict(label = k)].values, np.diag(A.sel(label=k)))

    # update 3: U
    U_prime = U + Lambda_prime - Z_prime
    
    return Lambda_prime, Z_prime, U_prime

# supervised joint GGM optimisation with GGL penalty
# not used in EMs, but for supervised problems
def hierarchical_ggm(Y, true_labels, rho, l1, l2,
                     Lambda_shift_threshold, loss_shift_threshold, verbose=False):
    p = len(Y.component_y)
    K = len(true_labels.label)

    # get number of observations by class
    N= true_labels.sum(dim="observation_id")

    # get empirical mu
    mu = (Y*true_labels).sum(dim="observation_id")/N
    # get empirical sigma (S)
    centered_observations = Y - mu
    centered_observations_T  = xr.DataArray(
                data = centered_observations,        
                coords = [centered_observations.observation_id, range(p), range(K)],        
                dims = ["observation_id", "component_yT", "label"])

    S = xr.dot(true_labels *centered_observations , centered_observations_T, 
               dims=['observation_id']) /N
    
    # penalised optimisation, GGL
    Lambda = xr.DataArray([np.eye(p)]*K,
                         coords=S.coords, 
                         dims=S.dims)
    Z = xr.DataArray(np.zeros((K, p, p)),
                     coords=S.coords, 
                     dims=S.dims)
    U = xr.DataArray(np.zeros((K, p, p)),
                     coords=S.coords, 
                     dims=S.dims)
    Lambda_shift = 1
    loss_shift = 1
    # while loop with convergence check
    while ((Lambda_shift > Lambda_shift_threshold or loss_shift >loss_shift_threshold)):
           #and loss_shift > 0):
        # if penalty == "GGL"
        loss = GGL_loss(Lambda, S, N, l1, l2)
        if verbose:
            print(loss)
        # one ADMM update
        #if penalty == "GGL":   
        Lambda_prime, Z, U = update_admm_GGL(S, N, Lambda, Z, U, rho, l1, l2)
        loss_prime = GGL_loss(Lambda_prime, S, N, l1, l2)

        loss_shift = (loss - loss_prime)/np.abs(loss)
        Lambda_shift = np.float(np.sum(np.abs(Lambda_prime - Lambda))/np.sum(np.abs(Lambda)))
        if verbose:
            print(Lambda_shift)
            print(loss_shift)
            print()

        Lambda= Lambda_prime
        loss = loss_prime

    return mu, Lambda


##########################################################################################################################
#                         Unsupervised (Mixture) likelihood and other unsupervised objective function
#
#                                               used for the E step of the EM 
##########################################################################################################################


# #### Likelihood and other objective functions

# generic gaussian log likelihood formula
def gaussian_log_likelihood(Y, mu, Lambda):
    # Y size: n x p
    # mu size: K x p
    # Lambda size: K x p x p
    # returns the log of the gaussian density of each observation in x 
    # for each values of the parameter across all clases
    # output size: n x K
    
    K=len(mu.label)
    p=len(mu.component_y)
    n=len(Y.observation_id)
    
    # (x-mu)^T Lambda (x-mu)
    dotproduct = xr.DataArray(xr.dot(Lambda, Y - mu, dims=['component_y']),
                              coords=[range(K), range(p), range(n)], 
                              dims=['label', 'component_y', 'observation_id'])
    dotproduct_final = xr.dot(Y - mu, dotproduct , dims=['component_y'])

    # det(Lambda)
    determinants = xr.DataArray([np.linalg.det(Lambda[k]) for k in range(K)], 
                                coords=[range(K)], 
                                dims=['label'])

    #Sigma_da.reduce(lambda x, axis: np.linalg.det(x), dim = 'label')

    # numerator of the Bayes rule to get p_i,k^t
    log_p = 0.5 * (- dotproduct_final + np.log(determinants) - p * np.log(2 * np.pi)) 
    return log_p

# weights p_i,k^t = p(y_i | x_i, theta_t) in the Q function (E step)
def label_weights_E_step(Y, pi_t, mu_t, Lambda_t):
    # return the p_i,k^t = p(y_i | x_i, theta_t)
    # size of the output: n x K
    # get dimensions of the problem
    p = len(mu_t.component_y)
    K = len(pi_t)
    n = len(Y.observation_id)
    
    # first the gaussian likelihood for each observation for each class
    # size: n x K
    log_p = gaussian_log_likelihood(Y, mu_t, Lambda_t)
    
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
        # trying to preserve code integrity by resetting the proba for obs. where there
        # are NANs
        p_t[np.isnan(p_t.sum(dim="label", skipna=False))] = 1./K
    elif not (np.abs(p_t.sum(dim='label').values -1) <= 1e-10).all():
        print("Computational error w/ low probabilities at E step")        
    if  (p_t.sum(dim='observation_id').values == 0).any():
        print("Classes w/ 0 mass assigned to them at E step")
        
    return p_t

# function Q(theta|theta_t)
def Q(Y, pi, mu, Lambda, pi_t, mu_t, Lambda_t):
    # get the class weights w/ param theta_t
    # p(y_i | x_i, theta_t)
    p_t = label_weights_E_step(Y, pi_t, mu_t, Lambda_t)
    # get the log likelihood by class w/ param theta
    # log P_theta(x_i, z_i = k) = log P_theta(x_i | z_i = k) + log(pi_k)
    log_likelihood = gaussian_log_likelihood(Y, mu, Lambda) + np.log(pi)
    return np.float(np.sum(p_t *log_likelihood))

# Unsupervised, Mixture of GGM negative log likelihood
def UGGM_neg_log_likelihood(Y, pi, mu, Lambda):
    # likelihood by label for each observation
    # size log_likelihood: n x K
    log_likelihood = gaussian_log_likelihood(Y, mu, Lambda) + np.log(pi)

    # direct way
    #return np.float(np.sum(np.log(np.exp(log_likelihood).sum(dim = "label"))))

    # dodge computation errors way
    # there should be no computation errors, the smaller values are put to 0,
    # that is all, there is no infinity here
    return -1.*np.float(np.sum(
        log_likelihood.max(dim= 'label') + 
        np.log(np.exp(log_likelihood - log_likelihood.max(dim= 'label')).sum(dim='label'))
    ))


# penalised observed negative log likelihood: the function maximised by the EM
def UGGM_full_loss(Y, pi, mu, Lambda, 
                   # penalty parameters
                   l1=0.1, l2=0.05):
    # divide by n*K*p**2 to display an easier to interpret number
    return (UGGM_neg_log_likelihood(Y, pi, mu, Lambda)- 0.5*GGL_penalty(Lambda, l1, l2))/(len(Lambda.label)*len(Y.observation_id)*len(Lambda.component_y)**2)
    
    

##########################################################################################################################
#                                       E step and M step together: the EM algorithm 
#
##########################################################################################################################
    
# One EM step
def EM_step(Y, 
            # penalty intensities
            l1, l2, 
            # current parameters
            pi_t0, mu_t0, Lambda_t0, 
            # optimiser parameter
            rho,
            # stopping criteria of the proximal gradient descent
            loss_shift_threshold=1e-3, 
            Lambda_shift_threshold=1e-3,
            # additional parameters
            verbose=False):

    # one EM step from _0 parameters

    # get dimensions of the problem
    p = len(mu_t0.component_y)
    K = len(pi_t0)
    n = len(Y.observation_id)
    # proba p_i,k^t
    p_t0 = label_weights_E_step(Y, pi_t0, mu_t0, Lambda_t0)

    # weight of each class at time t
    N_t0 = p_t0.sum(dim='observation_id')

    # optimise in (pi, mu, Lambda) jointly: possible if there is no penalty on mu
    # since:

    # pi is always free
    # update pi
    pi_t1 = N_t0/n

    # mu as well when it's not constrained
    # update mu with no constraints
    mu_t1 = (p_t0*Y).sum(dim="observation_id") / N_t0

    # define S
    centered_observations = Y - mu_t1
    centered_observations_T  = xr.DataArray(
                data = centered_observations,        
                coords = [centered_observations.observation_id, range(p), range(K)],        
                dims = ["observation_id", "component_yT", "label"])


    S = xr.dot(centered_observations * p_t0, centered_observations_T, 
               dims=['observation_id']) / N_t0

    ### if not penalised: just take Lambda=S^{-1} (you need rk(S_k)>p)
    if (l1 == 0) and (l2 ==0):
        Lambda_t1 = xr.DataArray(data = [np.eye(p)]*K,
                                coords = Lambda_t0.coords, 
                                dims = Lambda_t0.dims)
        l=1e-7
        for k in range(K):

            # regularize if rk(S_k) < p
            S_k = S.sel(label=k)+(np.linalg.matrix_rank(S.sel(label=k))<p)*l*np.eye(p)
            # invert the result
            Lambda_t1.loc[dict(label=k)] = np.linalg.inv(S_k)
        
    ### if penalised: we need ADMM
    else:
        ##### Now the real ADMM starts
        # Lambda, Z and U
        # three updates

        # init
        Lambda = xr.DataArray([np.eye(p)]*K,
                             coords=Lambda_t0.coords, 
                             dims=Lambda_t0.dims)
        Z = xr.DataArray(np.zeros((K, p, p)),
                         coords=Lambda_t0.coords, 
                         dims=Lambda_t0.dims)
        U = xr.DataArray(np.zeros((K, p, p)),
                         coords=Lambda_t0.coords, 
                         dims=Lambda_t0.dims)
        Lambda_shift = 1
        loss_shift = 1
        # while loop with convergence check
        while ((Lambda_shift > Lambda_shift_threshold or loss_shift >loss_shift_threshold)):
               #and loss_shift > 0):
            # if penalty == "GGL"
            loss = GGL_loss(Lambda, S, N_t0, l1, l2)
            if verbose:
                print(loss)
            # one ADMM update
            #if penalty == "GGL":   
            Lambda_prime, Z, U = update_admm_GGL(S, N_t0, Lambda, Z, U, rho, l1, l2)
            loss_prime = GGL_loss(Lambda_prime, S, N_t0, l1, l2)

            loss_shift = (loss - loss_prime)/np.abs(loss)
            Lambda_shift = np.float(np.sum(np.abs(Lambda_prime - Lambda))/np.sum(np.abs(Lambda)))
            if verbose:
                print(Lambda_shift)
                print(loss_shift)
                print()

            Lambda= Lambda_prime
            loss = loss_prime

        Lambda_t1 = Lambda
    
    return pi_t1, mu_t1, Lambda_t1

# EM full procedure
def EM(Y, 
       # number of classes desired
       K, 
       # penalty on the precision matrix
       l1, l2, 
       # initial values of the parameters
       pi_0, mu_0, Lambda_0, 
       # optimiser parameter
       rho = 1,
       # stopping criteria of the EM
       loss_shift_threshold_EM=1e-4, Lambda_shift_threshold_EM=1e-4,
       # stopping criteria of the ADMM in the M step
       loss_shift_threshold_ADMM=1e-4, Lambda_shift_threshold_ADMM=1e-4,
       # do we want the prints?
       verbose = False, 
       frequence_print = 1, 
       # maximum number of steps
       max_steps=200):

    # initial values
    pi_t, mu_t, Lambda_t = pi_0.copy(), mu_0.copy(), Lambda_0.copy()
    
    # initialise number of steps
    step = 0
    print_counter = 0

    # make sure that we enter the loop
    loss_shift = loss_shift_threshold_EM+1
    Lambda_shift = Lambda_shift_threshold_EM+1
    
    while (loss_shift > loss_shift_threshold_EM or  Lambda_shift > Lambda_shift_threshold_EM) & (step<max_steps):
        # current loss
        loss = UGGM_full_loss(Y, pi_t, mu_t, Lambda_t, l1, l2)
        # if wanted, display regularly the progress in log likelihood
        if verbose &(step*frequence_print >= print_counter):
            print(f"negative log likelihood at step {step} = {loss}")
            print_counter += 1
        
        # make an EM step
        pi_t, mu_t, Lambda_t1 = EM_step(Y, l1, l2, 
                                        # current parameters
                                       pi_t, mu_t, Lambda_t, 
                                        # ADMM settings
                                       rho, loss_shift_threshold_ADMM, Lambda_shift_threshold_ADMM)
        # new loss
        loss_prime = UGGM_full_loss(Y, pi_t, mu_t, Lambda_t1, l1, l2)
        
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

    return pi_t, mu_t, Lambda_t