# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 11:04:26 2023

@author: swarn
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import time


def lor_ros(states,t,r,eps):
    
    [xd,yd,zd,xr,yr,zr] = states[:]
    dx = np.zeros(states.shape)
    
    eta1 = 0.2
    eta2 = 0.2
    eta3 = 5.7
    sigma = 10
    beta = 8.0/3.0
    
    dx[0] = sigma*(yd - xd)
    dx[1] = r*xd - yd - xd*zd
    dx[2] = xd*yd - beta*zd
    dx[3] =  (-yr - zr)
    dx[4] = xr + eta1*yr
    dx[5] = eta2 + zr*(xr - eta3) + eps*(zd - zr)
    return dx


def ros_ros(states,t,r,eps):
    
    [xd,yd,zd,xr,yr,zr] = states[:]
    dx = np.zeros(states.shape)
    
    eta1 = 0.2
    eta2 = 0.2
    eta3 = 5.7
    
    dx[0] =  (-yd - zd)
    dx[1] = xd + eta1*yd
    dx[2] = eta2 + zd*(xd - r)
    dx[3] =  (-yr - zr)
    dx[4] = xr + eta1*yr
    dx[5] = eta2 + zr*(xr - eta3) + eps*(zd - zr)
    return dx





def model(param,train_sys):
    dim = 6
    
    x0 = np.random.random(dim)
    t = np.arange(0,1000,0.01)
    eps = 0.3
    
    
    n_col = 4   #Number of state variable to return
    x = np.zeros((len(t),len(param)*n_col))
    
    for i in range(len(param)):
        x[:,n_col*i:n_col*(i+1)] = odeint(train_sys,x0,t,args=(param[i],eps))[:,2:]  #[zd,xr,yr,zr]
        
    return x[10000:,:]



if __name__ == '__main__':
    
    start_time = time.time()
    
    
    tr_sys = ros_ros            # define training system
    param = np.array([5,10,15]) # training system parameters
    
    print("Preparing training data")
    data = model(param,tr_sys).T
    print("Done\n")
    
    
########  Define the parameter/hyper-parameters ################    

    N = 1200            #reservoir size
    res_rho = 0.5057    #Spectral radius
    res_k = 20          #Reservoir degree of connection
    W_in_a = 0.0639     #Range of input connection
    alpha = 0.6057      #Leaking parameter
    beta = 4.7487e-5    #Regularization parameter

    train_len = 60000          #Training length
    pred_len = 20000           #Predicting length
    trans = 1000               #Number of initial reservoir states to discard
    
    dim = 4         #Input dimension
    d_dim = 1       #Drive signal dimension



#..................Making Reservoir....................

    print("Making Reservoir")
    W = np.zeros((N,N))
    pr = res_k/(N-1)
    
    for i in range(N):
        for j in range(N):
            if((i!=j) and (np.random.random()<pr)):
                W[i,j] = np.random.random()
                
    eig_val,eig_vec = np.linalg.eig(W)
    m = np.abs(eig_val).max()
    
    W = (res_rho/m)*W
    print("Done\n")
    
    
    
#...................Input connection....................
    W_in = np.zeros((N,dim))
    
    for i in range(N):
        W_in[i,int(i*dim/N)] = W_in_a*(2*np.random.random() - 1)
            
    

#................Training the Reservoir................    
    
    store = train_len - trans
    R = np.zeros((N,len(param)*store))
    T = np.zeros((dim-1,len(param)*store))
    
    for ii in range(len(param)):
        print("Traning the reservoir for parameter %1.1f"%param[ii],end='\r')
        m = np.random.randint(2000)  #picking random intial point for training time series
        u = data[dim*ii:dim*(ii+1),m:m+train_len+1]
        
        x = np.zeros(N)
        xt = np.zeros(N)
        
        for i in range(train_len):            
            x = (1 - alpha)*x + alpha*np.tanh(np.dot(W_in,u[:,i]) + np.dot(W, x))
            xt[:] = x[:]
            xt[::2] = x[::2]**2
            
            if(i>=trans):
                R[:,(ii*store)+i-trans] = xt
        
        T[:,store*ii:store*(ii+1)] = u[1:dim,trans+1:train_len+1]
        
    print("\nTraining completed.\n")
    
    
#..............................Regression............................    
    W_out = np.dot(np.dot(T,R.T),np.linalg.inv((np.dot(R,R.T)+beta*np.identity(N))))
    # W_out =  np.dot(T,np.linalg.pinv(R))  
    # np.savetxt("w_out.txt",W_out)
    
    
    
#.........................Predicting Phase.......................            
    
    pred_len = 10000
    tran = 1000
    warmup = 100
    
    
    ## .............Preparing testing data.......................
    x0 = np.random.random(6)
    t = np.arange(0,(pred_len+tran)*0.01,0.01)
    test_data = odeint(lor_ros,x0,t,args=(38,0.3)).T
    
    test_data = test_data[:,tran:]  #Remove transient
    u = test_data[2:,0]     #Initial state for prediction


    pred = np.zeros((3,pred_len))
    x = np.zeros(N)
    xt = np.zeros(N)
    
    for i in range(warmup+pred_len):
        x = (1 - alpha)*x + alpha*np.tanh(np.dot(W_in,u) + np.dot(W, x))
        xt[:] = x[:]
        xt[::2] = x[::2]**2
        
        if(i>=warmup):
            pred[:,i-warmup] = u[-3:]
            u = np.dot(W_out,xt)    #Predicted response state at each time point
            u = np.append(test_data[2,i-warmup],u) #Taking drive signal from similated data
            
    print("\nTraining completed.\n")
    
        
    org = test_data[3:,:] #Actual response dynamics
    
    req_time = time.time() - start_time
    print('\nCalculation time: ',req_time)
    
    err = ((np.mean((org[0,:] - pred[0,:])**2))**0.5)/((np.max(org[0,:])-np.min(org[0,:]))*pred_len)
    print(err)
    
    plt.figure()
    plt.plot(pred[0,:])
    plt.plot(org[0,:])
    plt.show()