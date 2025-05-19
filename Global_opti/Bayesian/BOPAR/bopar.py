#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys,time
import sklearn.gaussian_process as gp
import scipy.optimize

def minimize(fun,bounds,nb_retries,args):
    res=scipy.optimize.minimize(fun=fun,x0=np.random.uniform(bounds[0],bounds[1],size=1),bounds=[bounds],method='L-BFGS-B',args=args)
    best_fun_val=res.fun
    next_sample=res.x
    if nb_retries>1:
        for r in range(nb_retries-1):
            res=scipy.optimize.minimize(fun=fun,x0=np.random.uniform(bounds[0],bounds[1],size=1),bounds=[bounds],method='L-BFGS-B',args=args)
            if res.fun<best_fun_val:
                best_fun_val=res.fun
                next_sample=res.x
    return next_sample,best_fun_val

def maximize(fun,bounds,nb_retries,args):
    ifun = lambda *x : fun(*x)*-1
    xmin,ymin=minimize(ifun,bounds,nb_retries,args)
    return xmin,ymin*-1
    
def thresh_stop_crit(bo,dist_thresh=0.2,nb_thresh=4):
    nb_ot=0
    for g in bo.samples_coords:
        if abs(g-bo.best_coords)<dist_thresh:
            nb_ot+=1
            #print("selected : %f"%(g))
    if nb_ot>=nb_thresh:
        return True
    else:
        print("%d proximate on %d"%(nb_ot,nb_thresh))
        return False

def ucb(val,model):
    kappa=2
    mean,sigma=model.predict(val.reshape(-1,1),return_std=True)
    mean=mean.reshape(-1,)
    res=mean-kappa*sigma
    return res

def nbiter_stop_crit(bo,nb_iters=12):
    if bo.nb_iters == nb_iters:
        return True
    else:
        return False

possible_callbacks=["new_point","converged"]

class BOParOpt():
    """initialize the parallel Bayesian optimizer : f is the function to optimize, pbounds are the limits of the parameter space"""
    def __init__(self,f,pbounds):
        self.f=f
        self.pbounds=pbounds
        self.callback={}
        for c in possible_callbacks:
            self.callback[c]=[]

    def eff_acq_fun(self,x,P,amax):
        if P==[]:
            return self.acq_func(x,self.model)
        else:
            csp=P[0]
            rsp=P[1:]
            eff=self.eff_acq_fun(x,rsp,amax)
            k=self.model.kernel_([csp],np.expand_dims(x,1)).reshape(-1,)
            vals=self.acq_func(csp,self.model)[0]
            #print("shape of k : ",k.shape)
            #return np.minimum(amax,self.eff_acq_fun(x,rsp,amax)*(1-k)+k*amax)
            return np.minimum(amax,eff+k*(amax-vals))

    def comm_eff_acq_fun(self,x,P,amax):
        #print("shape of x : ",x.shape)
        eff=self.acq_func(x,self.model)
        #print("shape of eff : ",eff.shape)
        for i in range(len(P)):
            keri=self.model.kernel_([P[i]],np.expand_dims(x,1)).reshape(-1,)
            #print("shape of keri : ",keri.shape)
            vali=self.acq_func(P[i],self.model)[0]
            #print("shape of vali : ",vali.shape)
            #print("vali = ",vali)
            #sumi=0
            #for j in range(len(P)):
            #    valj=self.acq_func(P[j],self.model)[0]   
            #    kerij=self.model.kernel_([P[i]],[P[j]])[0][0]
            #    #print("kerij=",kerij)
            #    if j!=i:
            #        sumi+=(amax-valj)*kerij
            #print("sumi=",sumi)
            #eff+=keri*np.maximum(0,amax-vali-sumi)
            eff+=keri*np.maximum(0,amax-vali)
        return np.minimum(amax,eff)
        
    def subscribe(self,event,callback):
        if event in possible_callbacks:
            self.callback[event].append(callback)
        else:
            raise("unknown callback")
    def init_plot(self):
        plt.ion()
        self.fig,self.axs=plt.subplots(3,figsize=(10,12))
        self.fig.canvas.set_window_title("Parallel BO")
        self.plot_space=np.linspace(self.pbounds[0],self.pbounds[1],1000)
        self.plot_ker=np.expand_dims(np.linspace(-10, 10, 41),1)
    def plot(self):
        self.axs[0].clear()
        self.axs[0].plot(self.plot_space,self.f(self.plot_space),'b-')
        mean,sigma=self.model.predict(self.plot_space.reshape(-1, 1),return_std=True)
        sigma=sigma.reshape(-1, 1)
        self.axs[0].plot(self.plot_space,mean,'g-')
        low=mean-sigma
        high=mean+sigma
        self.axs[0].fill_between(self.plot_space,low.reshape(-1,),high.reshape(-1,),alpha=0.1)
        self.axs[0].plot(self.plot_space,mean-sigma,'g--')
        self.axs[0].plot(self.plot_space,mean+sigma,'g--')
        self.axs[0].scatter(self.samples_coords,self.samples_eval,c="blue",s=50,zorder=10)
        P_eval=self.f(np.array(self.P))
        if len(self.P)>0:
            self.axs[0].scatter(self.P,P_eval,c="orange",s=50,zorder=10)

        if self.converged:
            self.axs[0].scatter([self.best_coords],[self.best_eval],c="red",s=50,zorder=10) 
            
        self.axs[0].set_ylabel("Obj. func. + GP + guesses")
        self.axs[0].relim()
        self.axs[0].autoscale_view(True,True,True)

        #plot acquisition functions (real and effective)
        self.axs[1].clear()
        #plot the amax line
        maxy=np.full(self.plot_space.shape,self.af_ymax)
        self.axs[1].plot(self.plot_space,maxy,'m-')
        #plot eff acq func in red
        if self.mode=="batch" or self.mode=="debug":
            eaf=self.eff_acq_fun(self.plot_space,self.P,self.af_ymax)
            self.axs[1].plot(self.plot_space,eaf,'r-')
        if self.mode=="async" or self.mode=="debug":
            ceaf=self.comm_eff_acq_fun(self.plot_space,self.P,self.af_ymax)
            self.axs[1].plot(self.plot_space,ceaf,'g-')
        #plot orig acq func in cyan
        af=self.acq_func(self.plot_space,self.model)
        self.axs[1].plot(self.plot_space,af,'c-')
        #plot pending points 
        if len(self.P)>0:
            self.axs[1].scatter(self.P,self.acq_func(np.array(self.P),self.model),c="orange",s=50,zorder=10)
        self.axs[1].set_ylabel("acq. func.(cyan)\neff. acq. func.(red)")
        self.axs[1].relim()
        self.axs[1].autoscale_view(True,True,True)

        self.axs[2].clear()
        self.axs[2].set_ylim(0,1.1)
        S=self.model.kernel_(self.plot_ker,self.plot_ker)
        line8=self.axs[2].plot(self.plot_ker,S[:,20])
        self.axs[2].set_ylabel("GP Kernel")
        plt.waitforbuttonpress()

    def waitforever(self):
        print("Press Ctrl-C to finish")
        while True:
            self.plot() 
        
    def batch_minimize(self,batch_size,init_points,crit,acq_func,kernel=gp.kernels.RBF()):
        self.acq_func=acq_func
        self.mode="batch"
        self.mode="debug"
        
        #define the kernel used to compute self-correlation of the GP
        self.kernel=kernel
        #define the Gaussian process itself
        self.model=gp.GaussianProcessRegressor(kernel=self.kernel,alpha=1e-4,n_restarts_optimizer=10,normalize_y=True)

        #define the number of retries for the gradient descent
        self.nb_retries=50


        #define the set of so-far sampled points
        self.samples_coords=np.random.uniform(self.pbounds[0],self.pbounds[1],init_points)
        #parallel part
        self.samples_eval=self.f(self.samples_coords)

        #fit the GP with 
        self.model.fit(self.samples_coords.reshape(-1, 1),self.samples_eval.reshape(-1,1))
        self.converged=False
        self.best_eval=None
        self.best_coords=None
        self.nb_iters=0

        #main loop
        while not self.converged:
            #clear the set of sampling candidates
            self.P=[]
            #compute the maximum of the acquisition function
            self.af_xmax,self.af_ymax=maximize(self.acq_func,self.pbounds,self.nb_retries,args=(self.model))
            self.af_xmin,self.af_ymin=minimize(self.acq_func,self.pbounds,self.nb_retries,args=(self.model))

            #find the best samling coordinates
            for i in range(batch_size):
                #argmin of the effective acquisition function
                next_sample,best_fun_val=minimize(self.eff_acq_fun,self.pbounds,self.nb_retries,args=(self.P,self.af_ymax))

                diff=self.af_ymax-best_fun_val
                rapp=diff/(self.af_ymax-self.af_ymin)
                print("diff between min and Amax is %f, %f %% of Amin-Amax"%(diff,rapp*100))
                
                
                #add the result to the current sampling set
                self.P.append(next_sample)
                #print("next sample : %f"%(next_sample))
                #call the callbacks
                for c in self.callback["new_point"]:
                    c(self)

            #evaluation of the sampling coordinates : parallel part
            for new_coords in self.P:
                #evaluate the value
                new_eval=self.f(new_coords)
                
                # Update the samples lists
                self.samples_coords=np.append(self.samples_coords,new_coords)
                self.samples_eval=np.append(self.samples_eval,new_eval)
                
                #find the best
                if self.best_eval==None or new_eval<self.best_eval:
                    self.best_eval=new_eval
                    self.best_coords=new_coords
                    print("new best point %f,%f"%(self.best_coords,self.best_eval))

            #fit the GP
            self.model.fit(self.samples_coords.reshape(-1, 1),self.samples_eval.reshape(-1,1))
                
            #check termination
            self.converged=crit(self)
            if self.converged:
                print("Converged at %f,%f"%(self.best_coords,self.best_eval))
                self.P=[]
            self.nb_iters+=1

    def async_fill_P(self):
        #find the best samling coordinates
        while len(self.P)<self.nb_resource:
            #argmin of the effective acquisition function
            next_sample,best_fun_val=minimize(self.comm_eff_acq_fun,self.pbounds,self.nb_retries,args=(self.P,self.af_ymax))

            #test if it is pertinent to add the point
            if self.af_ymax==self.af_ymin:
                rapp=1
                print("flat acq func")
            else:
                diff=self.af_ymax-best_fun_val
                rapp=diff/(self.af_ymax-self.af_ymin)
                print("diff between min and Amax is %f, %f %% of Amin-Amax"%(diff,rapp*100)) 
            if rapp>=self.delta:
                self.P.append(next_sample)
            else:
                print("useless point")
                break

            self.plot()
            
            #else:
            #    print("useless point")
            #print("next sample : %f"%(next_sample))
            #call the callbacks
        for c in self.callback["new_point"]:
            c(self)
            
    def async_init_minimize(self,nb_resource,init_points,crit,acq_func,kernel=gp.kernels.RBF(),delta=0):
        self.acq_func=acq_func
        self.nb_resource=nb_resource
        self.crit=crit
        self.delta=delta
        self.mode="async"
        self.mode="debug"
        
        #define the kernel used to compute self-correlation of the GP
        self.kernel=kernel
        #define the Gaussian process itself
        self.model=gp.GaussianProcessRegressor(kernel=self.kernel,alpha=1e-4,n_restarts_optimizer=10,normalize_y=True)

        #define the number of retries for the gradient descent
        self.nb_retries=50


        #define the set of so-far sampled points
        self.samples_coords=np.random.uniform(self.pbounds[0],self.pbounds[1],init_points)
        #parallel part
        self.samples_eval=self.f(self.samples_coords)

        #fit the GP with 
        self.model.fit(self.samples_coords.reshape(-1, 1),self.samples_eval.reshape(-1,1))
        self.converged=False
        self.best_eval=None
        self.best_coords=None
        self.nb_evals=0

        #clear the set of sampling candidates
        self.P=[]
        #compute the maximum of the acquisition function
        self.af_xmax,self.af_ymax=maximize(self.acq_func,self.pbounds,self.nb_retries,args=(self.model))
        self.af_xmin,self.af_ymin=minimize(self.acq_func,self.pbounds,self.nb_retries,args=(self.model))

        #fill the queue
        self.async_fill_P()

         

                    
    def async_new_eval(self,new_coords,new_eval):

        self.nb_evals+=1
        
        # Update the samples lists
        self.samples_coords=np.append(self.samples_coords,new_coords)
        self.samples_eval=np.append(self.samples_eval,new_eval)
        self.P.remove(new_coords)
                
        #find the best
        if self.best_eval==None or new_eval<self.best_eval:
            self.best_eval=new_eval
            self.best_coords=new_coords
            print("new best point %f,%f"%(self.best_coords,self.best_eval))

        #fit the GP
        self.model.fit(self.samples_coords.reshape(-1, 1),self.samples_eval.reshape(-1,1))
        #compute the maximum of the acquisition function
        self.af_xmax,self.af_ymax=maximize(self.acq_func,self.pbounds,self.nb_retries,args=(self.model))
        self.af_xmin,self.af_ymin=minimize(self.acq_func,self.pbounds,self.nb_retries,args=(self.model))
                
        #check termination
        self.converged=self.crit(self)
        if self.converged:
            print("Converged at %f,%f in %d evaluations"%(self.best_coords,self.best_eval,self.nb_evals))
            self.P=[]
            for c in self.callback["converged"]:
                c(self)
            return

        #fill the queue
        #print("fill the queue")
        #self.async_fill_P()
