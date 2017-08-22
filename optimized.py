
# coding: utf-8

# In[28]:

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import random
step = .010 #time step for integration
nodes = 100 # number of neurons with in network
stime = 200 # run time for simulation in ms
runtime = int(stime/step) # integration steps number
nodeg = 5 #number of different inputs

class BCM:

                    
    def sig(x):
        y = 1/(1+np.exp(-x))
        return y



    def integrate(runtime,step):
        Frate = []
        
        

        ip_time=10      # Duration of input ~ tau_r*10 (?)tau_theta
        # Define input
        ip_group=np.random.randint(0,nodeg,int(runtime))
        #I=np.zeros((N, int(T/dt)))
        c_in=np.ones((nodes, int(runtime)))*(-10)
        for k in range(len(ip_group)):
            c_in[20*ip_group[k]:20*(ip_group[k]+1),k*int(ip_time/step):(k+1)*int(ip_time/step)]=1


        
        tw=600# weighting time constant
        tt=4#theta time constant
        tr=1#rate time constant
        r=np.zeros((nodes,runtime))
        w=np.zeros(((nodes,nodes,runtime)))
        theta=np.zeros((nodes,runtime)) 
        #we put small values for the intial weight matrix
        w_c=.01
        for i in range(0,nodes):
            theta[i,0]=0
            r[i,0]=0
            for j in range(0,nodes):
                if i!=j:
                        w[i,j,0]=w_c*random.random()

        for t in range(0,runtime-1):
            # updating the weights rates
            w[:,:,t+1] = w[:,:,t]+step*(np.outer(r[:,t],r[:,t])*r[:,t]*(r[:,t]-theta[:,t]))/tw
            #updating the threshhold
            theta[:,t+1] = theta[:,t]+step*((r[:,t])**2-theta[:,t])/tt
            #Updating the firing rate
            r[:,t+1] = r[:,t]+step*(-r[:,t]+BCM.sig(c_in[:,t]+np.dot(w[:,:,t],r[:,t])))
            #making the weight positive
            w[:,:,t+1] = np.minimum(w[:,:,t+1],np.ones((nodes,nodes)))
            #keeping the track of firing rates
            Frate.append(np.mean(r[:,t+1]))
            #making sure that each neuron has not link to itself
            np.fill_diagonal(w[:,:,t+1],0)
            #print(BCM.sig(c_in[:,t]+np.dot(w[:,:,t],r[:,t])))
        W=w[:,:,runtime-1]
        plt.imshow(W)
        plt.colorbar()
        plt.show()
        plt.plot(Frate)
        plt.show()
        return #w[:,:,t+1],theta[:,t+1],r[:,t+1],Frate

BCM.integrate(runtime,step)



# In[ ]:



