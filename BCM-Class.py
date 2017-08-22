
# coding: utf-8

# In[ ]:

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import random
class BCM:
    step = .10 #time step for integration
    nodes = 100 # number of neurons with in network
    stime = 1000 # run time for simulation in ms
    runtime = int(stime/step) # integration steps number
    nodeg = 5 #number of different inputs
    #we put small values for the intial weight matrix
    w_c=.01    
    tw=600# weighting time constant
    tt=4#theta time constant
    tr=1#rate time constant
    r0=np.zeros(nodes)
    w0=np.zeros((nodes,nodes))
    theta0=np.zeros(nodes) 
    for i in range(0,nodes):
        theta0[i]=0
        r0[i]=0
        for j in range(0,nodes):
            if i!=j:
                    w0[i,j]=w_c*random.random()  
                    

    def integrate(runtime,step,theta0,r0,w0,tw,tt,tr,nodeg,nodes,method):
        Frate = []
        
        def sig(x):
         
        	return 1/(1+np.exp(-x))
        clus=int(nodes/nodeg)
        c_in=np.ones((nodes, int(runtime)))*(-10)
        if method=='int':
            ip_time=10      # Duration of input ~ tau_r*10 
            # Define input
            ip_group=np.random.randint(0,nodeg,int(runtime))
            #I=np.zeros((N, int(T/dt)))
            c_in=np.ones((nodes, int(runtime)))*(-10)
            for k in range(len(ip_group)):
                c_in[clus*ip_group[k]:clus*(ip_group[k]+1),k*int(ip_time/step):(k+1)*int(ip_time/step)]=1
        else :
            if method=='sin' :
                for i in range(0,runtime):
                    for j in range(0,nodes-clus*(nodeg-1)):
                        c_in[j,i]=np.cos(((20)**(1/2.347))*i*.02+np.pi/8)
                        
                    for j in range(clus*(nodeg-4),clus*(nodeg-2)):
                        c_in[j,i]=np.cos(((5)**(1/2.347))*i*.02)
                        
                    for j in range(clus*(nodeg-3),clus*(nodeg-3)):
                        c_in[j,i]=np.sin(((9)**(1/2.347))*i*.02)
                        
                    for j in range(clus*(nodeg-2),clus*(nodeg-4)):
                        c_in[j,i]=np.cos(((6)**(1/2.47))*i*.02+np.pi/3)
                        
                    for j in range(clus*(nodeg-1),nodes):
                        c_in[j,i]=np.cos(((2)**(1/1.07))*i*.02+np.pi/6)
                        
                    c_in[j,i]=.1*(c_in[j,i])
            
            
            
        r=np.zeros((nodes,runtime))
        w=np.zeros(((nodes,nodes,runtime)))
        theta=np.zeros((nodes,runtime)) 

        theta[:,0]=theta0
        r[:,0]=r0
        w[:,:,0]=w0


        for t in range(0,runtime-1):
            # updating the weights rates
            w[:,:,t+1] = w[:,:,t]+step*(np.outer(r[:,t],r[:,t])*r[:,t]*(r[:,t]-theta[:,t]))/tw
            #updating the threshhold
            theta[:,t+1] = theta[:,t]+step*((r[:,t])**2-theta[:,t])/tt
            #Updating the firing rate
            r[:,t+1] = r[:,t]+step*(-r[:,t]+sig(c_in[:,t]+np.dot(w[:,:,t],r[:,t])))/tr
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

        r0=r[:,runtime-1]
        w0=w[:,:,runtime-1]
        theta0=theta[:,runtime-1]
	
        r=w=theta=np.zeros(1)

        return r0,w0,theta0

    
Fr=[]    
for i in range(0,15):
	if i<1:
		a,b,c = BCM.integrate(BCM.runtime,BCM.step,BCM.theta0,BCM.r0,BCM.w0,BCM.tw,BCM.tt,BCM.tr,BCM.nodeg,BCM.nodes,'sin')
	a,b,c =BCM.integrate(BCM.runtime,BCM.step,c,a,b,BCM.tw,BCM.tt,BCM.tr,BCM.nodeg,BCM.nodes,'sin')


plt.plot(Fr)
plt.show()





