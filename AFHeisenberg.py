
import numpy as np

def random_complex(size):
    a = (np.random.random(size) - .5) * 10e-2
    b = (np.random.random(size) - .5) * 10e-2
    return a + 1j*b

N = 4
alpha = 2
M = alpha * N
hfield = 2
a = random_complex(N)
b = random_complex(M)
W = random_complex((N,M))
state = np.random.randint(2, size=N)
state[state == 0] = -1
state_i = list(range(N))
AFState = [1,-1,1,-1]

# Neural Network model for Wave Function 


def effective_angles(state):
    return b+np.inner(np.transpose(W),state)

def Psi_M(state,a,b,W):
    return np.exp(np.inner(a,state)) * np.prod(2*np.cosh(effective_angles(state)))


# Energy 


def E_loc(state):
    E = 0
    # \sigma^z part
    for i in state_i:
        if i == N-1:
            E+=(state[i]*state[0])
        else:
            E+=(state[i]*state[i+1])

    # \sigma^x part
    Psi_M_s = Psi_M(state,a,b,W)
   
    for i in state_i:
        if i == N-1:
            state[i] *= -1
            state[0] *= -1
            E += (1-(state[i]*state[0]))*Psi_M(state,a,b,W)/Psi_M_s
            state[i] *= -1
            state[0] *= -1
        else:
            state[i] *= -1
            state[i+1] *= -1
            E += (1-(state[i]*state[0]))*Psi_M(state,a,b,W)/Psi_M_s
            state[i] *= -1
            state[i+1] *= -1

    return E/N


# In[4]:


def step(state):
    # Choose random sites to be flipped
    sites = np.random.choice(state_i,1)
    Psi_M_before = Psi_M(state,a,b,W)
    E_loc_before = np.real(E_loc(state))
    for i in sites:
        state[i] *= -1 # flip
    Psi_M_after = Psi_M(state,a,b,W)
    E_loc_after = np.real(E_loc(state))
    
    acceptance = np.real(Psi_M_after*np.conj(Psi_M_after)/(Psi_M_before*np.conj(Psi_M_before)))

    if acceptance < np.random.uniform() and E_loc_after >= E_loc_before:
        for i in sites: 
            state[i] *= -1 # flip back
        return(state) # return 1 to count # of rejections
    else:
        return(state)


# In[5]:


block_E = []
block_Ef = []
for block_i in range(50):
    state = np.random.randint(2, size=N)
    state[state == 0] = -1
    for k in range(1000):
        state = step(state)
    
    iterations = 200000
    rejected = 0
    array_E_loc = []
    array_EAFS_loc = []
    
    array_a_d = []
    array_b_d = []
    array_w_d = []

    for k in range(iterations):
        #rejected += step()
        
        if k % 1000 == 0:
            state = step(state)
            Psi_M_s = Psi_M(state,a,b,W)
            
            # Derivative a
            a_deriv = np.zeros(N, dtype=np.complex_)
            for i in range(N):
                if i == N-1:
                    state[0] *= -1
                    state[i] *= -1 # flip
                    a_deriv[i] = Psi_M(state,a,b,W)/Psi_M_s*2.*(1-state[i]*state[0])*-2*state[i]
                    state[0] *= -1
                    state[i] *= -1 # flip back
                else:
                    state[i] *= -1
                    state[i+1] *= -1
                    a_deriv[i] = Psi_M(state,a,b,W)/Psi_M_s*2.*(1-state[i]*state[i+1])*-2*state[i]
                    state[i] *= -1
                    state[i+1] *= -1 # flip back
          
            # Derivative W
            dW = np.zeros((N,M),dtype=np.complex_)
            for w_i in range(N):
                for w_j in range(M):
                    dw_sum = 0
                    before_flip = np.tanh(effective_angles(state))
                    for i in range(N):
                        if i == N-1:
                            state[i] *= -1
                            state[0] *= -1 # flip
                            dw_sum += Psi_M(state,a,b,W)/Psi_M_s*(
                            -state[i]*np.tanh(effective_angles(state)[w_j])-state[i]*before_flip[w_j])
                            dw_sum *= (1-state[i]*state[0])
                            state[i] *= -1
                            state[0] *= -1 # flip back
                        else:
                            state[i] *= -1
                            state[i+1] *= -1 # flip
                            dw_sum += Psi_M(state,a,b,W)/Psi_M_s*(
                            -state[i]*np.tanh(effective_angles(state)[w_j])-state[i]*before_flip[w_j])
                            dw_sum *= (1-state[i]*state[i+1])
                            state[i] *= -1
                            state[i+1] *= -1 # flip back

                    dW[w_i,w_j] = dw_sum
            
            # Derivative b
            b_deriv = np.zeros(M, dtype=np.complex_)
            for b_j in range(M):
                tanh_before_flip = np.tanh(effective_angles(state))
                db_sum = 0
                for i in range(N):
                    if i == N-1:
                        state[i] *= -1
                        state[0] *= -1 # flip
                        db_sum += Psi_M(state,a,b,W)/Psi_M_s*(
                        np.tanh(effective_angles(state)[b_j])-tanh_before_flip[b_j])
                        db_sum *= (1-state[i]*state[0])
                        state[i] *= -1 
                        state[0] *= -1# flip back
                    else:
                        state[i] *= -1
                        state[i+1] *= -1 # flip
                        db_sum += Psi_M(state,a,b,W)/Psi_M_s*(
                        np.tanh(effective_angles(state)[b_j])-tanh_before_flip[b_j])
                        db_sum *= (1-state[i]*state[i+1])
                        state[i] *= -1 
                        state[i+1] *= -1# flip back

                b_deriv[b_j] = db_sum 
                    
            
            array_a_d.append(a_deriv)
            array_b_d.append(b_deriv)
            array_w_d.append(dW)
            array_E_loc.append(np.real(E_loc(state)))
            array_EAFS_loc.append(np.real(E_loc(AFState)))
            
   # print('%d. E_loc=%.4f std=%.4f (%.1f %% moves rejected)' % (block_i+1,
    #    np.mean(array_E_loc),np.std(array_E_loc)/(np.sqrt(len(array_E_loc))), 100.*rejected/iterations))
   # print(state)
    block_E.append(np.mean(array_E_loc))
    block_Ef.append(np.mean(array_EAFS_loc))
    mean_da = np.mean(np.array(array_a_d),axis=0)
    mean_db = np.mean(np.array(array_b_d),axis=0)
    mean_dw = np.mean(np.array(array_w_d),axis=0)
    #print(mean_da,mean_db,mean_dw)
    a = a - .05 * mean_da
    b = b - .05 * mean_db
    W = W - .05 * mean_dw
    #print(a,b,W)


# In[10]:


import matplotlib.pyplot as plt
#plt.plot(block_E)
plt.plot(block_Ef,label = 'Predicted $E_{g}$')

#plt.legend()
plt.title(r'Local energy $\frac{<E_{loc}>}{N}$')
plt.ylabel(r'$<E_{loc}>/ N$')
plt.xlabel('iterations')



# In[11]:


from itertools import product

basis = list(product([-1,1],repeat=N))

print('Generated %d basis functions' % (len(basis)))
#print(len(basis_functions))

#list(permutations([0,1,0,0]))
H = np.zeros((2**N,2**N))
for H_i in range(2**N):
    for H_j in range(2**N):
        H_sum = 0
        for i in range(N):
            if H_i == H_j:
                if i == N-1:
                    H_sum += basis[H_j][i]*basis[H_j][0]
                else:
                    H_sum += basis[H_j][i]*basis[H_j][i+1]
                    
            sj = list(basis[H_j])
            sj[i] *= -1
            if i == N-1:
                sj[0] *= -1
            else: 
                sj[i+1]*= -1
            if H_i == basis.index(tuple(sj)):
                if i == N-1:
                    H_sum += 1-basis[H_i][i]*basis[H_i][0]
                else:
                    H_sum += 1-basis[H_i][i]*basis[H_i][i+1]

        H[H_i,H_j] = H_sum
            
print('Ground state energy:', np.min(np.linalg.eigvals(H))/N)
G_E = np.ones(50)
G_E = G_E*np.min(np.linalg.eigvals(H))/N
plt.plot(G_E,'r.',label = 'Actual $E_{g}$')
plt.legend()
plt.show()
AFState = [1,-1,1,-1]
print('Eg=',np.real(E_loc(AFState)))

# In[ ]:




