import numpy as np
def random_complex(size):
    a = (np.random.random(size) - .5) * 10e-2
    b = (np.random.random(size) - .5) * 10e-2
    return a + 1j*b
N = 2
H = 2
O = 2
M1 = N*H
M2 = H*O
t = 10
#U = 1
a = random_complex(N)
b = random_complex(N)
W1 = random_complex((N,H))
W2 = random_complex((H,O))
n = np.random.randint(3)
state =[n,2-n]


P_state = [[2,0],[1,1],[0,2]]

#***************************************

def Angle(state):
    return(np.inner(np.transpose(W1),state)+a)

def Wave_function(state,W1,W2,a,b):
    U_1 = np.tanh(np.inner(np.transpose(W1),state)+a)
    U_2 = np.inner(np.transpose(W2),U_1)+b
    Psi_n = np.exp(U_2[0]+U_2[1]*1j)
    return(Psi_n)

#*****************************************

def E_loc(state):
    P_n = Wave_function(state,W1,W2,a,b)
    i = P_state.index(list(state))
    if (i == 0) or (i == 2):
        E = U - 2*t*Wave_function(P_state[1],W1,W2,a,b)/P_n
    else:
        E = -t*(Wave_function(P_state[0],W1,W2,a,b)+ Wave_function(P_state[2],W1,W2,a,b))/P_n
    return(E)

#********************************************

def step(state):
    rsn = np.random.randint(0,3)
    Bstate = P_state[rsn]
    E_loc_before = E_loc(Bstate)
    Wf_Bfr = Wave_function(Bstate,W1,W2,a,b)
    if rsn == 2:
        rsn = 1
    else:
        rsn += 1
    state = P_state[rsn]
    E_loc_after = E_loc(state)
    Wf_Aftr = Wave_function(state,W1,W2,a,b)
    acceptance = np.real(Wf_Aftr*np.conj(Wf_Aftr)/(Wf_Bfr*np.conj(Wf_Bfr)))
    if acceptance < np.random.uniform() and E_loc_after >= E_loc_before:
        return(Bstate)     #Bstate
    else:
        return(state)     #state

#************************************************

block_Ef = []
G_E = []
T =[]
for a in range(18,42):
    U = a/3
    block_E = []
    for block_i in range(10):
        rsn = np.random.randint(0,3)
        Bstate = P_state[rsn]
        for k in range(10000):
            rsn = np.random.randint(0,3)
            Bstate = P_state[rsn]
            step(state)
        iterations = 20000
        rejected = 0
        array_E_loc = []
    
        array_a_d = []
        array_w1_d = []
        array_w2_d = []
        for k in range(iterations):
       # state = step()
            if k % 100 == 0:
                state = step(state)
                Psi_M_s = Wave_function(state,W1,W2,a,b)
            # Derivative b = 0
            # Derivative a 
                a_deriv = np.zeros(N, dtype=np.complex_)
                for i in range(N):
                    if rsn == 2:
                        rsn = 1
                    else:
                        rsn += 1
                    a_deriv[i] = -t*Wave_function(state,W1,W2,a,b)*((np.tanh(Angle(P_state[rsn])[i]))**2 - (np.tanh(Angle(state)[i]))**2)*(W2[0][i]+W2[1][i]*1j)/Psi_M_s
            # Derivative W2
                dW2 = np.zeros((H,O),dtype=np.complex_)
                for w_i in range(H):
                    for w_j in range(O):
                    #dw_sum = 0
                    #before_flip = np.tanh(effective_angles(state))
                        dW2[w_i,w_j] = -t*Wave_function(state,W1,W2,a,b)*((1j)**(w_i))*(np.tanh(Angle(state)[w_j])- np.tanh(Angle(P_state[rsn])[w_j]))/Psi_M_s
           # Derivative W1
                dW1 = np.zeros((N,H),dtype=np.complex_)
                for w_i in range(N):
                    for w_j in range(H):
                        dW1[w_i,w_j] = t*Wave_function(state,W1,W2,a,b)*(((1-(np.tanh(Angle(P_state[rsn])[w_j]))**2)*(P_state[rsn])[w_j]) - (1-(np.tanh(Angle(state)[w_j]))**2)*state[w_j])*(W2[0][w_i]+W2[1][w_i]*1j)/Psi_M_s

                array_a_d.append(a_deriv)
                array_w1_d.append(dW1)
                array_w2_d.append(dW2)
                array_E_loc.append(np.real(E_loc(state)))

    #print('%d. E_loc=%.4f std=%.4f (%.1f %% moves rejected)' % (block_i+1,
        #np.mean(array_E_loc),np.std(array_E_loc)/(np.sqrt(len(array_E_loc))), 100.*rejected/iterations))
    #print(state,np.mean(array_E_loc))
        block_E.append(np.mean(array_E_loc))
        mean_da = np.mean(np.array(array_a_d),axis=0)
        mean_dw1 = np.mean(np.array(array_w1_d),axis=0)
        mean_dw2 = np.mean(np.array(array_w2_d),axis=0)
    #print(mean_da,mean_dw1,mean_dw2)
        a = a - .05 * mean_da
        W1 = W1 - .05 * mean_dw1
        W2 = W2 - .05 * mean_dw2
    block_Ef.append(np.mean(block_E)+1)
    G_E.append(((U-(U**2+16*t**2)**.5)/2))
    T.append(U)
    print(np.mean(block_E),(U-(U**2+16*t**2)**.5)/2,block_E[9])

#**********************************************************
#G_E = np.ones(30)
#G_E = G_E*((U-(U**2+16*t**2)**.5)/2)
#*********************************************************

import matplotlib.pyplot as plt
plt.plot(T,block_Ef,'.',label = 'predicted_GE')
#plt.legend()
plt.title(r'Local energy $<E_{loc}>$')
plt.ylabel(r'$<E_{loc}>$')
plt.xlabel('U')
plt.plot(T,G_E,label = 'Exact_GE')
plt.legend()
plt.show()

#*******************************
