from numpy import *
from numpy.linalg import *

def update(X, P, Y, H, R):
    IM = dot(H, X)
    IS = R + dot(H, dot(P, H.T))
    K = dot(P, dot(H.T, inv(IS)))
    P = P - dot(K, dot(IS, K.T))
    LH = gauss(Y, IM, IS)

    return (X,P,K,IM,IS,LH)



def gauss(X, M, S):
    if M.shape()[1] == 1:
        DX = X - tile(M, X.shape()[1])
        E = 0.5* sum(DX * (dot(inv(S), DX)), axis = 0 )
    elif X.shape()[1] == 1:
        DX = tile(X, M.shape()[1]) - M
        E = 0.5* sum(DX * (dot(inv(S), DX)), axis = 0 )
    else:
        DX = X-M
        E = 0.5 * dot(DX.T, dot( inv(S), DX))

    E = E + 0.5* (M.shape()[0] * log(2*pi) + log(det(S)))
    P = exp(-E)

    return (P[0],E[0])


def predict(X,P,A,Q,B,U):
     X = dot(A,X) + dot(B,U)
     P = dot(A,dot(P,A.T)) + Q
     return (X,P)



##################################################################################################################################

dt = 0.1 # time step for movement

X = array([[0.0], [0.0], [0.1], [0.1]])
P = diag((0.01, 0.01, 0.01, 0.01))
A = array([[1,0,dt,0], [0,1,0,dt], [0,0,1,0], [0,0,0,1]])

Q = eye(X.shape()[0])
B = Q
U = zeros((X.shape()[0],1))

# Measurements matrices

Y = array([[X[0,0] + abs(randn(1)[0])], [X[1,0] +  abs(randn(1)[0])]])

H = array([[1, 0, 0, 0], [0, 1, 0, 0]]) 

R = eye(Y.shape()[0]) 

# No. of iterations in Kalman Filter
N_iter = 50 

#Applying the Kalman FIlter

for i in arange(0,N_iter):
    (X,P) = predict(X,P,A,Q,B,U)
    (X,P,K,IM,IS,LH) = update(X,P,Y,H,R)
    Y = array([[X[0,0] + abs(0.1 * randn(1)[0])],[X[1, 0] +  abs(0.1 * randn(1)[0])]])


########################################################################################################################################
'''

Predict step

X: The mean state estimate of the previous step (k−1).

F: State covariance of the previous step

A: Transition nXn matrix

Q: The process noise covariance matrix

B: The input effect matrix

U: The control input


Update step

K: Kalman Gain Matrix
IM : Mean of predictive distribution of Y
IS : The covariance or predictive mean of Y
LH: The   Predictive   probability   (likelihood)   of   measurement   which   is computed using the Python function gauss

'''