import numpy as np

#remove 0 valuses and order them with their vactors
def clean_eigs(w, v, threshold = 0):
    start = 0
    end = len(w)-1
    middle = (end + start)//2
    slice = 0

    while start < end:
        if w[middle] > threshold:
            slice = middle
            end = middle -1
        else:
            start = middle + 1
        middle = (end + start)//2
    if w[middle]>0:
        slice = middle
    return w[-1:slice-1:-1], v[:,-1:slice-1:-1]


def our_svd(A, threshold=0):

    #calculate the Gremian matrix with the smallest dimensions
    dimension = np.shape(A)
    X = np.empty(1)
    if dimension[0] <= dimension[1]:
        X = np.dot(A, np.transpose(A))
    else:
        X = np.dot(np.transpose(A), A)

    #Find eignevalues and vectors of Gramian matrix 
    w,V = np.linalg.eig(X)
    w, V = clean_eigs(w, V, threshold)

    Sigma = np.diag(np.sqrt(w))
    U = np.dot(np.dot(A, V), np.linalg.inv(Sigma))
    VT = np.transpose(V)

    return U, Sigma, VT


def our_DMD(Y, X):
    # SVD of X
    U, S, VT = our_svd(X)
    S = np.diag(S)


    #Defie the low-rank matrix A-tilde
    UT = np.transpose(U)
    S_1 = np.linalg.inv(S)
    V = np.transpose(VT)

    A_tilde = np.dot(np.dot(UT, Y),np.dot(V, S_1))

    #Calculate eigens of A-tilde

    W, V = np.linalg.eig(A_tilde)

    W = np.diag(W)

    #Calculate the modes
    modes = np.dot(np.dot(Y, V),np.dot(S_1, V))

