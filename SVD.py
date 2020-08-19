import numpy as np
from scipy.linalg import svd
from sklearn.decomposition import TruncatedSVD

def oldschoolstepbystep():
    A = np.array([[1,2,3,4,5,6,7,8,9,10],
        [11,12,13,14,15,16,17,18,19,20],
        [21,22,23,24,25,26,27,28,29,30]])
    print(A)
    U, s, VT = svd(A)
    print(U)
    print(s)
    print(VT)

    Sigma = np.zeros(A.shape)
    Sigma[:A.shape[0], :A.shape[0]] = np.diag(s)
    B = U.dot(Sigma.dot(VT))
    print(Sigma)
    print(B)

    n_elements = 2

    Sigma = Sigma[:, :n_elements]
    VT = VT[:n_elements, :]
    B = U.dot(Sigma.dot(VT))

    print(14*'*')
    print(U)
    print(Sigma)
    print(VT)
    print(B)

    T = U.dot(Sigma)
    print(T)
    T = A.dot(VT.T)
    print(T)

def usingsklearndecomposition():
    A = np.array([[1,2,3,4,5,6,7,8,9,10],
        [11,12,13,14,15,16,17,18,19,20],
        [21,22,23,24,25,26,27,28,29,30]])
    print(A)
    svd = TruncatedSVD(n_components=2)
    svd.fit(A)
    result = svd.transform(A)
    print(result)

def main():
    print("1) SVD Transform oldschool way\n2) SVD using sklearn")
    c = int(input())
    if c == 1:
        oldschoolstepbystep()
    else:
        usingsklearndecomposition()

if __name__=="__main__":
    main()
