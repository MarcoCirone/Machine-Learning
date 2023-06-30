import numpy
import scipy
import sys
import matplotlib.pyplot as plt
sys.path.append("../../LAB2")
sys.path.append("../../LAB4")
sys.path.append("../../LAB5")
from lab2 import mcol, mrow
from lab4 import logpdf_GAU_ND
from lab5 import (MVG_likelihood_domain, MVG_log_likelihood_domain, 
                  TVG_likelihood_domain, TVG_log_likelihood_domain,
                  Naive_Bayes_likelihood_domain, Naive_Bayes_log_likelihood_domain,
                  Tied_Naive_Bayes_likelihood_domain, Tied_Naive_Bayes_log_likelihood_domain, #possibilmente questi
                  compute_predicted_labels, compute_predicted_labels_log,
                  KFold, leave_one_out)  
def load(file):
    D = []
    L = []
    with open(file, "r") as f:
        for line in f:
            attributes = line.split(",")[:-1:1]
            properties = mcol(numpy.array([float (i) for i in attributes]))
            D.append(properties)
            label = line.split(",")[-1].replace("\n", "")
            L.append(label)
    return numpy.hstack(D), numpy.array(L, dtype = numpy.int32)

def scatter2D(D, L):
    for i in range(D.shape[0]):
        for j in range(D.shape[0]):
            if i != j:
                plt.figure()
                for c in range(L.max()+1):
                    D_c = D[:, L==c]
                    x1 = D_c[i, :]
                    x2 = D_c[j, :]
                    plt.scatter(x1, x2, label = c)
                plt.xlabel(f'Dimension {i}')
                plt.xlabel(f'Dimension {j}')
                plt.legend()
                plt.show()
                
def scatter3D(D, L):
    for i in range(D.shape[0]):
        for j in range(D.shape[0]):
            for k in range(D.shape[0]):
                if i != j and i != k and  j != k:
                    fig = plt.figure()
                    ax = fig.add_subplot(projection = '3d')
                    for c in range(L.max()+1):
                        D_c = D[:, L==c]
                        x = D_c[i, :]
                        y = D_c[j, :]
                        z = D_c[k, :]
                        ax.scatter(x, y, z, label = c)
                        ax.set_xlabel('X Label')
                        ax.set_ylabel('Y Label')
                        ax.set_zlabel('Z Label')
                    plt.legend()
                    plt.show()
                    
def hist(D, L):
    for i in range(D.shape[0]):
        plt.figure()
        for c in range(L.max()+1):
            x = D[i, L==c]
            plt.hist(x.reshape(x.size,), alpha = 0.4, label = c)
        plt.xlabel(f'Dimension {i+1}')
        plt.legend()
        plt.show()
                    
def PCA(D, n):
    mu = mcol(D.mean(axis = 1))
    DC = D - mu
    C = numpy.dot(DC, DC.T)/D.shape[1]
    _, U = numpy.linalg.eigh(C)
    return U[:, ::-1][:, 0:n]

def LDA(D, L, n):
    mu = mcol(D.mean(axis = 1))
    SB = numpy.zeros((D.shape[0], D.shape[0]))
    SW = numpy.zeros((D.shape[0], D.shape[0]))
    for c in range(L.max()+1):
        D_c = D[:, L==c]
        mu_c = mcol(D_c.mean(axis = 1))
        SB += D_c.shape[1] * numpy.dot((mu_c - mu), (mu_c - mu).T)
        SW += numpy.dot((D_c - mu_c), (D_c - mu_c).T)
    SB /= D.shape[1]
    SW /= D.shape[1]
    _,U=scipy.linalg.eigh(SB, SW)
    W=U[:, ::-1][:, 0:n]
    return W

def kfold_cross_validation(DTR, LTR, Prior, file_name, model):
    file = open(file_name, "w")
    
    content = ""
    
    for i in range(DTR.shape[0], 0, -1):
        if i != DTR.shape[0]:
            P1 = PCA(DTR, i)
            nDTR = numpy.dot(P1.T, DTR)
        else:
            nDTR = DTR
        for j in range(i, 0, -1):
            if j != i:
                P2 = LDA(nDTR, LTR, j)
                nDTR = numpy.dot(P2.T, nDTR)
            for k in range(2, 6, 1):
                print(f"i={i} j={j} k={k}\n")
                logS = KFold(nDTR, LTR, k, model)
                PL = compute_predicted_labels_log(logS, Prior)
                CL = PL ==LTR
                acc = CL.sum(0)/CL.shape[0]
                err_rate = (1-acc)*100
                content += f"Dim_after_PCA:{i},Dim_after_LDA:{j},K={k},Error_Rate:{err_rate}%\n"
    
    file.write(content)
    file.close()
    
def LeaveOneOut(DTR, LTR, Prior, file_name, model):
    file = open(file_name, "w")
    
    content = ""
    
    for i in range(DTR.shape[0], 0, -1):
        if i != DTR.shape[0]:
            P1 = PCA(DTR, i)
            nDTR = numpy.dot(P1.T, DTR)
        else:
            nDTR = DTR
        for j in range(i, 0, -1):
            if j != i:
                P2 = LDA(nDTR, LTR, j)
                nDTR = numpy.dot(P2.T, nDTR)
            print(f"i={i} j={j}\n")
            logS = leave_one_out(nDTR, LTR, model)
            PL = compute_predicted_labels_log(logS, Prior)
            CL = PL ==LTR
            acc = CL.sum(0)/CL.shape[0]
            err_rate = (1-acc)*100
            content += f"Dim_after_PCA:{i},Dim_after_LDA:{j},Error_Rate:{err_rate}%\n"
            
    file.write(content)
    file.close()

def compute_results(DTR, LTR, DTE, LTE, Prior, file_name, model):
    file = open(file_name, "w")
    
    content = ""
    
    for i in range(DTR.shape[0], 0, -1):
        if i != DTR.shape[0]:
            P1 = PCA(DTR, i)
            nDTR = numpy.dot(P1.T, DTR)
            nDTE = numpy.dot(P1.T, DTE)
        else:
            nDTR = DTR
            nDTE = DTE
        for j in range(i, 0, -1):
            print(f"i={i} j={j}\n")
            if j != i:
                P2 = LDA(nDTR, LTR, j)
                nDTR = numpy.dot(P2.T, nDTR)
                nDTE = numpy.dot(P2.T, nDTE)
            logS = model(nDTR, LTR, nDTE)
            PL = compute_predicted_labels_log(logS, Prior)
            
            CL = PL == LTE
            acc = CL.sum(0)/CL.shape[0]
            err_rate = (1-acc)*100
            content += f"Dim_after_PCA:{i},Dim_after_LDA:{j},Error_Rate:{err_rate}%\n"
            
    file.write(content)
    file.close()

if __name__ == '__main__':
    DTR, LTR = load("Train.txt")
    DTE, LTE = load("Test.txt")
    
    Prior = numpy.ones((2, 1))*0.5
    
    #compute_results(DTR, LTR, DTE, LTE, Prior, "./accuracies/MVG_Results.txt", MVG_log_likelihood_domain)
    #compute_results(DTR, LTR, DTE, LTE, Prior, "./accuracies/TVG_Results.txt", TVG_log_likelihood_domain)
    #compute_results(DTR, LTR, DTE, LTE, Prior, "./accuracies/Naive_Bayes_Results.txt", Naive_Bayes_log_likelihood_domain)
    #compute_results(DTR, LTR, DTE, LTE, Prior, "./accuracies/Tied_Naive_Bayes_Results.txt", Tied_Naive_Bayes_log_likelihood_domain)
    
    #kfold_cross_validation(DTR, LTR, Prior, "./kfold_cross_validation/MVG_Results.txt", MVG_log_likelihood_domain)
    #kfold_cross_validation(DTR, LTR, Prior, "./kfold_cross_validation/TVG_Results.txt", TVG_log_likelihood_domain)
    #kfold_cross_validation(DTR, LTR, Prior, "./kfold_cross_validation/Naive_Bayes_Results.txt", Naive_Bayes_log_likelihood_domain)
    #kfold_cross_validation(DTR, LTR, Prior, "./kfold_cross_validation/Tied_Naive_Bayes_Results.txt", Tied_Naive_Bayes_log_likelihood_domain)
    
    #LeaveOneOut(DTR, LTR, Prior, "./leave_one_out/MVG_Results.txt", MVG_log_likelihood_domain)  FATTO
    #LeaveOneOut(DTR, LTR, Prior, "./leave_one_out/TVG_Results.txt", TVG_log_likelihood_domain)  FATTO
    #LeaveOneOut(DTR, LTR, Prior, "./leave_one_out/Naive_Bayes_Results.txt", Naive_Bayes_log_likelihood_domain)
    #LeaveOneOut(DTR, LTR, Prior, "./leave_one_out/Tied_Naive_Bayes_Results.txt", Tied_Naive_Bayes_log_likelihood_domain)
    
    #hist(DTR, LTR)
    #scatter3D(DTR, LTR)
    #D, L = load("iris.csv")
    
    #P1 = PCA(DTR, 6)
    #y = numpy.dot(P1.T, DTR)
    #scatter3D(y, LTR)
    #hist(y, LTR)
    
    #P2 = LDA(y, LTR, 3)
    #y = numpy.dot(P2.T, y)
    
    #mu0 = mcol(y[:, LTR==0].mean(axis=1))
    #C0 = numpy.dot(y[:, LTR==0]-mu0, (y[:, LTR==0]-mu0).T)/y[:, LTR==0].shape[1]
        
    #mu1 = mcol(y[:, LTR==1].mean(axis=1))
    #C1 = numpy.dot(y[:, LTR==1]-mu1, (y[:, LTR==1]-mu1).T)/y[:, LTR==1].shape[1]
    
    #print(numpy.abs(C0-C1).max())
    
    #hist(y, LTR)
    #scatter3D(y, LTR)