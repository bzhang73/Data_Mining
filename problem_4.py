import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile

from scipy import stats
import math
import cv2

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.fftpack import fft, dct

#draw list to hist
def draw_hist(myList,Title,Xlabel,Ylabel,Xmin,Xmax,Ymin,Ymax):
    for i in range(1,len(myList)):
        myList[i]=myList[i-1]+myList[i]
#    plt.hist(myList,len(myList))
    for i in range(0,len(myList)):
        plt.plot(i+1,myList[i],marker='*');
    plt.xlabel(Xlabel)
    plt.xlim(Xmin,Xmax)
    plt.ylabel(Ylabel)
    plt.ylim(Ymin,Ymax)
    plt.title(Title)
    plt.show()

def draw_hist_penc(myList,Title,Xlabel,Ylabel,Xmin,Xmax,Ymin,Ymax,penc):
#    for i in range(1,len(myList)):
#        myList[i]=myList[i-1]+myList[i]
    #    plt.hist(myList,len(myList))
    for i in range(0,len(myList)):
        if myList[i]>penc:
            print("The nubmer of features: %d"%i)
            break
        plt.plot(i+1,myList[i],marker='*');
    plt.xlabel(Xlabel)
    plt.xlim(Xmin,Xmax)
    plt.ylabel(Ylabel)
    plt.ylim(Ymin,Ymax)
    plt.title(Title)
    plt.show()

def draw_hist_pca(myList,Title,Xlabel,Ylabel,Xmin,Xmax,Ymin,Ymax):
    myList[0]=myList[0]*100;
    for i in range(1,len(myList)):
        myList[i]=myList[i-1]+myList[i]*100
    #    plt.hist(myList,len(myList))
    for i in range(0,len(myList)):
        plt.plot(i+1,myList[i],marker='*');
    plt.xlabel(Xlabel)
    plt.xlim(Xmin,Xmax)
    plt.ylabel(Ylabel)
    plt.ylim(Ymin,Ymax)
    plt.title(Title)
    plt.show()




def g(x):
    return np.tanh(x)

def g_der(x):
    return 1 - g(x) * g(x)

def center(X):
    X = np.array(X)
    
    mean = X.mean(axis=0)
    
    return X- mean

def whitening(X):
    X=X.T
    cov = np.cov(X)
    d, E = np.linalg.eigh(cov)
    D = np.diag(d)
    #    print(D)
    #    exit()
    temp=np.linalg.inv(D)
    #    print(temp)
    D_inv = np.sqrt(temp)
    X_whiten = np.dot(E, np.dot(D_inv, np.dot(E.T, X)))
    #    print(X_whiten)
    #    exit()
    return X_whiten

def calculate_new_w(w, X):
    w_new = (X * g(np.dot(w.T, X))).mean(axis=1) - g_der(np.dot(w.T, X)).mean() * w
    w_new /= np.sqrt((w_new ** 2).sum())
    return w_new

def ica(X, iterations, tolerance=1e-5):
    X = center(X)
    
    X = whitening(X)
    #    print(X)
    #    exit()
    
    components_nr = X.shape[0]
    
    W = np.zeros((components_nr, components_nr), dtype=X.dtype)
    
    for i in range(components_nr):
        
        w = np.random.rand(components_nr)
        #        print(w)
        #        exit()
        
        for j in range(iterations):
            
            w_new = calculate_new_w(w, X)
            #            print(w_new)
            #            exit()
            if i >= 1:
                w_new -= np.dot(np.dot(w_new, W[:i].T), W[:i])
        
            distance = np.abs(np.abs((w * w_new).sum()) - 1)
            #                print(distance)
            #                exit()
            
            w = w_new
            
            if distance < tolerance:
                break
                    
            W[i, :] = w

    S = np.dot(W, X)

    return S


temp1=np.loadtxt('dist1_500_1.txt')
temp2=np.loadtxt('dist1_500_2.txt')

data1 =np.concatenate((temp1,temp2),axis=0)
#print(len(data1))

temp1=np.loadtxt('dist2_500_1.txt')
temp2=np.loadtxt('dist2_500_2.txt')

data2 =np.concatenate((temp1,temp2),axis=0)



x_std=StandardScaler().fit_transform(data1)

print("PCA data1 original:")

pca=PCA()
principalComponednts1=pca.fit_transform(x_std)

print(pca.explained_variance_ratio_)
print(len(pca.singular_values_))

draw_hist_pca(pca.explained_variance_ratio_,'My pca_data1_original variance ratio','X','Y',0,101,0,100);


x_std2=StandardScaler().fit_transform(data2)

print("PCA data2 original:")
pca2=PCA()
principalComponednts2=pca2.fit_transform(x_std2)

print(pca2.explained_variance_ratio_)
print(len(pca2.singular_values_))
draw_hist_pca(pca2.explained_variance_ratio_,'My pca_data2_original variance ratio','X','Y',0,101,0,100);

print("PCA data1 70%:")
pca3=PCA(.70)
principalComponednts3=pca3.fit_transform(x_std)

print(pca3.explained_variance_ratio_)
print(len(pca3.singular_values_))
draw_hist_pca(pca3.explained_variance_ratio_,'My pca_data1_reduce70% variance ratio','X','Y',0,101,0,100);

print("PCA data2 70%:")
pca4=PCA(.70)
principalComponednts4=pca4.fit_transform(x_std2)

print(pca4.explained_variance_ratio_)
print(len(pca4.singular_values_))
draw_hist_pca(pca4.explained_variance_ratio_,'My pca_data2_reduce70% variance ratio','X','Y',0,101,0,100);









print("My ICA data_set1: ")
ica_temp=data1

S1 = ica(ica_temp, iterations=100)
#print("pass")
#print(S)
#print(len(S))
#print(len(S[9]))

dct_res1=ica_temp[0]

for i in range(0,len(S1)):
    sum=0
    for j in range(0,len(S1[i])):
        sum=sum+abs(S1[i][j])
    dct_res1[i]=sum/len(S1[i])

total=0.0
for i in range(0,len(dct_res1)):
    total=total+dct_res1[i]

for i in range(0,len(dct_res1)):
    dct_res1[i]=dct_res1[i]/total
    dct_res1[i]=dct_res1[i]*100

draw_hist(dct_res1,'My ICA_data_set1 variance ratio','X','Y',0,101,0,100);


print("My ICA data_set2: ")
ica_temp2=data2

S2 = ica(ica_temp2, iterations=100)
#print("pass")
#print(S)
#print(len(S))
#print(len(S[9]))

dct_res2=ica_temp2[0]

for i in range(0,len(S2)):
    sum=0
    for j in range(0,len(S2[i])):
        sum=sum+abs(S2[i][j])
    dct_res2[i]=sum/len(S2[i])

total=0.0
for i in range(0,len(dct_res2)):
    total=total+dct_res2[i]

for i in range(0,len(dct_res2)):
    dct_res2[i]=dct_res2[i]/total
    dct_res2[i]=dct_res2[i]*100

draw_hist(dct_res2,'My ICA_data_set2 variance ratio','X','Y',0,101,0,100);


print("my ICA data_set1 reduce 70% dct:\n")
draw_hist_penc(dct_res1,'My roiginal data1 dct variance ratio','X','Y',0,101,0,100,70);
print("my ICA data_set2 reduce 70% dct:\n")
draw_hist_penc(dct_res2,'My roiginal data2 dct variance ratio','X','Y',0,101,0,100,70);




##read data
#temp1=pd.read_table('dist1_500_1.txt',header=None,delim_whitespace=True,index_col=0);
#temp2=pd.read_table('dist1_500_2.txt',header=None,delim_whitespace=True,index_col=0);
#
#data1=temp1.append(temp2,ignore_index=False)
##print(data1)
##exit()

#temp1=pd.read_table('dist2_500_1.txt',header=None,delim_whitespace=True,index_col=0);
#temp2=pd.read_table('dist2_500_2.txt',header=None,delim_whitespace=True,index_col=0);
#
#data2=temp1.append(temp2,ignore_index=False)
#print(data1.values.tolist())
#exit()

#data=load_iris()
#X=data1.loc[:].values.tolist()


