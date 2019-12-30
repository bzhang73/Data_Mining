import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import math
import cv2

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.fftpack import fft, dct

#draw list to hist
def draw_hist(myList,Title,Xlabel,Ylabel,Xmin,Xmax,Ymin,Ymax):
    
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

#PCA
#def pca(data, n_dim):
#    data = data - np.mean(data, axis = 0, keepdims = True)
#    cov = np.dot(data.T, data)
#    eig_values, eig_vector = np.linalg.eig(cov)
#    indexs_ = np.argsort(-eig_values)[:n_dim]
#    picked_eig_values = eig_values[indexs_]
#    picked_eig_vector = eig_vector[:, indexs_]
#    data_ndim = np.dot(data, picked_eig_vector)
#    return data_ndim

temp1=np.loadtxt('dist1_500_1.txt')
temp2=np.loadtxt('dist1_500_2.txt')

data1 =np.concatenate((temp1,temp2),axis=0)
#print(len(data1))

temp1=np.loadtxt('dist2_500_1.txt')
temp2=np.loadtxt('dist2_500_2.txt')

data2 =np.concatenate((temp1,temp2),axis=0)
#print(len(data1))



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


x_std=StandardScaler().fit_transform(data1)

#print(len(x_std))
#exit()
print("PCA data1 original:")

pca=PCA()
principalComponednts1=pca.fit_transform(x_std)

print(pca.explained_variance_ratio_)
print(len(pca.singular_values_))

draw_hist_pca(pca.explained_variance_ratio_,'My pca_data1_original variance ratio','X','Y',0,101,0,100);


x_std2=StandardScaler().fit_transform(data2)
#print(x_std)

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


print("my data1 original dct:\n")
#DCT
dct_data1=data1
dct_data2=data2

#dct_std1 =StandardScaler().fit_transform(dct_data1)
#dct_std2 =StandardScaler().fit_transform(dct_data2)

for i in range(len(dct_data1)):
    temp=dct(dct_data1[i],1)
    temp=temp.tolist()
    total=0.0
    for ele in range(0, len(temp)):
        total = total + abs(temp[ele])
#    print(total)
    for ele in range(0, len(temp)):
        dct_data1[i][ele]=(abs(temp[ele])/total)*100
#        print(dct_data1[i][ele])
#    exit()
#    print(dct_data1[i][0])

final1=data1[0]
for i in range(len(dct_data1[i])):
    total=0.0
    for j in range(0,len(dct_data1)):
        total=total+dct_data1[j][i]
    final1[i]=total/len(dct_data1)
#    x.append(i)
#print(final1)
#draw_hist(df1,i,'index %d'%i,'number',0,340,0,10)
draw_data1=final1
for i in range(1,len(draw_data1)):
    draw_data1[i]=draw_data1[i]+draw_data1[i-1]
draw_hist(draw_data1,'My roiginal data1 dct variance ratio','X','Y',0,101,0,100);

print("my data2 original dct:\n")

for i in range(len(dct_data2)):
    temp=dct(dct_data2[i],1)
    temp=temp.tolist()
    total=0.0
    for ele in range(0, len(temp)):
        total = total + abs(temp[ele])
    for ele in range(0, len(temp)):
        dct_data2[i][ele]=(abs(temp[ele])/total)*100

final2=data2[0]
for i in range(len(dct_data2[i])):
    total=0.0
    for j in range(0,len(dct_data2)):
        total=total+dct_data2[j][i]
    final2[i]=total/len(dct_data1)
#    x.append(i)
#print(final1)
#draw_hist(df1,i,'index %d'%i,'number',0,340,0,10)
draw_data2=final2
for i in range(1,len(draw_data2)):
    draw_data2[i]=draw_data2[i]+draw_data2[i-1]
draw_hist(draw_data2,'My roiginal data2 dct variance ratio','X','Y',0,101,0,100);

print("my data1 reduce 70% dct:\n")
draw_hist_penc(draw_data1,'My roiginal data1 dct variance ratio','X','Y',0,101,0,100,70);
print("my data2 reduce 70% dct:\n")
draw_hist_penc(draw_data2,'My roiginal data2 dct variance ratio','X','Y',0,101,0,100,70);



#print(my_patient_data_X[0][0])
#exit()
#
#for i in range(len(dct_std1)):
#    temp=dct(dct_std1[i],1)
#    print(temp)
#
#    exit()





#for i in range(len(pca))
#print(pca.singular_values_)



#plt.figure(figsize=(8,4))
#plt.subplot(121)
#plt.title("my_PCA")
#for i in range(len(data_pca)):
#    for j in range(len(data_pca[i])):
#        plt.scatter(i, data_pca[i][j],c='b',  alpha=0.5)
#plt.show()
#
#exit((0))
#B=2 #blocksize
#dct_data=X
#mat=np.mat(dct_data)
#h,w=np.array(mat.shape[:2])/B * B
##print(list(mat[0]))
#test=list(mat[0])
#print(test[0][0][0][0])
#exit()
#print(h)
#print(w)
#mat=mat[:h,:w]

#blocksV=h/B
#blocksH=w/B
#vis0 = mat
#Trans = mat
#for row in range((int)(blocksV)):
#    for col in range((int)(blocksH)):
#        currentblock = cv2.dct(vis0[row*B:(row+1)*B,col*B:(col+1)*B])
#        Trans[row*B:(row+1)*B,col*B:(col+1)*B]=currentblock

#print(Trans[0][0])
#print(list(Trans))
#res=list(Trans)
#print(res)

#pd.plotting.scatter_matrix(Trans,)
#plt.show()
#
#exit(0)
#xplo=[0 for x in range(100)]
#plt.subplot(122)
#plt.title("my_DCT")
#for i in range(len(Trans)):
#    for j in range(len(Trans[i])):
##        print(i)
#        print(Trans[i][0])
#        print(len(Trans))
#        print(len(Trans[0][0]))
#        plt.scatter(xplo, Trans[i],c='b',  alpha=0.5)
#
#plt.show()
