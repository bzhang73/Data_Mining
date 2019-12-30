import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

#draw list to hist
def draw_hist(myList,Title,Xlabel,Ylabel,Xmin,Xmax,Ymin,Ymax):
    plt.hist(myList,100)
    plt.xlabel(Xlabel)
    plt.xlim(Xmin,Xmax)
    plt.ylabel(Ylabel)
    plt.ylim(Ymin,Ymax)
    plt.title(Title)
    plt.show()

#read data
temp1=pd.read_table('dist1_500_1.txt',header=None,delim_whitespace=True,index_col=0);
temp2=pd.read_table('dist1_500_2.txt',header=None,delim_whitespace=True,index_col=0);

data1=temp1.append(temp2,ignore_index=False)
#print(data1)
#exit()

temp1=pd.read_table('dist2_500_1.txt',header=None,delim_whitespace=True,index_col=0);
temp2=pd.read_table('dist2_500_2.txt',header=None,delim_whitespace=True,index_col=0);

data2=temp1.append(temp2,ignore_index=False)
#print(data1)
#exit()


my_list1_mean=[]
my_list1_std=[]
#my_list1_D=[]
my_list1_nor_pvalue=[]
my_list1_uniform_pvalue=[]

#show hist while 10
i=0
while i<10 :
    df1=data1.sample();
    #mean
    set=df1.mean()
    u=set.mean()
    my_list1_mean.append(u)
    
    #std
    std=set.std()
    my_list1_std.append(std)
#    print(std)
#    print(u)
#    print(df1)
#    print("\n")
#    exit()
#    draw_hist(df1,i,'index %d'%i,'number',0,340,0,10)
#    exit()

    D,P_value=stats.kstest(set,'norm',(u,std))
#    my_list1_D.append(D)
    my_list1_nor_pvalue.append(P_value)
    
#    D,P_value=stats.kstest(set,'uniform',(u,std))
    #    my_list1_D.append(D)
    unique_set=[]
    for x in set:
        if x not in unique_set:
            unique_set.append(x)
    uniform_value=1/len(unique_set)
    my_list1_uniform_pvalue.append(uniform_value)
    i+=1

draw_hist(data1.sample(),i,'index %d'%i,'number',0,340,0,10)



my_list2_mean=[]
my_list2_std=[]
#my_list2_D=[]
my_list2_nor_pvalue=[]
my_list2_uniform_pvalue=[]

#show hist while 10
i=0
while i<10 :
    df2=data2.sample();
    #mean
    set=df2[0:].mean()
    u=set.mean()
    my_list2_mean.append(u)
    
    #std
    std=set.std()
    my_list2_std.append(std)

    D,P_value=stats.kstest(set,'norm',(u,std))
#    my_list2_D.append(D)
    my_list2_nor_pvalue.append(P_value)
    
#    D,P_value=stats.kstest(set,'uniform',(u,std))
    #    my_list1_D.append(D)
    unique_set=[]
    for x in set:
        if x not in unique_set:
            unique_set.append(x)
    uniform_value=1/len(unique_set)
    my_list2_uniform_pvalue.append(uniform_value)
    i+=1

draw_hist(data2.sample(),i,'index %d'%i,'number',0,340,0,10)

#print("Gaussian distributions")
i=0
sum1=0.0;
sum2=0.0;
while i<10:
    print("The %d set: "%i)
    print(" Mean Standard Deviation  kstest P-value   Each-element-probablity")
    mean1=my_list1_mean[i]
    print("%4d     "%mean1,end=' ')
    std1=my_list1_std[i]
    print("%7d       "%std1,end=' ')
#    D1=my_list1_D[i]
#    print("%f "%D1,end=' ')
    P1=my_list1_nor_pvalue[i]
    print("%f                "%P1,end=' ')
    PU1=my_list1_uniform_pvalue[i]
    print("%f "%PU1)
    
    mean2=my_list2_mean[i]
    print("%4d     "%mean2,end=' ')
    std2=my_list2_std[i]
    print("%7d       "%std2,end=' ')
#    D2=my_list2_D[i]
#    print("%f "%D2,end=' ')
    P2=my_list2_nor_pvalue[i]
    print("%f                "%P2,end=' ')
    PU2=my_list2_uniform_pvalue[i]
    print("%f        "%PU2,end=' ')
    
#    print("%f          "%(P1-P2),end=' ')
    sum1+=abs(float(P1))
#    print("%f    "%(PU1-PU2))
    sum2+=abs(float(P2))
    print()
    
    i+=1

res1=(float(sum1)/10)
res2=(float(sum2)/10)
print("The Data1 average kstest of Gaussian : %f"%res1)
print("The Data2 average kstest of Gaussian : %f"%res2)

print("The Data1 set is",end=' ')
if abs(res1)>=0.5:
    print("Gaussian distribution")
else:
    print("uniform distribution")

print("The Data2 set is",end=' ')
if abs(res2)>=0.5:
    print("Gaussian distribution")
else:
    print("uniform distribution")

