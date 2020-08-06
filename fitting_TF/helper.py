import numpy as np
import numpy.random as rand

rand.seed(42)
rand_data = rand.normal(10,2,20000)
print(np.mean(rand_data))
print(np.mean(np.concatenate((rand_data,[100]))))
data = np.array([np.concatenate((rand_data,[1000000])),np.concatenate((rand_data,[10000])),np.concatenate((rand_data,[100]))])
def remove_outlier(datas,p=3):
    Q1,Q3 = np.quantile(datas,q=[0.25,0.75],axis=1)
    IQR = Q3-Q1
    min_data,max_data=Q1-p*IQR,Q3+p*IQR
    result = np.empty(datas.shape)
    result[:]=np.NAN
    for i,data in enumerate(datas):
        data_pre =  data[np.logical_and(min_data[i]<data,data<max_data[i])]
        result[i,:data_pre.shape[0]] = data_pre
    return result

print(np.mean(data,axis=1))
print(np.nanmean(remove_outlier(data),axis=1))
print(np.nanmin(remove_outlier(data),axis=1))
print(np.min(data,axis=1))
