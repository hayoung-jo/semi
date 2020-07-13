import os
import pandas as pd 
import numpy as np
import gudhi
import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sklearn.metrics as metrics
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import train_test_split

def make_data_dictionary(basic_path='./'):
    file_list=os.listdir(basic_path)
    result_dict={'good':[],'bad':[]}
    for file_name in file_list:
        if 'Fault' in file_name:
            work_list=result_dict['bad']
        else:
            work_list=result_dict['good']
        data_frame=pd.read_csv(os.path.join(basic_path,file_name))
        data_frame=data_frame.fillna(0)
        for col_i in range(1,len(data_frame.columns)):
            work_list.append(list(data_frame.iloc[:,col_i]))
    return result_dict


def preprocessing_data_dict(data_dict):
    #####processing np.inf
    #'''
    for key in data_dict.keys():
        for data in data_dict[key]:
            for index in range(len(data)):
                if data[index][1][1]==np.inf:
                    data[index][1][1]=999999
    #'''
    #####extract cordinate
    result_dict={'good':[],'bad':[]}
    good_data=result_dict['good']
    for index, data in enumerate(data_dict['good']):
        #good_data.append([])
        for _, cord in data:
            #good_data[index].append(cord)
            good_data.append(np.array(cord))
    bad_data=result_dict['bad']
    for index, data in enumerate(data_dict['bad']):
        #bad_data.append([])
        for _, cord in data:
            #bad_data[index].append(cord)    
            bad_data.append(np.array(cord))
    return result_dict



### make data set dict from csv file
basic_path='./FDC_DATA/tcmp'
data_dict=make_data_dictionary(basic_path)

### extract time embedding resutl to json file
#set time embedding param
embedding_dimension=[2,3,5,7]
for dim in embedding_dimension:
    for delay in [1,2,3,5]:#range(1,dim):
        for skip in [1,2,3,5]:#range(1,dim,2):
#make time embedding object
            te=gudhi.point_cloud.TimeDelayEmbedding(dim=dim,delay=delay,skip=skip)
            te_dict={'good':[],'bad':[]}
            for key in data_dict.keys():
                for data in data_dict[key]:
#run time embedding
                    te_dict[key].append(te(data).tolist())
#save result
            num='.'.join([str(dim),str(delay),str(skip)])
            result_path=os.path.join(os.getcwd(),'result','embeding')
            with open(os.path.join(result_path,num+'.txt'),'w') as f:
                json.dump(te_dict,f)


### extract persistence homology from time ebedding result json file to persistence homology josn
#set complex dict
result_path=os.path.join(os.getcwd(),'result','embeding')
complex_dict={'alpha':gudhi.AlphaComplex,'rips':gudhi.RipsComplex}
complex_result_path=os.path.join(os.getcwd(),'result','complex')
embedding_file_list=[]
for embedding_file in os.listdir(result_path):
    embedding_file_list.append(os.path.join(result_path,embedding_file))
embedding_file_list.sort(key=os.path.getsize)

for embedding_file in embedding_file_list:
    path=os.path.join(result_path,embedding_file)
    with open(path,'r') as f:
        data=json.load(f)
    for complex_key in complex.keys():
        if not os.path.isdir(os.path.join(complex_result_path,complex_key)):
            os.mkdir(os.path.join(complex_result_path,complex_key))
        work_directory_path=os.path.join(complex_result_path,complex_key)
        result_dict={'good':[],'bad':[]}
        for key in data.keys():
            for dt in data[key]:
                complex_=complex_dict[complex_key](dt)
                tree=complex_.create_simplex_tree()
                persistence=tree.persistence()
                result_dict[key].append(persistence)
        with open(os.path.joint(work_direcotry,'_'.join([complex_key,embedding_file])),'r') as f:
            json.dump(result_dict,f)



def merge_data_set(good_data,bad_data,good_label,bad_label):
    train=[]
    for good in good_data:
        train.append(good)
    for bad in bad_data:
        train.append(bad)
    train_label=good_label+bad_label    
    return train, train_label


### split test and train data set
# load persistence data
basic_path=os.path.join(os.getcwd(),'result','complex')
complex_='alpha'
complex_path=os.path.join(basic_path,complex_)
file_list=os.listdir(complex_path)
file_name=os.path.join(complex_path,file_list[1])
with open(file_name,'r') as f:
    data=json.load(f)
data=preprocessing_data_dict(data)    
#split train and test data set
good_train,good_test,good_label_train,good_label_test=train_test_split(data['good'],[0 for _ in range(len(data['good']))],test_size=0.5,shuffle=True)
bad_train,bad_test,bad_label_train,bad_label_test=train_test_split(data['bad'],[1 for _ in range(len(data['bad']))],test_size=0.5,shuffle=True)
train_data,train_label=merge_data_set(good_train,bad_train,good_label_train,bad_label_train)
test_data,test_label=merge_data_set(good_test,bad_test,good_label_test,bad_label_test)



### train classifier
classifier=KNN(n_neighbors=5)
classifier.fit(train_data,train_label)

### test classifier
result=metrics.classification_report(test_label,classifier.predict(test_data))


print(result)

