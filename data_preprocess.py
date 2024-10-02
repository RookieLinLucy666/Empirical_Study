# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 09:39:07 2020

@author: user
"""
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from torch.utils.data import Dataset,TensorDataset, Subset
from torchvision import datasets, transforms
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import torchvision

torch.manual_seed(42)

"""Function: load data"""
def data_init(FL_params):

    kwargs = {'num_workers': 0, 'pin_memory': True} if FL_params.cuda_state else {}
    trainset, testset = data_set(FL_params.data_name)
    # shadow_split_idx = [int(whole_trainset.__len__()/2), int(whole_trainset.__len__()) -int(whole_trainset.__len__()/2)]
    # trainset, shadow_trainset = torch.utils.data.random_split(whole_trainset, shadow_split_idx)

    # shadow_split_idx = [int(whole_testset.__len__()/2), int(whole_testset.__len__()) -int(whole_testset.__len__()/2)]
    # testset, shadow_testset = torch.utils.data.random_split(whole_testset, shadow_split_idx)
    #

    # train_loader = DataLoader(trainset, batch_size=FL_params.local_batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(testset, batch_size=FL_params.test_batch_size, shuffle=True, **kwargs)
    # shadow_test_loader = DataLoader(shadow_testset, batch_size=FL_params.test_batch_size, shuffle=False, **kwargs)


    split_index = [int(trainset.__len__()/FL_params.N_total_client)]*(FL_params.N_total_client-1)
    split_index.append(int(trainset.__len__() - int(trainset.__len__()/FL_params.N_total_client)*(FL_params.N_total_client-1)))
    client_dataset = torch.utils.data.random_split(trainset, split_index)

    # split_index = [int(shadow_trainset.__len__()/FL_params.N_total_client)]*(FL_params.N_total_client-1)
    # split_index.append(int(shadow_trainset.__len__() - int(shadow_trainset.__len__()/FL_params.N_total_client)*(FL_params.N_total_client-1)))
    # shadow_client_dataset = torch.utils.data.random_split(shadow_trainset, split_index)
    client_loaders = []
    # shadow_client_sloaders = []
    for ii in range(FL_params.N_total_client):
        client_loaders.append(DataLoader(client_dataset[ii], FL_params.local_batch_size, shuffle=True, **kwargs))
        # shadow_client_loaders.append(DataLoader(shadow_client_dataset[ii], FL_params.local_batch_size, shuffle=False, **kwargs))

    return client_loaders, test_loader

def data_init_niid(FL_params):

    kwargs = {'num_workers': 0, 'pin_memory': True} if FL_params.cuda_state else {}
    trainset, testset = data_set(FL_params.data_name)

    # 构建测试数据加载器
    test_loader = DataLoader(testset, batch_size=FL_params.test_batch_size, shuffle=True, **kwargs)

    num_devices = FL_params.N_total_client
    if FL_params.data_name == "cifar100":
        num_classes = 100
    else:
        num_classes = 10
    samples_per_device = int(trainset.__len__() / num_devices)
    dominant_class_ratio = 0.8

    client_datasets = []

    for i in range(num_devices):
        dominant_class = i % num_classes

        dominant_indices = np.where(np.array(trainset.targets) == dominant_class)[0]
        dominant_indices = np.random.choice(dominant_indices, int(samples_per_device*dominant_class_ratio), replace=False)

        nondominant_indices = np.where(np.array(trainset.targets) != dominant_class)[0]
        nondominant_indices = np.random.choice(nondominant_indices, int(samples_per_device*(1-dominant_class_ratio)), replace=False)

        indices = np.concatenate([dominant_indices, nondominant_indices])
        np.random.shuffle(indices)

        subset = Subset(trainset, indices)
        client_datasets.append(subset)
    client_loaders = []
    for i in range(FL_params.N_total_client):
        client_loaders.append(DataLoader(client_datasets[i], FL_params.local_batch_size, shuffle=True, **kwargs))

    return client_loaders, test_loader

"""Function: load data"""
def data_init_with_shadow(FL_params):
    
    kwargs = {'num_workers': 0, 'pin_memory': True} if FL_params.cuda_state else {}
    whole_trainset, whole_testset = data_set(FL_params.data_name)
    shadow_split_idx = [int(whole_trainset.__len__()/2), int(whole_trainset.__len__()) -int(whole_trainset.__len__()/2)]
    trainset, shadow_trainset = torch.utils.data.random_split(whole_trainset, shadow_split_idx)
    
    shadow_split_idx = [int(whole_testset.__len__()/2), int(whole_testset.__len__()) -int(whole_testset.__len__()/2)]
    testset, shadow_testset = torch.utils.data.random_split(whole_testset, shadow_split_idx)
    
    
    # train_loader = DataLoader(trainset, batch_size=FL_params.local_batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(testset, batch_size=FL_params.test_batch_size, shuffle=False, **kwargs)
    shadow_test_loader = DataLoader(shadow_testset, batch_size=FL_params.test_batch_size, shuffle=False, **kwargs)

    split_index = [int(trainset.__len__()/FL_params.N_client)]*(FL_params.N_client-1)
    split_index.append(int(trainset.__len__() - int(trainset.__len__()/FL_params.N_client)*(FL_params.N_client-1)))
    client_dataset = torch.utils.data.random_split(trainset, split_index)
    
    split_index = [int(shadow_trainset.__len__()/FL_params.N_client)]*(FL_params.N_client-1)
    split_index.append(int(shadow_trainset.__len__() - int(shadow_trainset.__len__()/FL_params.N_client)*(FL_params.N_client-1)))
    shadow_client_dataset = torch.utils.data.random_split(shadow_trainset, split_index)

    client_loaders = []
    shadow_client_loaders = []
    for ii in range(FL_params.N_client):
        client_loaders.append(DataLoader(client_dataset[ii], FL_params.local_batch_size, shuffle=False, **kwargs))
        shadow_client_loaders.append(DataLoader(shadow_client_dataset[ii], FL_params.local_batch_size, shuffle=False, **kwargs))
    
    return client_loaders, test_loader, shadow_client_loaders, shadow_test_loader

def data_set(data_name):
    if not data_name in ['mnist', 'fmnist', 'purchase','adult','cifar10','cifar100']:
        raise TypeError('data_name should be a string, including mnist,purchase,adult,cifar10. ')
    
    #model: 2 conv. layers followed by 2 FC layers
    if(data_name == 'mnist'):
        trainset = datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))

        testset = datasets.MNIST('./data', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))

    elif(data_name == 'fmnist'):
        trainset = datasets.FashionMNIST('./data', train=True, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))
                                  ]))

        testset = datasets.FashionMNIST('./data', train=False, download=True,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.1307,), (0.3081,))
                                 ]))
        
    #model: ResNet-50
    elif(data_name == 'cifar10'):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        trainset = datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        
        testset = datasets.CIFAR10(root='./data', train=False,
                                                download=True, transform=transform)

    # model: ResNet-50 or other suitable models
    elif(data_name == 'cifar100'):
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),  # CIFAR-100 的标准化参数
        ])

        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

    #model: 2 FC layers
    elif(data_name == 'purchase'):
        xx = np.load("./data/purchase/purchase_xx.npy")
        yy = np.load("./data/purchase/purchase_y2.npy")
        # yy = yy.reshape(-1,1)
        # enc = preprocessing.OneHotEncoder(categories='auto')
        # enc.fit(yy)
        # yy = enc.transform(yy).toarray()
        X_train, X_test, y_train, y_test = train_test_split(xx, yy, test_size=0.2, random_state=42)
        
        X_train_tensor = torch.Tensor(X_train).type(torch.FloatTensor)
        X_test_tensor = torch.Tensor(X_test).type(torch.FloatTensor)
        y_train_tensor = torch.Tensor(y_train).type(torch.LongTensor)
        y_test_tensor = torch.Tensor(y_test).type(torch.LongTensor)
        
        trainset = TensorDataset(X_train_tensor,y_train_tensor)
        testset = TensorDataset(X_test_tensor,y_test_tensor)
        
    
    #model: 2 FC layers
    elif(data_name == 'adult'):
        #load data
        file_path = "./data/adult/"
        data1 = pd.read_csv(file_path + 'adult.data', header=None)
        data2 = pd.read_csv(file_path + 'adult.test', header=None)
        data2 = data2.replace(' <=50K.', ' <=50K')    
        data2 = data2.replace(' >50K.', ' >50K')
        train_num = data1.shape[0]
        data = pd.concat([data1,data2])
       
        #data transform: str->int
        data = np.array(data, dtype=str)
        labels = data[:,14]
        le= LabelEncoder()
        le.fit(labels)
        labels = le.transform(labels)
        data = data[:,:-1]
        
        categorical_features = [1,3,5,6,7,8,9,13]
        # categorical_names = {}
        for feature in categorical_features:
            le = LabelEncoder()
            le.fit(data[:, feature])
            data[:, feature] = le.transform(data[:, feature])
            # categorical_names[feature] = le.classes_
        data = data.astype(float)
        
        n_features = data.shape[1]
        numerical_features = list(set(range(n_features)).difference(set(categorical_features)))
        for feature in numerical_features:
            scaler = MinMaxScaler()
            sacled_data = scaler.fit_transform(data[:,feature].reshape(-1,1))
            data[:,feature] = sacled_data.reshape(-1)
        
        #OneHotLabel
        oh_encoder = ColumnTransformer(
            [('oh_enc', OneHotEncoder(sparse=False), categorical_features),], 
            remainder='passthrough' )
        oh_data = oh_encoder.fit_transform(data)
        
        xx = oh_data
        yy = labels

        xx = preprocessing.scale(xx)
        yy = np.array(yy)
        
        xx = torch.Tensor(xx).type(torch.FloatTensor)
        yy = torch.Tensor(yy).type(torch.LongTensor)
        xx_train = xx[0:data1.shape[0],:]
        xx_test = xx[data1.shape[0]:,:]
        yy_train = yy[0:data1.shape[0]]
        yy_test = yy[data1.shape[0]:]
        
        # trainset = Array2Dataset(xx_train, yy_train)
        # testset = Array2Dataset(xx_test, yy_test)
        trainset = TensorDataset(xx_train,yy_train)
        testset = TensorDataset(xx_test,yy_test)
        
    return trainset, testset


#define class->dataset  for adult and purchase datasets
#for the purchase, we use TensorDataset function to transform numpy.array to datasets class
#for the adult, we custom an AdultDataset class that inherits torch.util.data.Dataset class
"""
Array2Dataset: A class that can transform np.array(tensor matrix) to a torch.Dataset class.  
"""
class Array2Dataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform
    def __getitem__(self, index):
        x = self.data[index,:]
        y = self.targets[index]
        return x, y
    def __len__(self):
        return len(self.data)
    
    




