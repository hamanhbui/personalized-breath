import os
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
import random
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from scipy.spatial import distance
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE

from modeling.cnn_lstm import Acce_Gyro_CNN_LSTM, Audio_CNN_LSTM, Multimodality_CNN_LSTM
from modeling.tcn import Acce_Gyro_TCN, Audio_TCN, Multimodality_TCN
from training.dataloader import Acce_Gyro_Dataset, Audio_Dataset, Multimodality_Dataset
from evaluation.verification import get_embedded_set

def tsne_plot(embedded_features,labels):
    def unique(list1): 
        unique_list = [] 
        for x in list1: 
            if x not in unique_list: 
                unique_list.append(x) 
        return unique_list

    labels=np.asarray(labels)
    tsne_model = TSNE(n_components=2,init='pca')
    X_2d = tsne_model.fit_transform(embedded_features)
    target_names=unique(labels) 
    target_ids=range(len(target_names))
    plt.figure(figsize=(16, 16)) 
    male_colors=['red','green','blue','black','brown','grey','orange','yellow','pink','cyan','magenta']
    female_colors=['red','blue','green','black','grey','orange','yellow','purple']   
    females=[4-1,11-1,19-1,20-1]
    idm_color=0
    idfm_color=0
    for i, label in zip(target_ids, target_names):
        if label in females:
            plt.scatter(X_2d[labels == i, 0], X_2d[labels == i, 1],c=female_colors[idfm_color], label=label+1)
            idfm_color+=1
        else:
            if idm_color>=len(male_colors):
                plt.scatter(X_2d[labels == i, 0], X_2d[labels == i, 1],c=male_colors[idm_color-len(male_colors)],marker='1', label=label+1)
            else:
                plt.scatter(X_2d[labels == i, 0], X_2d[labels == i, 1],c=male_colors[idm_color],marker='2', label=label+1)
            idm_color+=1
    plt.legend(loc=2, fontsize = 'x-small')
    plt.show()    

def main():
    model = Multimodality_CNN_LSTM(no_outer=0)
    model.load_state_dict(torch.load("results/models/normal/normal_multi_cnn-lstm_0_checkpoint.pt", map_location="cpu"))
    model.eval()
    with open('results/models/normal/list_test_filename', 'rb') as filehandle:
        list_test_filename = pickle.load(filehandle)
    with torch.no_grad():
        test_set = Multimodality_Dataset(root_dir="",filenamelist=list_test_filename,old_new_name_map={})
        test_loader = DataLoader(test_set,batch_size=128, shuffle=True)
        X_test,Y_test=get_embedded_set(test_loader,model)
        Y_test,X_test=(list(t) for t in zip(*sorted(zip(Y_test, X_test))))
        tsne_plot(X_test,Y_test)

main()