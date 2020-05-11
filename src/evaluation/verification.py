import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from scipy.spatial import distance
from sklearn.mixture import GaussianMixture
import torch

def get_embedded_set(loader,model):
    X_out, Y_out=[],[]
    for batch_idx, (types, samples,labels) in enumerate(loader):
        if types[0] == "Multimodality":
            feature_1 = samples[0].float()
            feature_2 = samples[1].float()
            if torch.cuda.is_available():
                feature_1 = feature_1.cuda()
                feature_2 = feature_2.cuda()
                labels = labels.cuda()
            embedded_vec = model(feature_1, feature_2)
        else:
            features=samples.float()
            if torch.cuda.is_available():
                features= features.cuda()
                labels=labels.cuda()
            embedded_vec=model(features)

        X_out+=embedded_vec.tolist()
        Y_out+=labels.tolist()
    return X_out,Y_out
    
def get_centroid_embed_vec(X_train,Y_train,no_outer):
    subject_list_idx={}
    for idx_label in range(len(Y_train)):
        label=Y_train[idx_label]
        if label not in subject_list_idx:
            subject_list_idx[label]=[]
        subject_list_idx[label].append(idx_label)
    X_out, Y_out=[],[]
    for subject, list_idx in subject_list_idx.items():
        if subject not in Y_out:
            Y_out.append(subject)
        avg_embed_vec=np.zeros(20-no_outer)
        for idx in list_idx:
            avg_embed_vec+=X_train[idx]
        avg_embed_vec=avg_embed_vec/len(list_idx)
        X_out.append(avg_embed_vec)
    
    return X_out,Y_out

def evaluate(X_train,Y_train,X_test,Y_test,threshold):
    Y_true,Y_pred,Y_dis=[],[],[]
    for idx_test in range(len(Y_test)):
        query_x=X_test[idx_test]
        subject_name=Y_test[idx_test]
        for idx_train in range(len(Y_train)):
            centroid_x=X_train[idx_train]
            distance_x=distance.euclidean(centroid_x,query_x)
            if distance_x<=threshold:
                Y_pred.append(0)
            else:
                Y_pred.append(1)
            if subject_name == Y_train[idx_train]:
                # print("Same: "+str(distance_x))
                Y_true.append(0)
            else:
                # print("Diff: "+str(distance_x))
                Y_true.append(1)
            Y_dis.append(distance_x)

    return Y_true,Y_pred,Y_dis

def get_EER(X_train,Y_train,X_test,Y_test,no_outer):
    X_train,Y_train=get_centroid_embed_vec(X_train,Y_train,no_outer)
    Y_true,Y_pred,Y_dis=evaluate(X_train,Y_train,X_test,Y_test,23.6454)
        
    fpr, tpr, thresholds = roc_curve(Y_true,Y_dis, pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer

def verification_evaluate(train_loader,valid_loader,model,no_outer):
    X_train,Y_train=get_embedded_set(train_loader,model)
    X_test,Y_test=get_embedded_set(valid_loader,model)
    return get_EER(X_train,Y_train,X_test,Y_test,no_outer)