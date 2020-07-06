import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from scipy.spatial import distance
from sklearn.mixture import GaussianMixture
import torch
import torch.nn as nn

def get_predicted_set(loader, models):
    Y_true, Y_out=[],[]
    for batch_idx, (types, samples, labels) in enumerate(loader):
        if types[0] == "Multimodality":
            feature_1 = samples[0].float()
            feature_2 = samples[1].float()
            if torch.cuda.is_available():
                feature_1 = feature_1.cuda()
                feature_2 = feature_2.cuda()
                labels = labels.cuda()
            
            label = labels.data.cpu().numpy()[0]
            for idx in range(20):
                model = models[idx]
                predicted_value = nn.Softmax(dim = 1)(model(feature_1, feature_2))
                _, predicted_value = torch.max(predicted_value, 1)
                Y_out.append(predicted_value.data.cpu().numpy()[0])
                if label == idx:
                    Y_true.append(0)
                else:
                    Y_true.append(1)
        else:
            features = samples.float()
            if torch.cuda.is_available():
                features = features.cuda()
                labels = labels.cuda()
            
            label = labels.data.cpu().numpy()[0]
            for idx in range(20):
                model = models[idx]
                predicted_value = nn.Softmax(dim = 1)(model(features))
                _, predicted_value = torch.max(predicted_value, 1)
                Y_out.append(predicted_value.data.cpu().numpy()[0])
                if label == idx:
                    Y_true.append(0)
                else:
                    Y_true.append(1)

    return Y_true, Y_out

def get_EER(Y_true, Y_pred):        
    fpr, tpr, thresholds = roc_curve(Y_true,Y_pred, pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer

def verification_one_vs_all(valid_loader, models, no_outer):
    Y_true, Y_pred = get_predicted_set(valid_loader, models)
    return get_EER(Y_true,Y_pred)