import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_fscore_support,accuracy_score,f1_score
from numba import jit, cuda 

from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from scipy.spatial import distance
import pickle
from matplotlib import pyplot as plt

class TristouNet(nn.Module):
    """TristouNet sequence embedding
    RNN ( » ... » RNN ) » temporal pooling › ( MLP › ... › ) MLP › normalize
    Parameters
    ----------
    n_features : int
        Input feature dimension
    rnn : {'LSTM', 'GRU'}, optional
        Defaults to 'LSTM'.
    recurrent: list, optional
        List of output dimension of stacked RNNs.
        Defaults to [16, ] (i.e. one RNN with output dimension 16)
    bidirectional : bool, optional
        Use bidirectional recurrent layers. Defaults to False.
    pooling : {'sum', 'max'}
        Temporal pooling strategy. Defaults to 'sum'.
    linear : list, optional
        List of hidden dimensions of linear layers. Defaults to [16, 16].
    Reference
    ---------
    Hervé Bredin. "TristouNet: Triplet Loss for Speaker Turn Embedding."
    ICASSP 2017 (https://arxiv.org/abs/1609.04301)
    """

    def __init__(self, n_features,
                 rnn='LSTM', recurrent=[16], bidirectional=False,
                 pooling='sum', linear=[16, 16]):

        super(TristouNet, self).__init__()

        self.n_features = n_features
        self.rnn = rnn
        self.recurrent = recurrent
        self.bidirectional = bidirectional
        self.pooling = pooling
        self.linear = [] if linear is None else linear

        self.num_directions_ = 2 if self.bidirectional else 1

        if self.pooling not in {'sum', 'max'}:
            raise ValueError('"pooling" must be one of {"sum", "max"}')

        # create list of recurrent layers
        self.recurrent_layers_ = []
        input_dim = self.n_features
        for i, hidden_dim in enumerate(self.recurrent):
            if self.rnn == 'LSTM':
                recurrent_layer = nn.LSTM(input_dim, hidden_dim,
                                          bidirectional=self.bidirectional,
                                          batch_first=True)
            elif self.rnn == 'GRU':
                recurrent_layer = nn.GRU(input_dim, hidden_dim,
                                         bidirectional=self.bidirectional,
                                         batch_first=True)
            else:
                raise ValueError('"rnn" must be one of {"LSTM", "GRU"}.')
            self.add_module('recurrent_{0}'.format(i), recurrent_layer)
            self.recurrent_layers_.append(recurrent_layer)
            input_dim = hidden_dim * (2 if self.bidirectional else 1)

        # create list of linear layers
        self.linear_layers_ = []
        for i, hidden_dim in enumerate(self.linear):
            linear_layer = nn.Linear(input_dim, hidden_dim, bias=True)
            self.add_module('linear_{0}'.format(i), linear_layer)
            self.linear_layers_.append(linear_layer)
            input_dim = hidden_dim

    @property
    def output_dim(self):
        if self.linear:
            return self.linear[-1]
        return self.recurrent[-1] * (2 if self.bidirectional else 1)

    def forward(self, sequence):
        """
        Parameters
        ----------
        sequence : (batch_size, n_samples, n_features) torch.Tensor
        """

        packed_sequences = isinstance(sequence, PackedSequence)

        if packed_sequences:
            _, n_features = sequence.data.size()
            batch_size = sequence.batch_sizes[0].item()
            device = sequence.data.device
        else:
            # check input feature dimension
            batch_size, _, n_features = sequence.size()
            device = sequence.device

        if n_features != self.n_features:
            msg = 'Wrong feature dimension. Found {0}, should be {1}'
            raise ValueError(msg.format(n_features, self.n_features))

        output = sequence

        # recurrent layers
        for hidden_dim, layer in zip(self.recurrent, self.recurrent_layers_):

            if self.rnn == 'LSTM':
                # initial hidden and cell states
                h = torch.zeros(self.num_directions_, batch_size, hidden_dim,
                                device=device, requires_grad=False)
                c = torch.zeros(self.num_directions_, batch_size, hidden_dim,
                                device=device, requires_grad=False)
                hidden = (h, c)

            elif self.rnn == 'GRU':
                # initial hidden state
                hidden = torch.zeros(
                    self.num_directions_, batch_size, hidden_dim,
                    device=device, requires_grad=False)

            # apply current recurrent layer and get output sequence
            output, _ = layer(output, hidden)

        if packed_sequences:
            output, lengths = pad_packed_sequence(output, batch_first=True)

        # batch_size, n_samples, dimension

        # average temporal pooling
        if self.pooling == 'sum':
            output = output.sum(dim=1)
        elif self.pooling == 'max':
            if packed_sequences:
                msg = ('"max" pooling is not yet implemented '
                       'for variable length sequences.')
                raise NotImplementedError(msg)
            output, _ = output.max(dim=1)

        # batch_size, dimension

        # stack linear layers
        for hidden_dim, layer in zip(self.linear, self.linear_layers_):

            # apply current linear layer
            output = layer(output)

            # apply non-linear activation function
            output = torch.tanh(output)

        # batch_size, dimension

        # unit-normalize
        norm = torch.norm(output, 2, 1, keepdim=True)
        output = output / norm

        return output

class BreathDataset(Dataset):
    def __init__(self,root_dir,filenames_filename):
        self.filenamelist = []
        self.root_dir=root_dir
        for line in open(filenames_filename):
            self.filenamelist.append(line.rstrip('\n'))
        self.dict_filename=self.__init__dict_filename(self.filenamelist)

    def __init__dict_filename(self,filenamelist):
        dict_filename={}
        for idx in range(len(filenamelist)):
            subject_name=filenamelist[idx].split("/")[2]
            if subject_name not in dict_filename:
                dict_filename.update({subject_name:[]})
            dict_filename[subject_name].append(idx)

        return dict_filename

    def set_uniform_triplets(self):
        self.triplets=[]
        for anchor_subject_name,list_anchor_instance_idx in self.dict_filename.items():
            for anchor_idx in range(len(list_anchor_instance_idx)):
                anchor_feature=self.audio_extracted_loader(self.root_dir+self.filenamelist[list_anchor_instance_idx[anchor_idx]])
                list_neg_instance_idx=[]
                for neg_subject_name,neg_list in self.dict_filename.items():
                    if anchor_subject_name!=neg_subject_name:
                        list_neg_instance_idx+=neg_list
                for pos_idx in range(anchor_idx+1,len(list_anchor_instance_idx)):
                    neg_idx=random.randrange(0,len(list_neg_instance_idx),1)
                    self.triplets.append([list_anchor_instance_idx[anchor_idx],list_anchor_instance_idx[pos_idx],
                        list_neg_instance_idx[neg_idx]])
            print(anchor_subject_name,len(self.triplets))

    def set_hard_negative_triplets(self,embedding_model,margin):
        embedding_model.eval()
        self.triplets=[]
        for anchor_subject_name,list_anchor_instance_idx in self.dict_filename.items():
            for anchor_idx in range(len(list_anchor_instance_idx)):
                anchor_feature=self.audio_extracted_loader(self.root_dir+self.filenamelist[list_anchor_instance_idx[anchor_idx]])
                anchor_feature=anchor_feature.reshape(1,anchor_feature.shape[0],anchor_feature.shape[1])
                anchor_feature=torch.from_numpy(anchor_feature).float()
                if torch.cuda.is_available():
                    anchor_feature= anchor_feature.cuda()
                anchor_feature=embedding_model(anchor_feature)
                list_neg_instance_idx=[]
                for neg_subject_name,neg_list in self.dict_filename.items():
                    if anchor_subject_name!=neg_subject_name:
                        list_neg_instance_idx+=neg_list
                for pos_idx in range(anchor_idx+1,len(list_anchor_instance_idx)):
                    positive_feature=self.audio_extracted_loader(self.root_dir+self.filenamelist[list_anchor_instance_idx[pos_idx]])
                    positive_feature=positive_feature.reshape(1,positive_feature.shape[0],positive_feature.shape[1])
                    positive_feature=torch.from_numpy(positive_feature).float()
                    if torch.cuda.is_available():
                        positive_feature= positive_feature.cuda()
                    positive_feature=embedding_model(positive_feature)
                    neg_idx=0
                    list_neg_instance_idx_idx=random.sample(range(len(list_neg_instance_idx)),len(list_neg_instance_idx))
                    while neg_idx<len(list_neg_instance_idx_idx):
                        negative_feature=self.audio_extracted_loader(self.root_dir+self.filenamelist[list_neg_instance_idx[list_neg_instance_idx_idx[neg_idx]]])
                        negative_feature=negative_feature.reshape(1,negative_feature.shape[0],negative_feature.shape[1])
                        negative_feature=torch.from_numpy(negative_feature).float()
                        if torch.cuda.is_available():
                            negative_feature = negative_feature.cuda()
                        negative_feature=embedding_model(negative_feature)
                        distance_positive_feature = (anchor_feature - positive_feature).pow(2).sum(1)  # .pow(.5)
                        distance_negative_feature = (anchor_feature - negative_feature).pow(2).sum(1)  # .pow(.5)
                        # if distance_negative_feature<distance_positive_feature+margin:
                        #     break
                        if distance_negative_feature<distance_positive_feature+margin:
                            break
                        neg_idx+=1
                    if neg_idx<len(list_neg_instance_idx_idx):
                        self.triplets.append([list_anchor_instance_idx[anchor_idx],list_anchor_instance_idx[pos_idx],
                            list_neg_instance_idx[list_neg_instance_idx_idx[neg_idx]]])
            print(anchor_subject_name,len(self.triplets))

    def set_semihard_negative_triplets(self,embedding_model,margin):
        embedding_model.eval()
        self.triplets=[]
        for anchor_subject_name,list_anchor_instance_idx in self.dict_filename.items():
            for anchor_idx in range(len(list_anchor_instance_idx)):
                anchor_feature=self.audio_extracted_loader(self.root_dir+self.filenamelist[list_anchor_instance_idx[anchor_idx]])
                anchor_feature=anchor_feature.reshape(1,anchor_feature.shape[0],anchor_feature.shape[1])
                anchor_feature=torch.from_numpy(anchor_feature).float()
                if torch.cuda.is_available():
                    anchor_feature= anchor_feature.cuda()
                anchor_feature=embedding_model(anchor_feature)
                list_neg_instance_idx=[]
                for neg_subject_name,neg_list in self.dict_filename.items():
                    if anchor_subject_name!=neg_subject_name:
                        list_neg_instance_idx+=neg_list
                for pos_idx in range(anchor_idx+1,len(list_anchor_instance_idx)):
                    positive_feature=self.audio_extracted_loader(self.root_dir+self.filenamelist[list_anchor_instance_idx[pos_idx]])
                    positive_feature=positive_feature.reshape(1,positive_feature.shape[0],positive_feature.shape[1])
                    positive_feature=torch.from_numpy(positive_feature).float()
                    if torch.cuda.is_available():
                        positive_feature= positive_feature.cuda()
                    positive_feature=embedding_model(positive_feature)
                    neg_idx=0
                    list_neg_instance_idx_idx=random.sample(range(len(list_neg_instance_idx)),len(list_neg_instance_idx))
                    while neg_idx<len(list_neg_instance_idx_idx):
                        negative_feature=self.audio_extracted_loader(self.root_dir+self.filenamelist[list_neg_instance_idx[list_neg_instance_idx_idx[neg_idx]]])
                        negative_feature=negative_feature.reshape(1,negative_feature.shape[0],negative_feature.shape[1])
                        negative_feature=torch.from_numpy(negative_feature).float()
                        if torch.cuda.is_available():
                            negative_feature = negative_feature.cuda()
                        negative_feature=embedding_model(negative_feature)
                        distance_positive_feature = (anchor_feature - positive_feature).pow(2).sum(1)  # .pow(.5)
                        distance_negative_feature = (anchor_feature - negative_feature).pow(2).sum(1)  # .pow(.5)
                        # if distance_negative_feature<distance_positive_feature+margin:
                        #     break
                        if distance_positive_feature<distance_negative_feature and distance_negative_feature<distance_positive_feature+margin:
                            break
                        neg_idx+=1
                    if neg_idx<len(list_neg_instance_idx_idx):
                        self.triplets.append([list_anchor_instance_idx[anchor_idx],list_anchor_instance_idx[pos_idx],
                            list_neg_instance_idx[list_neg_instance_idx_idx[neg_idx]]])
            print(anchor_subject_name,len(self.triplets))

    def __len__(self):
        return len(self.triplets)

    def audio_extracted_loader(self,file_name):
        file_name=file_name.replace("extracted","mfcc")
        return np.load(file_name+".npy")  

    def __getitem__(self, index):
        path1,path2,path3 = self.triplets[index]
        img1 = self.audio_extracted_loader(self.root_dir+self.filenamelist[path1])
        img2 = self.audio_extracted_loader(self.root_dir+self.filenamelist[path2])
        img3 = self.audio_extracted_loader(self.root_dir+self.filenamelist[path3])

        return img1, img2, img3

class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()

def train(train_loader,model,loss_fn,optimizer,epoch):
    batch_losses = []
    model.train()
    losses=[]
    for batch_idx, (data1, data2, data3) in enumerate(train_loader):
        data1=data1.float()
        data2=data2.float()
        data3=data3.float()
        if torch.cuda.is_available():
            data1, data2, data3 = data1.cuda(), data2.cuda(), data3.cuda()
        data1, data2, data3 = Variable(data1), Variable(data2), Variable(data3)

        optimizer.zero_grad()
        embedded_x, embedded_y, embedded_z = model(data1), model(data2), model(data3)
        loss_outputs=loss_fn(embedded_x,embedded_y,embedded_z)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        batch_losses.append(loss.item())
        losses.append(loss.item())
        loss.backward()
        plot_grad_flow(model.named_parameters())
        optimizer.step()
        
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{}]\t'
                  'Loss: {:.4f} \t'.format(
                epoch, batch_idx * len(data1), len(train_loader.dataset),
                np.mean(batch_losses)))     
            batch_losses = []
    
    return np.mean(losses)

def audio_extracted_loader(file_name):
    file_name=file_name.replace("extracted","mfcc")
    return np.load(file_name+".npy")    

def get_dataset(model,root_dir,list_filename):
    filename_set=[]
    for line in open(list_filename):
        filename_set.append(line.rstrip('\n'))
    labels = []
    embedded_features = []
    for test_file in filename_set:
        audio_feature=audio_extracted_loader(root_dir+test_file)
        audio_feature=audio_feature.reshape(1,audio_feature.shape[0],audio_feature.shape[1])
        audio_feature = torch.from_numpy(audio_feature).float()
        if torch.cuda.is_available():
            audio_feature=audio_feature.cuda()
        embedded_feature=model(audio_feature)
        embedded_feature=embedded_feature.cpu().numpy()
        embedded_feature=embedded_feature.reshape(16)
        embedded_features.append(embedded_feature)
        subject_name=test_file.split("/")[2].split("_")[0]
        labels.append(subject_name)
    return embedded_features,labels

def get_centroid_embed_vec(X_train,Y_train):
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
        avg_embed_vec=np.zeros(16)
        for idx in list_idx:
            avg_embed_vec+=X_train[idx]
        avg_embed_vec=avg_embed_vec/len(list_idx)
        X_out.append(avg_embed_vec)
    
    return X_out,Y_out

def evaluate(X_train,Y_train,X_test,Y_test):
    Y_true,Y_dis=[],[]
    for idx_test in range(len(Y_test)):
        query_x=X_test[idx_test]
        subject_name=Y_test[idx_test]
        for idx_train in range(len(Y_train)):
            centroid_x=X_train[idx_train]
            distance_x=distance.euclidean(centroid_x,query_x)

            if subject_name == Y_train[idx_train]:
                Y_true.append(0)
            else:
                Y_true.append(1)
            Y_dis.append(distance_x)

    return Y_true,Y_dis

def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")

# @jit(target ="cuda")                              
def main(root_dir):
    # embedding_model=torch.load("44k_mfcc_embedding_model_epoch_20")
    embedding_model=TristouNet(n_features=35)
    if torch.cuda.is_available():
        embedding_model.cuda()
    losses=[]
    eer_list=[]
    score_list=[]
    for epoch in range(0, 100):
        with torch.no_grad():
            if epoch%10==0: 
                torch.save(embedding_model, "checkpoints/embedding_model_epoch_"+str(epoch))
                dataset = BreathDataset(root_dir=root_dir,filenames_filename='train_valid_set/list_train_filename_16k.txt')
                dataset.set_hard_negative_triplets(embedding_model,margin=0.2)
                # dataset.set_uniform_triplets()
                train_loader = DataLoader(dataset,batch_size=128, shuffle=True)
        
            X_train,Y_train=get_dataset(embedding_model,root_dir,'train_valid_set/list_train_filename_16k.txt')
            X_test,Y_test=get_dataset(embedding_model,root_dir,'train_valid_set/list_test_filename_16k.txt')
            X_train,Y_train=get_centroid_embed_vec(X_train,Y_train)
            Y_true,Y_dis=evaluate(X_train,Y_train,X_test,Y_test)

            fpr, tpr, thresholds = roc_curve(Y_true,Y_dis, pos_label=1)
            eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
            thresh = interp1d(fpr, thresholds)(eer)
            print(eer)
            eer_list.append(eer)

            knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean').fit(X_train,Y_train)
            Y_pred=knn.predict(X_test)
            acc_score=accuracy_score(Y_test,Y_pred)
            precision_recall_fscore_support_score=precision_recall_fscore_support(Y_test,Y_pred,average="weighted")
            print(precision_recall_fscore_support_score)
            score_list.append(precision_recall_fscore_support_score)

        loss=train(train_loader=train_loader,model=embedding_model,loss_fn=TripletLoss(margin=0.2),
            optimizer=torch.optim.Adam(embedding_model.parameters(), lr=1e-3),epoch=epoch)
        losses.append(loss)

    with open("plots/eer_list.txt", "wb") as fp:   #Pickling
        pickle.dump(eer_list, fp)
    with open("plots/score_list.txt", "wb") as fp:   #Pickling
        pickle.dump(score_list, fp)
    with open("plots/losses.txt", "wb") as fp:   #Pickling
        pickle.dump(losses, fp)
    torch.save(embedding_model, "checkpoints/embedding_model_epoch_"+str(epoch))

os.environ["CUDA_VISIBLE_DEVICES"]="1"
torch.set_num_threads(1)
if os.path.exists("/mnt/habm1/vinai/habm1/"):
    main(root_dir="/mnt/habm1/vinai/habm1/")
elif os.path.exists("dataset/drive/My Drive/"):
    main(root_dir="dataset/drive/My Drive/")
else:
    main(root_dir="")