import os
import argparse
import pickle
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from matplotlib import pyplot as plt

from training.trainer import train, test, EarlyStopping
from utils.process_dataset import split_train_valid_test_set, get_outer_set 
from modeling.cnn_lstm import Acce_Gyro_CNN_LSTM, Audio_CNN_LSTM, Multimodality_CNN_LSTM
from modeling.tcn import Acce_Gyro_TCN, Audio_TCN, Multimodality_TCN
from training.dataloader import Acce_Gyro_Dataset, Audio_Dataset, Multimodality_Dataset
from evaluation.verification import verification_evaluate

def main(model_name, Model, Dataset_Loader, breath_type, root_dir, no_outer):
    print(model_name)
    list_test_acc=[]
    list_test_eer_KNN=[]
    list_test_eer_GMM=[]
    
    if os.path.isfile('results/outputs/'+breath_type+'/list_test_acc_'+model_name):
        with open('results/outputs/'+breath_type+'/list_test_acc_'+model_name, 'rb') as filehandle:
            list_test_acc = pickle.load(filehandle)
        with open('results/outputs/'+breath_type+'/list_test_eer_KNN_'+model_name, 'rb') as filehandle:
            list_test_eer_KNN = pickle.load(filehandle)
        with open('results/outputs/'+breath_type+'/list_test_eer_GMM_'+model_name, 'rb') as filehandle:
            list_test_eer_GMM = pickle.load(filehandle)

    for eval_time in range(1):
        # list_train_loss=[]
        # list_valid_loss=[]

        list_filename=[]
        for line in open('data/raw/labels/list_'+breath_type+'_filename.txt'):
            list_filename.append(line.rstrip('\n'))
        list_train_filename, list_valid_filename, list_test_filename=split_train_valid_test_set(list_filename)
        list_train_filename, list_valid_filename, list_test_filename, list_outer_valid_filename, list_outer_test_filename, old_new_name_map=get_outer_set(list_train_filename,list_valid_filename,list_test_filename,no_outer=no_outer)
    
        model=Model(no_outer=no_outer)
        if torch.cuda.is_available():
            model.cuda()
            
        train_set = Dataset_Loader(root_dir=root_dir,filenamelist=list_train_filename,old_new_name_map=old_new_name_map)
        train_loader = DataLoader(train_set,batch_size=128, shuffle=True)

        valid_set = Dataset_Loader(root_dir=root_dir,filenamelist=list_valid_filename,old_new_name_map=old_new_name_map)
        valid_loader = DataLoader(valid_set,batch_size=128, shuffle=True)
        
        test_set = Dataset_Loader(root_dir=root_dir,filenamelist=list_test_filename,old_new_name_map=old_new_name_map)
        test_loader = DataLoader(test_set,batch_size=128, shuffle=True)

        optimizer=torch.optim.Adam(model.parameters(), lr=0.001)
        early_stopping = EarlyStopping(checkpoint_name = 'results/models/'+breath_type+'/'+model_name+'_'+'checkpoint.pt', 
            lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.5), patiences=[20, 15, 10, 5])

        epoch=0
        coverage = False
        while True:
            epoch+=1
            train(train_loader=train_loader,model=model,loss_fn=nn.CrossEntropyLoss(),optimizer=optimizer,epoch=epoch)

            train_losses, train_correct = test(test_loader=train_loader,model=model,loss_fn=nn.CrossEntropyLoss())
            valid_losses, valid_correct = test(test_loader=valid_loader,model=model,loss_fn=nn.CrossEntropyLoss())
            # print('Valid set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            #     np.mean(valid_losses), valid_correct, len(valid_loader.dataset),
            #     100. * valid_correct / len(valid_loader.dataset))) 
            
            # list_train_loss.append(np.mean(train_losses))
            # list_valid_loss.append(np.mean(valid_losses))

            if coverage == False and (train_correct / len(train_loader.dataset)) > 0.9:
                coverage = True
            
            if coverage == True:
                early_stopping(np.mean(valid_losses), model)
                if early_stopping.early_stop:
                    break

        model.load_state_dict(torch.load('results/models/'+breath_type+'/'+model_name+'_'+'checkpoint.pt'))
        train_losses, train_correct = test(test_loader=train_loader,model=model,loss_fn=nn.CrossEntropyLoss())
        print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            np.mean(train_losses), train_correct, len(train_loader.dataset),
            100. * train_correct / len(train_loader.dataset)))

        test_losses, test_correct = test(test_loader=test_loader,model=model,loss_fn=nn.CrossEntropyLoss())
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            np.mean(test_losses), test_correct, len(test_loader.dataset),
            100. * test_correct / len(test_loader.dataset)))   
        list_test_acc.append(100. * test_correct / len(test_loader.dataset))
        
        list_test_filename += list_outer_test_filename
        test_set = Dataset_Loader(root_dir=root_dir,filenamelist=list_test_filename,old_new_name_map=old_new_name_map)
        test_loader = DataLoader(test_set,batch_size=128, shuffle=True)
        eer_KNN,eer_GMM=verification_evaluate(train_loader,test_loader,model,no_outer)
        list_test_eer_KNN.append(eer_KNN)
        list_test_eer_GMM.append(eer_GMM)
        
        # plt.savefig('results/outputs/multi_modality_grad_TCN.png')
        # plt.clf()
        # plt.plot(list_train_loss,color='blue', label = 'train loss')
        # plt.plot(list_valid_loss,color='green', label = 'valid loss')
        # plt.savefig('results/outputs/acce_gyro_loss.png')
        # exit()
        # torch.save(model, "checkpoints/model_epoch_"+str(eval_time))
  
    with open('results/outputs/'+breath_type+'/list_test_acc_'+model_name, 'wb') as filehandle:
        pickle.dump(list_test_acc, filehandle)
    with open('results/outputs/'+breath_type+'/list_test_eer_KNN_'+model_name, 'wb') as filehandle:
        pickle.dump(list_test_eer_KNN, filehandle)
    with open('results/outputs/'+breath_type+'/list_test_eer_GMM_'+model_name, 'wb') as filehandle:
        pickle.dump(list_test_eer_GMM, filehandle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train model')
    parser.add_argument('--breath_type', required=True, type=str, help='breath type: normal, deep, strong')
    parser.add_argument('--model_type', required=True, type=str, help='model type: acce-gyro, audio, multi')
    parser.add_argument('--model_name', required=True, type=str, help='model name: cnn-lstm, tcn')
    parser.add_argument('--no_outer', default=0, type=int, help='number of outers')
    args = parser.parse_args()
    
    if args.model_type=="acce-gyro":
        Dataset_Loader=Acce_Gyro_Dataset
    elif args.model_type=="audio":
        Dataset_Loader=Audio_Dataset
    else:
        Dataset_Loader=Multimodality_Dataset
    
    if args.model_name == "cnn-lstm":
        if args.model_type == "acce-gyro":
            Model=Acce_Gyro_CNN_LSTM
        elif args.model_type == "audio":
            Model=Audio_CNN_LSTM
        else:
            Model=Multimodality_CNN_LSTM
    else:
        if args.model_type == "acce-gyro":
            Model=Acce_Gyro_TCN
        elif args.model_type == "audio":
            Model=Audio_TCN
        else:
            Model=Multimodality_TCN
    
    main(model_name=args.breath_type+'_'+args.model_type+'_'+args.model_name+'_'+str(args.no_outer), Model=Model,
        Dataset_Loader=Dataset_Loader, breath_type=args.breath_type, root_dir="", no_outer=args.no_outer)
