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
from evaluation.verification_one_vs_all import verification_one_vs_all

def main(model_name, Model, Dataset_Loader, breath_type, root_dir, no_outer):
    print(model_name)
    list_test_acc=[]
    list_test_eer=[]
    
    if os.path.isfile('results/outputs/'+breath_type+'/list_test_acc_'+model_name):
        with open('results/outputs/'+breath_type+'/list_test_acc_'+model_name, 'rb') as filehandle:
            list_test_acc = pickle.load(filehandle)
        with open('results/outputs/'+breath_type+'/list_test_eer_'+model_name, 'rb') as filehandle:
            list_test_eer = pickle.load(filehandle)

    # list_train_loss=[]
    # list_valid_loss=[]

    list_filename=[]
    for line in open('data/raw/labels/list_'+breath_type+'_filename.txt'):
        list_filename.append(line.rstrip('\n'))

    list_train_filename, list_valid_filename, list_test_filename=split_train_valid_test_set(list_filename)
    list_train_filename, list_valid_filename, list_test_filename, list_outer_valid_filename, list_outer_test_filename, old_new_name_map=get_outer_set(list_train_filename,list_valid_filename,list_test_filename,no_outer=no_outer)

    models = []
    for idx in range(20):
        model=Model(no_outer=no_outer, one_vs_all = True)
        if torch.cuda.is_available():
            model.cuda()
                
        train_set = Dataset_Loader(root_dir=root_dir,filenamelist=list_train_filename,old_new_name_map=old_new_name_map, object_id = idx)
        train_loader = DataLoader(train_set,batch_size=128, shuffle=True)

        valid_set = Dataset_Loader(root_dir=root_dir,filenamelist=list_valid_filename,old_new_name_map=old_new_name_map, object_id = idx)
        valid_loader = DataLoader(valid_set,batch_size=128, shuffle=True)
            
        test_set = Dataset_Loader(root_dir=root_dir,filenamelist=list_test_filename,old_new_name_map=old_new_name_map, object_id = idx)
        test_loader = DataLoader(test_set,batch_size=128, shuffle=True)

        optimizer=torch.optim.Adam(model.parameters(), lr=0.001)
        early_stopping = EarlyStopping(checkpoint_name = 'results/models/'+breath_type+'/'+model_name+'_'+str(idx)+'_'+'checkpoint.pt', 
            lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.5), patiences=[20, 15, 10, 5])
            
        coverage = False
        for epoch in range(1000):
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

        model.load_state_dict(torch.load('results/models/'+breath_type+'/'+model_name+'_'+str(idx)+'_'+'checkpoint.pt'))
        model.eval()
        models.append(model)
        # train_losses, train_correct = test(test_loader=train_loader,model=model,loss_fn=nn.CrossEntropyLoss())
        # print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        #     np.mean(train_losses), train_correct, len(train_loader.dataset),
        #     100. * train_correct / len(train_loader.dataset)))

        # test_losses, test_correct = test(test_loader=test_loader,model=model,loss_fn=nn.CrossEntropyLoss())
        # print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        #     np.mean(test_losses), test_correct, len(test_loader.dataset),
        #     100. * test_correct / len(test_loader.dataset)))   
        # list_test_acc.append(100. * test_correct / len(test_loader.dataset))
    
    test_set = Dataset_Loader(root_dir=root_dir,filenamelist=list_test_filename,old_new_name_map=old_new_name_map)
    test_loader = DataLoader(test_set,batch_size=1, shuffle=True)
    correct=0
    with torch.no_grad():
        for batch_idx, (types, samples,labels) in enumerate(test_loader):
            if types[0] == "Multimodality":
                feature_1 = samples[0].float()
                feature_2 = samples[1].float()
                if torch.cuda.is_available():
                    feature_1 = feature_1.cuda()
                    feature_2 = feature_2.cuda()
                    labels = labels.cuda()
                predicted_values = []
                for idx in range(20):
                    model = models[idx]
                    predicted_value = nn.Softmax(dim = 1)(model(feature_1, feature_2))
                    _, predicted_value = torch.max(predicted_value, 1)
                    predicted_values.append(predicted_value.data.cpu().numpy()[0])
            else:
                features = samples.float()
                if torch.cuda.is_available():
                    features= features.cuda()
                    labels=labels.cuda()
                predicted_values = []
                for idx in range(20):
                    model = models[idx]
                    predicted_value = nn.Softmax(dim = 1)(model(features))
                    _, predicted_value = torch.max(predicted_value, 1)
                    predicted_values.append(predicted_value.data.cpu().numpy()[0])

            predicted_value = predicted_values.index(min(predicted_values))
            labels = labels.data.cpu().numpy()[0]
            
            if predicted_value == labels:
                correct += 1
    
    test_correct = correct
    print('Test set: Accuracy: {}/{} ({:.0f}%)'.format(
        test_correct, len(test_loader.dataset),
        100. * test_correct / len(test_loader.dataset)))   
    list_test_acc.append(100. * test_correct / len(test_loader.dataset))

    list_test_filename += list_outer_test_filename
    test_set = Dataset_Loader(root_dir=root_dir,filenamelist=list_test_filename,old_new_name_map=old_new_name_map)
    test_loader = DataLoader(test_set,batch_size=1, shuffle=True)
    eer_score = verification_one_vs_all(test_loader, models, no_outer)
    print(eer_score)
    list_test_eer.append(eer_score)
        
    # plt.savefig('results/outputs/multi_modality_grad_TCN.png')
    # plt.clf()
    # plt.plot(list_train_loss,color='blue', label = 'train loss')
    # plt.plot(list_valid_loss,color='green', label = 'valid loss')
    # plt.savefig('results/outputs/acce_gyro_loss.png')
    # exit()
    # torch.save(model, "checkpoints/model_epoch_"+str(eval_time))
  
    with open('results/outputs/'+breath_type+'/list_test_acc_'+model_name, 'wb') as filehandle:
        pickle.dump(list_test_acc, filehandle)
    with open('results/outputs/'+breath_type+'/list_test_eer_'+model_name, 'wb') as filehandle:
        pickle.dump(list_test_eer, filehandle)


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
