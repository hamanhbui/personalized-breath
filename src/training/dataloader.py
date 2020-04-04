import numpy as np
from torch.utils.data import Dataset

class Audio_Dataset(Dataset):
    def __init__(self,root_dir,filenamelist,old_new_name_map):
        self.filenamelist = filenamelist
        self.root_dir=root_dir
        self.old_new_name_map=old_new_name_map

    def __len__(self):
        return len(self.filenamelist)

    def data_extracted_loader(self,file_name):
        audio_file_name=file_name.replace("processed","processed/audio")
        return np.load(audio_file_name+".npy")

    def __getitem__(self, index):
        if len(self.old_new_name_map)==0:
            subject_name=int(self.filenamelist[index].split("/")[2].split("_")[0])-1
        else:
            subject_name=self.old_new_name_map[self.filenamelist[index].split("/")[2].split("_")[0]]
        return self.data_extracted_loader(self.root_dir+self.filenamelist[index]),subject_name

class Acce_Gyro_Dataset(Dataset):
    def __init__(self,root_dir,filenamelist,old_new_name_map):
        self.filenamelist = filenamelist
        self.root_dir=root_dir
        self.old_new_name_map=old_new_name_map

    def __len__(self):
        return len(self.filenamelist)

    def data_extracted_loader(self,file_name):
        sensor_file_name=file_name.replace("processed","processed/acce-gyro")
        return np.load(sensor_file_name+".npy")

    def __getitem__(self, index):
        if len(self.old_new_name_map)==0:
            subject_name=int(self.filenamelist[index].split("/")[2].split("_")[0])-1
        else:
            subject_name=self.old_new_name_map[self.filenamelist[index].split("/")[2].split("_")[0]]
        return self.data_extracted_loader(self.root_dir+self.filenamelist[index]),subject_name
        
class Multimodality_Dataset(Dataset):
    def __init__(self,root_dir,filenamelist,old_new_name_map):
        self.filenamelist = filenamelist
        self.root_dir=root_dir
        self.old_new_name_map=old_new_name_map

    def __len__(self):
        return len(self.filenamelist)

    def data_extracted_loader(self,file_name):
        audio_file_name=file_name.replace("processed","processed/audio")
        sensor_file_name=file_name.replace("processed","processed/acce-gyro")
        return np.load(sensor_file_name+".npy"), np.load(audio_file_name+".npy")

    def __getitem__(self, index):
        if len(self.old_new_name_map)==0:
            subject_name=int(self.filenamelist[index].split("/")[2].split("_")[0])-1
        else:
            subject_name=self.old_new_name_map[self.filenamelist[index].split("/")[2].split("_")[0]]
        return self.data_extracted_loader(self.root_dir+self.filenamelist[index]),subject_name
