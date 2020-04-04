
import os
import glob
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
from itertools import chain, repeat, islice
import librosa

def extract_audio(file_name,sampling_rate,extractor):
    data, fs = librosa.load(file_name,sr=sampling_rate)
    if len(data)>fs*9:
        return
    '''Padding center with max length of a breathing (9s)'''
    data=librosa.util.pad_center(data,fs*9)
    
    '''Extract acoustic features'''
    if extractor=='mfcc':
        Zxx=librosa.feature.mfcc(data,sr=fs,n_mfcc=32,n_fft=int(sampling_rate*0.032),hop_length=int(sampling_rate*0.02))
    elif extractor=='stft':
        Zxx=librosa.core.stft(data,n_fft=int(sampling_rate*0.032),hop_length=int(sampling_rate*0.02))
        Zxx=Zxx.astype(float)
    else:
        return -1

    Zxx=Zxx[:, :-1]
    return Zxx

def main():
    root_folder=""
    
    for file_name in glob.iglob('data/raw/audio/**', recursive=True):
        if os.path.isfile(file_name):
            '''Extract acoustic signals using MFCC'''
            extracted_features=extract_audio(file_name,sampling_rate=16000,extractor='mfcc')
            
            '''Write extracted files to data/processed'''
            ef_file_name=file_name.replace("raw/audio","processed/audio")
            dir_name_splited=ef_file_name.split("/")[:-1]
            dir_name=root_folder
            for sub_dir in dir_name_splited:
                dir_name+=sub_dir+"/"
                if not os.path.exists(dir_name):
                    os.mkdir(dir_name)
            np.save(root_folder+ef_file_name,extracted_features)
            
main()