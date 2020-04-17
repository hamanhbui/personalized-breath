
import os
import glob
import numpy as np
import pandas as pd
import librosa
def read_acce_gyro_data(file_name):
    column_names = ['timestamp','x_acc', 'y_acc', 'z_acc','x_gyr', 'y_gyr', 'z_gyr']
    data = pd.read_csv(file_name,header = None, names = column_names,delimiter='\t')
    return data

def padding_signals(segmented_df):
    segmented_df_x_acc=librosa.util.pad_center(np.asarray(segmented_df['x_acc'].tolist()),226)
    segmented_df_y_acc=librosa.util.pad_center(np.asarray(segmented_df['y_acc'].tolist()),226)
    segmented_df_z_acc=librosa.util.pad_center(np.asarray(segmented_df['z_acc'].tolist()),226)
    segmented_df_x_gyr=librosa.util.pad_center(np.asarray(segmented_df['x_gyr'].tolist()),226)
    segmented_df_y_gyr=librosa.util.pad_center(np.asarray(segmented_df['y_gyr'].tolist()),226)
    segmented_df_z_gyr=librosa.util.pad_center(np.asarray(segmented_df['z_gyr'].tolist()),226)
    segments=[segmented_df_x_acc,segmented_df_y_acc,segmented_df_z_acc,segmented_df_x_gyr,segmented_df_y_gyr,segmented_df_z_gyr]
    segments=np.asarray(segments)
    return segments

def main():
    root_folder=""
    
    for file_name in glob.iglob('data/raw/acce-gyro/**', recursive=True):
        if os.path.isfile(file_name):
            data = read_acce_gyro_data(file_name)

            '''Padding center with max length of a breathing (9s) sampling_rate=50Hz'''
            extracted_features=padding_signals(data)

            '''Write padded files to data/processed'''
            ef_file_name=file_name.replace("raw/acce-gyro","processed/acce-gyro")
            dir_name_splited=ef_file_name.split("/")[:-1]
            dir_name=root_folder
            for sub_dir in dir_name_splited:
                dir_name+=sub_dir+"/"
                if not os.path.exists(dir_name):
                    os.mkdir(dir_name)
            np.save(root_folder+ef_file_name,extracted_features)
            
main()