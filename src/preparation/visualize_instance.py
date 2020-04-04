import librosa
import pandas as pd
import matplotlib.pyplot as plt
import librosa.display
def read_acce_gyro_data(file_name):
    column_names = ['timestamp','x_acc', 'y_acc', 'z_acc','x_gyr', 'y_gyr', 'z_gyr']
    data = pd.read_csv(file_name,header = None, names = column_names,delimiter='\t')
    return data

def visual_audio(file_name, type_name, pos):
    file_name=file_name.replace("out","out/audio")
    y, sr = librosa.load(file_name,sr=16000)
    plt.subplot(3, 3, pos)
    librosa.display.waveplot(y, sr=sr)
    plt.xlabel("Time (s)")
    plt.title(type_name)

def visual_acce(file_name, type_name,pos):
    file_name=file_name.replace("out","out/acce_gyro")
    data = read_acce_gyro_data(file_name)
    plt.subplot(3, 3, pos)
    times=data['timestamp']
    plt.plot(times, data['x_acc'], 'r')
    plt.plot(times, data['y_acc'], 'b') 
    plt.plot(times, data['z_acc'], 'g')
    plt.xlabel("Time (s)")
    plt.title(type_name)

def visual_gyro(file_name, type_name,pos):
    file_name=file_name.replace("out","out/acce_gyro")
    data = read_acce_gyro_data(file_name)
    plt.subplot(3, 3, pos)
    times=data['timestamp']
    plt.plot(times, data['x_gyr'], 'r')
    plt.plot(times, data['y_gyr'], 'b') 
    plt.plot(times, data['z_gyr'], 'g')
    plt.xlabel("Time (s)")
    plt.title(type_name)

def main():
    normal_file_name="data/raw/01_male_BQuyen/deep/01_male_BQuyen_deep_fae944610a39d5f3f810c475dc46c0c5"
    deep_file_name="data/raw/10_male_Khanh/deep/10_male_Khanh_deep_49c3c5d24ac65fcac2f55a32bec9481e"
    strong_file_name="data/raw/01_male_BQuyen/strong/01_male_BQuyen_strong_4cb964b0dc93081395d06b5d40aa4d52"
    visual_audio(normal_file_name,"Normal Audio",1)
    visual_acce(normal_file_name,"Normal Accelerometer",2)
    visual_gyro(normal_file_name,"Normal Gyroscope",3)
    visual_audio(deep_file_name,"Deep Audio",4)
    visual_acce(deep_file_name,"Deep Accelerometer",5)
    visual_gyro(deep_file_name,"Deep Gyroscope",6)
    visual_audio(strong_file_name,"Strong Audio",7)
    visual_acce(strong_file_name,"Strong Accelerometer",8)
    visual_gyro(strong_file_name,"Strong Gyroscope",9)
    plt.tight_layout()
    plt.show()

main()