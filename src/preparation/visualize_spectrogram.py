import librosa
import matplotlib.pyplot as plt
import librosa.display

def visual_audio(file_name, type_name, pos):
    file_name=file_name.replace("out","out/audio")
    y, sr = librosa.load(file_name,sr=16000)
    plt.subplot(3, 2, pos)
    librosa.display.waveplot(y, sr=sr)
    plt.xlabel("Time (s)")
    plt.title(type_name)

def visual_spectrogram(file_name, type_name, pos):
    file_name=file_name.replace("out","out/audio")
    data, fs = librosa.load(file_name,sr=16000)
    Zxx=librosa.feature.mfcc(data,sr=fs,n_mfcc=32,n_fft=int(fs*0.032),hop_length=int(fs*0.02))
    plt.subplot(3, 2, pos)
    librosa.display.specshow(Zxx, x_axis='time')
    plt.colorbar()
    plt.xlabel("Time (s)")
    plt.title(type_name)

def main():
    normal_file_name="data/raw/01_male_BQuyen/deep/01_male_BQuyen_deep_fae944610a39d5f3f810c475dc46c0c5"
    deep_file_name="data/raw/10_male_Khanh/deep/10_male_Khanh_deep_49c3c5d24ac65fcac2f55a32bec9481e"
    strong_file_name="data/raw/01_male_BQuyen/strong/01_male_BQuyen_strong_4cb964b0dc93081395d06b5d40aa4d52"
    visual_audio(normal_file_name,"Normal Audio",1)
    visual_spectrogram(normal_file_name,"Normal Spectrogram",2)
    visual_audio(deep_file_name,"Deep Audio",3)
    visual_spectrogram(deep_file_name,"Deep Spectrogram",4)
    visual_audio(strong_file_name,"Strong Audio",5)
    visual_spectrogram(strong_file_name,"Strong Spectrogram",6)
    plt.tight_layout()
    plt.show()

main()