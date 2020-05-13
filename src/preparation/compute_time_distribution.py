import os
import glob
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import collections
import seaborn as sns
import librosa

def convert_dict_2_matplotlib_input(person_label):
    list_person_name=[]
    list_normal_no=[]
    list_deep_no=[]
    list_strong_no=[]
    for key in person_label:
        person_name=key
        label_no=person_label[key]
        list_person_name.append(person_name)
        if 'normal' in label_no:
            list_normal_no.append(label_no['normal'])
        else:
            list_normal_no.append(0)
        if 'deep' in label_no:
            list_deep_no.append(label_no['deep'])
        else:
            list_deep_no.append(0)
        if 'strong' in label_no:
            list_strong_no.append(label_no['strong'])
        else:
            list_strong_no.append(0)

    return list_person_name,list_normal_no,list_deep_no,list_strong_no

def flatten(list_input):
    new_list = []
    for i in list_input:
        if i==0:
            continue
        for j in i:
            new_list.append(j)
    return new_list

def main():
    person_label=dict()
    for file_name in glob.iglob('data/raw/audio/**', recursive=True):
        if os.path.isfile(file_name):
            person_name=file_name.split("/",3)[3].split(".",1)[0].split("_",1)[0]
            label_name=file_name.split("/",5)[4]
            if person_name not in person_label:
                person_label.update({person_name:{'normal':[],'strong':[],'deep':[]}})
            data, fs = librosa.load(file_name,sr=16000)
            time_length=len(data)/fs
            person_label[str(person_name)][str(label_name)].append(time_length)
    
    person_label=collections.OrderedDict(sorted(person_label.items()))
    list_person_name,list_normal_no,list_deep_no,list_strong_no=convert_dict_2_matplotlib_input(person_label)
    list_normal_no=flatten(list_normal_no)
    list_strong_no=flatten(list_strong_no)
    list_deep_no=flatten(list_deep_no)

    print(len(list_normal_no))
    print(len(list_deep_no))
    print(len(list_strong_no))

    print("MIN")
    print(np.min(list_normal_no))
    print(np.min(list_deep_no))
    print(np.min(list_strong_no))
    print("MAX")
    print(np.max(list_normal_no))
    print(np.max(list_deep_no))
    print(np.max(list_strong_no))
    print("MEDIAN")
    print(np.median(list_normal_no))
    print(np.median(list_deep_no))
    print(np.median(list_strong_no))
    print("MEAN")
    print(np.mean(list_normal_no))
    print(np.mean(list_deep_no))
    print(np.mean(list_strong_no))
    print("STD")
    print(np.std(list_normal_no))
    print(np.std(list_deep_no))
    print(np.std(list_strong_no))
    # plt.hist(list_strong_no, color = 'blue', edgecolor = 'black',
    #         bins = int(180/5))

    # sns.distplot(list_strong_no, hist=True, kde=False, 
    #          bins=int(180/5), color = 'blue',
    #          hist_kws={'edgecolor':'black'})

    # sns.distplot(list_strong_no, hist=True, kde=True, 
    #          color = 'blue',bins=int(100),
    #          hist_kws={'edgecolor':'black'})

    # # seaborn histogram
    # sns.distplot(list_normal_no, hist=False, kde=True, kde_kws = {'shade': True},
    #             color = 'blue',label='normal')
    # sns.distplot(list_deep_no, hist=False, kde=True, kde_kws = {'shade': True},
    #             color = 'orange',label='deep')
    # sns.distplot(list_strong_no, hist=False, kde=True, kde_kws = {'shade': True},
    #             color = 'green',label='strong')

    # Add labels
    # plt.title('Density Plot of Duration for each Breathing with Different Types')
    # plt.xlabel('Breathing Duration (min)')
    # plt.ylabel('Density')
    # plt.show()
    # visualize_instances_per_person(person_label)

main()