import os
import glob
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import collections

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

def visualize_instances_per_person(person_label):
    list_del=[]
    for person,samples in person_label.items():
        if ('normal' not in samples.keys()) or ('strong' not in samples.keys()) or ('deep' not in samples.keys()):
            list_del.append(person)
        else:
            min_value=min(samples["normal"],samples["strong"],samples["deep"])
            if min_value<15:
                list_del.append(person)
            else:
                samples["normal"]=min_value
                samples["strong"]=min_value
                samples["deep"]=min_value
    
    for person in list_del:
        del person_label[person]
        
    list_person_name,list_normal_no,list_deep_no,list_strong_no=convert_dict_2_matplotlib_input(person_label)
    x = np.arange(len(list_person_name))  # the label locations
    width = 0.5  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, list_normal_no, width/2, label='Normal')
    rects2 = ax.bar(x, list_deep_no, width/2, label='Deep')
    rects3 = ax.bar(x + width/2, list_strong_no, width/2, label='Strong')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Units')
    ax.set_title('Number of breathing instances per person')
    ax.set_xticks(x)
    ax.set_xticklabels(list_person_name)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',fontsize=6)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    fig.tight_layout()
    plt.show()    

def main():
    person_label=dict()
    for file_name in glob.iglob('data/raw/audio/**', recursive=True):
        if os.path.isfile(file_name):
            person_name=file_name.split("/",3)[3].split(".",1)[0].split("_",1)[0]
            label_name=file_name.split("/",5)[4]
            if person_name not in person_label:
                person_label.update({person_name:{'normal':0,'strong':0,'deep':0}})
            person_label[str(person_name)][str(label_name)]+=1
    
    person_label=collections.OrderedDict(sorted(person_label.items()))
    visualize_instances_per_person(person_label)

main()