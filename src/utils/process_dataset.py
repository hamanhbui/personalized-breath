import random

def split_train_valid_test_set(list_filename):
    random.shuffle(list_filename)
    list_test_filename=[]
    list_valid_filename=[]
    list_train_filename=[]
    for filename in list_filename:
        subject_name=filename.split("/")[2]
        ''' Add to valid set '''
        count=0
        for valid_intance in list_valid_filename:
            if subject_name in valid_intance:
                count+=1
        if count<5:
            list_valid_filename.append(filename)
            continue

        ''' Add to test set '''
        count=0
        for test_instance in list_test_filename:
            if subject_name in test_instance:
                count+=1
        if count<5:
            list_test_filename.append(filename)
            continue
        
        list_train_filename.append(filename)

    return list_train_filename,list_valid_filename,list_test_filename

def get_outer_set(list_train_filename,list_valid_filename,list_test_filename,no_outer):
    list_train,list_valid,list_test,list_outer_valid,list_outer_test=[],[],[],[],[]
    outer_subjects=[]
    while(len(outer_subjects)<no_outer):
        file_name=random.choice(list_test_filename)
        subject_name=file_name.split("/")[2].split("_")[0]
        if subject_name not in outer_subjects:
            outer_subjects.append(subject_name)
            
    old_new_name_map=dict()  
    for file_name in list_train_filename:
        subject_name=file_name.split("/")[2].split("_")[0]
        if subject_name not in outer_subjects:
            if subject_name not in old_new_name_map:
                old_new_name_map.update({subject_name:len(old_new_name_map)})
            
            list_train.append(file_name)

    for file_name in list_valid_filename:
        subject_name=file_name.split("/")[2].split("_")[0]
        if subject_name in outer_subjects:
            list_outer_valid.append(file_name)
        else:
            list_valid.append(file_name)

    for file_name in list_test_filename:
        subject_name=file_name.split("/")[2].split("_")[0]
        if subject_name in outer_subjects:
            list_outer_test.append(file_name)
        else:
            list_test.append(file_name)

    for subject_name in outer_subjects:
        old_new_name_map.update({subject_name:len(old_new_name_map)})

    return list_train, list_valid, list_test, list_outer_valid, list_outer_test, old_new_name_map